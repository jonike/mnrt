////////////////////////////////////////////////////////////////////////////////////////////////////
// MNRT License
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2010 Mathias Neumann, www.maneumann.com.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
//
// 3. Neither the name Mathias Neumann, nor the names of contributors may be 
//    used to endorse or promote products derived from this software without 
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MainFrame.h"
#include "CUDACanvas.h"
#include "StatsDialog.h"
#include "BenchResultDialog.h"
#include "ProgressListenerWx.h"
#include "MNRTSettingsWx.h"
#include "SceneConfigDialog.h"
#include "MNRTConfigDialog.h"
#include "AboutDialog.h"
#include <cuda_runtime.h>
#include <wx/config.h>
#include <wx/docview.h> // For wxFileHistory
#include <wx/tglbtn.h>
#include "../RTCore.h"
#include "../MNCudaMemPool.h"
#include "../MNStatContainer.h"
#include "../SceneConfig.h"

// DevIL includes to load/save images.
#include <IL/il.h>

// Image kernels.
extern "C"
void KernelIMGGenerateErrorImage(uchar4* d_ioImage, uchar4* d_inReference, uint numPixels, float fScale);
extern "C" 
void KernelIMGTestDiscrepancy(uint numSamples, uint screenW, uint screenH, uchar4* d_outScreenBuffer);

enum
{
	IDM_LOAD = 1,
	IDM_SAVEIMAGE,
	IDM_DISPLAYERROR,
	IDM_SHOWLOG,
	IDM_SHOWSTATS,
 
	IDM_BENCHMARK_KD,
	IDM_BENCHMARK_RT,

	IDM_TEST_DISCREPANCY,

	IDM_TOGGLE_RENDERMODE,
	IDM_TOGGLE_DYNAMICSCENE,
	IDM_CAM_AROUNDAT,
	IDM_CAM_WASD,
	IDM_TOGGLE_DIRECTRT,
	IDM_TOGGLE_SHADOWRAYS,
	IDM_TOGGLE_REFLECT,
	IDM_TOGGLE_TRANSMIT,

	IDM_VIEWMODE_RESULT,
	IDM_VIEWMODE_INITIALSAMPLES,
	IDM_VIEWMODE_CLUSTER,
	IDM_VIEWMODE_CLUSTERCTR,

	IDM_PMAP_DISABLED,
	IDM_PMAP_VISUALIZE,
	IDM_PMAP_FULLFG,
	IDM_PMAP_ADAPTIVEBESTFIT,
	IDM_PMAP_ADAPTIVEWANG,
	IDM_PMAP_TOGGLE_ICUT,
	IDM_PMAP_TOGGLE_USELEAFS,

	IDM_CONTACT,
	IDM_TOGGLE_ERRORCHECKS,

	IDB_LOADEXAMPLE,
	IDB_RENDERMODE,
	IDB_SCENECONFIG,

	IDT_MEMPOOL,
};

/// Special GUI log. Reports errors and fatal errors only.
class wxLogGuiError : public wxLogGui
{
public:
	virtual void DoLog(wxLogLevel level, const wxChar *msg, time_t timestamp)
	{
		if(level == wxLOG_Error || level == wxLOG_FatalError)
			wxLogGui::DoLog(level, msg, timestamp);
	}
};

BEGIN_EVENT_TABLE(MainFrame, wxFrame)
	EVT_CLOSE(MainFrame::OnClose)
	EVT_TIMER(IDT_MEMPOOL, MainFrame::OnTimer)

	EVT_MENU(IDM_LOAD, MainFrame::OnButtonLoad)
	EVT_MENU(IDM_SAVEIMAGE, MainFrame::OnSaveImage)
	EVT_MENU(IDM_DISPLAYERROR, MainFrame::OnDisplayError)
	EVT_MENU(IDM_SHOWLOG, MainFrame::OnShowLog)
	EVT_MENU(IDM_SHOWSTATS, MainFrame::OnShowStats)
	EVT_MENU_RANGE(wxID_FILE1, wxID_FILE9, MainFrame::OnMRUFile)
	EVT_MENU(wxID_EXIT, MainFrame::OnButtonExit)

	EVT_MENU(IDM_BENCHMARK_KD, MainFrame::OnBenchmark)
	EVT_MENU(IDM_BENCHMARK_RT, MainFrame::OnBenchmark)

	EVT_MENU(IDM_TEST_DISCREPANCY, MainFrame::OnTest)

	EVT_MENU(IDM_TOGGLE_RENDERMODE, MainFrame::OnButtonRenderMode)
	EVT_MENU(IDM_TOGGLE_DYNAMICSCENE, MainFrame::OnToggleDynamicScene)
	EVT_MENU(IDM_CAM_AROUNDAT, MainFrame::OnCameraMode)
	EVT_MENU(IDM_CAM_WASD, MainFrame::OnCameraMode)
	EVT_MENU(IDM_TOGGLE_DIRECTRT, MainFrame::OnToggleDirectRT)
	EVT_MENU(IDM_TOGGLE_SHADOWRAYS, MainFrame::OnToggleShadowRays)
	EVT_MENU(IDM_TOGGLE_REFLECT, MainFrame::OnToggleReflect)
	EVT_MENU(IDM_TOGGLE_TRANSMIT, MainFrame::OnToggleTransmit)
	EVT_MENU(wxID_PREFERENCES, MainFrame::OnSettings)

	EVT_MENU(IDM_VIEWMODE_RESULT, MainFrame::OnViewModeChange)
	EVT_MENU(IDM_VIEWMODE_INITIALSAMPLES, MainFrame::OnViewModeChange)
	EVT_MENU(IDM_VIEWMODE_CLUSTER, MainFrame::OnViewModeChange)
	EVT_MENU(IDM_VIEWMODE_CLUSTERCTR, MainFrame::OnViewModeChange)

	EVT_MENU(IDM_PMAP_DISABLED, MainFrame::OnPMModeChange)
	EVT_MENU(IDM_PMAP_VISUALIZE, MainFrame::OnPMModeChange)
	EVT_MENU(IDM_PMAP_FULLFG, MainFrame::OnPMModeChange)
	EVT_MENU(IDM_PMAP_ADAPTIVEBESTFIT, MainFrame::OnPMModeChange)
	EVT_MENU(IDM_PMAP_ADAPTIVEWANG, MainFrame::OnPMModeChange)
	EVT_MENU(IDM_PMAP_TOGGLE_ICUT, MainFrame::OnPMToggleICut)
	EVT_MENU(IDM_PMAP_TOGGLE_USELEAFS, MainFrame::OnPMToggleICutUseLeafs)

	EVT_MENU(wxID_HELP, MainFrame::OnHelp)
	EVT_MENU(IDM_CONTACT, MainFrame::OnContact)
	EVT_MENU(IDM_TOGGLE_ERRORCHECKS, MainFrame::OnToggleErrorChecks)
	EVT_MENU(wxID_ABOUT, MainFrame::OnAbout)

	EVT_BUTTON(IDB_LOADEXAMPLE, MainFrame::OnButtonLoadExample)
	EVT_BUTTON(IDB_SCENECONFIG, MainFrame::OnButtonSceneConfig)
	EVT_TOGGLEBUTTON(IDB_RENDERMODE, MainFrame::OnButtonRenderMode)
END_EVENT_TABLE()


MainFrame::MainFrame(const wxString& title, const wxSize& sizeScreen, bool bProfile /*= false*/)
			: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxDefaultSize)
{
	m_pSC = NULL;
	m_pRTCore = NULL;
	m_pSettings = new MNRTSettingsWx();
	m_pSettings->Load();
	m_bSingleFrame = true;
	m_bSingleDone = false;
	m_bCanRender = false;
	m_bProfile = false;
	m_strRefImage = wxEmptyString;
	m_dErrorScale = 4.;
	m_RenderCommand = -1;
	
	wxIcon icon(wxT("Media/MNRT.ico"), wxBITMAP_TYPE_ICO, 32, 32);
	SetIcon(icon);

	MNStatContainer& cont = MNStatContainer::GetInstance();
	cont.SetTimersEnabled(false);

	wxLogChain* pLogChain = new wxLogChain(new wxLogGuiError());

	m_pWndLog = new wxLogWindow(this, "Log", false, true);
	m_pWndLog->GetFrame()->SetIcon(icon);
	wxWindow* pTextCtrl = m_pWndLog->GetFrame()->GetChildren().GetFirst()->GetData();
	pTextCtrl->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
	m_pWndLog->GetFrame()->SetSize(650, 500);

	// Create/load file history for 6 files.
	m_pFileHistory = new wxFileHistory(6);
	m_pFileHistory->Load(*wxConfig::Get());

	// Check for CUDA.
	if(!CheckForCUDA())
		MNFatal("Failed to find compatible CUDA devices.");

	CreateMenuBar();

	m_pCanvas = new CUDACanvas(this, wxID_ANY, wxDefaultPosition, sizeScreen);

	CreateControlBar();

	m_pSizerBox = new wxBoxSizer(wxVERTICAL);
	m_pSizerBox->AddStretchSpacer();
	m_pSizerBox->Add(m_pCanvas, wxSizerFlags().Center());
	m_pSizerBox->AddStretchSpacer();
	m_pSizerBox->Add(m_pPanelCtrls, wxSizerFlags().Expand());

    wxStatusBar* pStatus = CreateStatusBar(3);
	int widths[3] = {-1, 100, 100}; // Negative values for variable widths
	pStatus->SetStatusWidths(3, widths);

	// Memory pool update timer.
	m_pTimerPool = new wxTimer(this, IDT_MEMPOOL);
	m_pTimerPool->Start(1000);

	m_bProfile = bProfile;
	if(m_bProfile)
	{
		// Load from file, however do not initialize core as we might not have the canvas
		// visible, hence CUDA might be uninitialized.
		if(!LoadFromFile(_("")))
			MNFatal("Failed to load default scene for profiling");
		m_bSingleFrame = true;
	}

	SetSizer(m_pSizerBox);
	m_pSizerBox->SetSizeHints(this);
	m_pWndLog->Show();
	ActivateUpdateLoop(true);
	m_bCanRender = true;
	UpdateControls();
}

MainFrame::~MainFrame(void)
{
}

bool MainFrame::CheckForCUDA()
{
	MNMessage("Looking for CUDA devices...");

	// Get CUDA device count.
	int nDevCount = 0;
	cudaGetDeviceCount(&nDevCount);
	if(nDevCount > 0)
		MNMessage("Found %d CUDA-enabled device(s):", nDevCount);
	else
	{
		MNFatal("No CUDA-enabled devices found.");
		return false;
	}

	// Print device information.
	int nGpuArchCoresPerSM[] = { -1, 8, 32 };
	bool bHasCompatible = false;
	int idToUse = 0, bestPower = 0;
	for(int d=0; d<nDevCount; d++)
	{
		cudaDeviceProp props;
		if(cudaGetDeviceProperties(&props, d) != cudaSuccess)
			MNMessage("%d: Failed to get device properties", d);
		else
		{
			int cores = props.multiProcessorCount;
			if(props.major <= 2)
				cores = nGpuArchCoresPerSM[props.major]*props.multiProcessorCount;
			MNMessage("%d: %s (%d MPs, %d cores, %.0f MB global mem, %d.%d compute cap.)", 
				d, props.name, props.multiProcessorCount, cores, props.totalGlobalMem/(1024.f*1024.f),
				props.major, props.minor);
			bool isCompatible = (props.major >= 2 || props.minor >= 1);
			bHasCompatible |= isCompatible;
			int computePower = cores * props.clockRate;
			if(isCompatible && computePower > bestPower)
			{
				idToUse = d;
				bestPower = computePower;
			}

		}
	}

	if(bHasCompatible)
	{
		MNMessage("-> Selecting CUDA device %d.", idToUse);
		m_nCUDADeviceID = idToUse;
		return true;
	}
	else
	{
		MNFatal("Failed to detect compatible CUDA devices. Need compute cabability 1.1 or better.");
		return false;
	}
}

void MainFrame::CreateMenuBar()
{
	m_pMenuMRU = new wxMenu();
	m_pFileHistory->UseMenu(m_pMenuMRU);
	m_pFileHistory->AddFilesToMenu(m_pMenuMRU); // Need to initialize the menu.

	wxMenu* pMenuBench = new wxMenu();
	pMenuBench->Append(IDM_BENCHMARK_KD, _("&kd-Tree Construction"));
	pMenuBench->Append(IDM_BENCHMARK_RT, _("&Ray Tracing"));
	
	wxMenu* pMenuTest = new wxMenu();
	pMenuTest->Append(IDM_TEST_DISCREPANCY, _("&Discrepancy"), _("Visual test of discrepancy of Halton sequence"));

	m_pMenuFile = new wxMenu();
	m_pMenuFile->Append(IDM_LOAD, _("&Load Scene"), _("Load a scene from file"));
	m_pMenuFile->AppendSeparator();
	m_pMenuFile->Append(IDM_SAVEIMAGE, _("Save &Image"), _("Saves current image to file"));
	m_pMenuFile->Append(IDM_DISPLAYERROR, _("Display &Error Image"), _("Shows an error image according to chosen settings"));
	m_pMenuFile->AppendSeparator();
	m_pMenuFile->AppendSubMenu(pMenuBench, _("&Benchmark"));
	m_pMenuFile->AppendSeparator();
	m_pMenuFile->AppendSubMenu(pMenuTest, _("&Test"));
	m_pMenuFile->AppendSeparator();
	m_pMenuFile->Append(IDM_SHOWLOG, _("View Lo&g"), _T("Views the applications log"));
	m_pMenuFile->Append(IDM_SHOWSTATS, _("View &Statistics"), _T("Views execution statistics"));
	m_pMenuFile->AppendSeparator();
	m_pMenuFile->AppendSubMenu(m_pMenuMRU, _("&Recent Models"));
	m_pMenuFile->AppendSeparator();
    m_pMenuFile->Append(wxID_EXIT);

	m_pMenuSettings = new wxMenu();
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_RENDERMODE, _("&Single Frame"), _("Render mode: single frame or frame sequence"));
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_DYNAMICSCENE, _("D&ynamic Scene"), _("Whether to rebuild kd-trees every frame"));
	m_pMenuSettings->AppendSeparator();
	m_pMenuSettings->AppendRadioItem(IDM_CAM_AROUNDAT, _("Camera: Rotate Around &At"), _("Camera rotation around look at point using mouse"));
	m_pMenuSettings->AppendRadioItem(IDM_CAM_WASD, _("Camera: &WASD"), _T("WASD camera movement and mouse rotation around eye point"));
	m_pMenuSettings->AppendSeparator();
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_DIRECTRT, _("&Direct Lighting"), _("Direct ligthing using ray tracing"));
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_SHADOWRAYS, _("Tra&ce Shadow Rays"), _("Trace shadow rays for direct lighting"));
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_REFLECT, _("Specular &Reflection"), _("Specular reflection during ray tracing"));
	m_pMenuSettings->AppendCheckItem(IDM_TOGGLE_TRANSMIT, _("Specular &Transmission"), _("Specular transmission during ray tracing"));
	m_pMenuSettings->AppendSeparator();
	m_pMenuSettings->Append(wxID_PREFERENCES);
 
	m_pMenuViewMode = new wxMenu();
	m_pMenuViewMode->AppendRadioItem(IDM_VIEWMODE_RESULT, _("&Result"), _("Show usual algorithm result image"));
	m_pMenuViewMode->AppendRadioItem(IDM_VIEWMODE_INITIALSAMPLES, _("&Initial Samples"), _("Show initial samples after adaptive sample seeding"));
	m_pMenuViewMode->AppendRadioItem(IDM_VIEWMODE_CLUSTER, _("&Clusters"), _("Show k-means clusters using discrete grey scale regions"));
	m_pMenuViewMode->AppendRadioItem(IDM_VIEWMODE_CLUSTERCTR, _("Cluster C&enters"), _("Show k-means cluster centers on ray traced image"));

	m_pMenuPMode = new wxMenu();
	m_pMenuPMode->AppendRadioItem(IDM_PMAP_DISABLED, _("Disabled"), _("No photon mapping for indirect illumination"));
	m_pMenuPMode->AppendRadioItem(IDM_PMAP_VISUALIZE, _("Visualize Photons"), _("Visualize photons on ray traced image"));
	m_pMenuPMode->AppendRadioItem(IDM_PMAP_FULLFG, _("Full Final Gathering"), _("Perform unaccelerated final gathering"));
	m_pMenuPMode->AppendRadioItem(IDM_PMAP_ADAPTIVEBESTFIT, _("Adaptive FG + Best Fit"), _("Adaptive sampled final gathering with best fit assignment"));
	m_pMenuPMode->AppendRadioItem(IDM_PMAP_ADAPTIVEWANG, _("Adaptive FG + Interpolation"), _("Adaptive sampled final gathering with interpolation"));
	m_pMenuPMode->AppendSeparator();
	m_pMenuPMode->AppendCheckItem(IDM_PMAP_TOGGLE_ICUT, _("Illumination Cuts"), _("Use illumination cuts for final gathering"));
	m_pMenuPMode->AppendCheckItem(IDM_PMAP_TOGGLE_USELEAFS, _("ICUT: Use Leafs (Simplification)"), _("Just use all leafs as cut nodes to avoid cut computation"));

	m_pMenuHelp = new wxMenu();
	m_pMenuHelp->Append(wxID_HELP);
	m_pMenuHelp->Append(IDM_CONTACT, _("Contact"), _("Visit my website to contact me"));
	m_pMenuHelp->AppendSeparator();
	m_pMenuHelp->AppendCheckItem(IDM_TOGGLE_ERRORCHECKS, _("Enable Error Checks"), _("Performs an error check after each kernel"));
	m_pMenuHelp->AppendSeparator();
	m_pMenuHelp->Append(wxID_ABOUT);

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(m_pMenuFile, _("&File"));
	menuBar->Append(m_pMenuSettings, _("&Settings"));
	menuBar->Append(m_pMenuViewMode, _("&View Mode"));
	menuBar->Append(m_pMenuPMode, _("&Photon Mapping"));
	menuBar->Append(m_pMenuHelp, _("&Help"));
    SetMenuBar(menuBar);
}

void MainFrame::CreateControlBar()
{
	m_pPanelCtrls = new wxPanel(this, wxID_ANY, wxPoint(0, 512), wxSize(512, 30));

	wxBoxSizer* pBox = new wxBoxSizer(wxHORIZONTAL);

	wxButton* pButInit = new wxButton(m_pPanelCtrls, IDB_LOADEXAMPLE, "Load Example");
	pButInit->SetToolTip(_("Loads an example scene"));

	m_pButSceneConfig = new wxButton(m_pPanelCtrls, IDB_SCENECONFIG, "Scene Configuration");
	m_pButSceneConfig->SetToolTip(_("Show scene specific configuration window"));

	m_pButRenderMode = new wxToggleButton(m_pPanelCtrls, IDB_RENDERMODE, "Single Frame");
	m_pButRenderMode->SetValue(true);
	m_pButRenderMode->SetToolTip(_("Render mode: single frame or frame sequence"));

	pBox->AddStretchSpacer();
	pBox->Add(pButInit, wxSizerFlags().DoubleBorder(wxALL));
	pBox->AddSpacer(50);
	pBox->Add(m_pButSceneConfig, wxSizerFlags().DoubleBorder(wxALL));
	pBox->Add(m_pButRenderMode, wxSizerFlags().DoubleBorder(wxALL));
	pBox->AddStretchSpacer();

	m_pPanelCtrls->SetSizer(pBox);
}

void MainFrame::UpdateControls()
{
	m_pMenuFile->Enable(IDM_SAVEIMAGE, m_bSingleFrame && m_pRTCore);
	m_pMenuFile->Enable(IDM_DISPLAYERROR, m_bSingleFrame && m_pRTCore);

	// Update menu states.
	m_pMenuSettings->Check(IDM_TOGGLE_RENDERMODE, m_bSingleFrame);
	m_pMenuSettings->Check(IDM_CAM_AROUNDAT,	m_pCanvas->GetCameraMode() == CUDACanvas::CamMode_RotAroundLookAt);
	m_pMenuSettings->Check(IDM_CAM_WASD,		m_pCanvas->GetCameraMode() == CUDACanvas::CamMode_WASD);
	m_pMenuSettings->Check(IDM_TOGGLE_DIRECTRT, m_pSettings->GetEnableDirectRT());
	m_pMenuSettings->Check(IDM_TOGGLE_SHADOWRAYS, m_pSettings->GetEnableShadowRays());
	m_pMenuSettings->Check(IDM_TOGGLE_REFLECT, m_pSettings->GetEnableSpecReflect());
	m_pMenuSettings->Check(IDM_TOGGLE_TRANSMIT, m_pSettings->GetEnableSpecTransmit());

	m_pMenuViewMode->Check(IDM_VIEWMODE_RESULT,		m_pSettings->GetViewMode() == MNRTView_Result);
	m_pMenuViewMode->Check(IDM_VIEWMODE_INITIALSAMPLES,	m_pSettings->GetViewMode() == MNRTView_InitialSamples);
	m_pMenuViewMode->Check(IDM_VIEWMODE_CLUSTER,	m_pSettings->GetViewMode() == MNRTView_Cluster);
	m_pMenuViewMode->Check(IDM_VIEWMODE_CLUSTERCTR,	m_pSettings->GetViewMode() == MNRTView_ClusterCenters);

	PhotonMapMode pmMode = m_pSettings->GetPhotonMapMode();
	m_pMenuPMode->Check(IDM_PMAP_DISABLED,	      pmMode == PhotonMap_Disabled);
	m_pMenuPMode->Check(IDM_PMAP_VISUALIZE,	      pmMode == PhotonMap_Visualize);
	m_pMenuPMode->Check(IDM_PMAP_FULLFG,	      pmMode == PhotonMap_FullFinalGather);
	m_pMenuPMode->Check(IDM_PMAP_ADAPTIVEBESTFIT, pmMode == PhotonMap_AdaptiveSamplesBestFit);
	m_pMenuPMode->Check(IDM_PMAP_ADAPTIVEWANG,	  pmMode == PhotonMap_AdaptiveSamplesWang);
	m_pMenuPMode->Check(IDM_PMAP_TOGGLE_ICUT, m_pSettings->GetUseIllumCuts());
	m_pMenuPMode->Check(IDM_PMAP_TOGGLE_USELEAFS, m_pSettings->GetICutUseLeafs());

	m_pMenuHelp->Check(IDM_TOGGLE_ERRORCHECKS, mncudaIsErrorChecking());

	m_pButSceneConfig->Enable(m_pSC != NULL);
	m_pButRenderMode->SetValue(m_bSingleFrame);
}

void MainFrame::OnClose(wxCloseEvent& event)
{
	SAFE_DELETE(m_pTimerPool);
	ActivateUpdateLoop(false);

	SAFE_DELETE(m_pRTCore);
	SAFE_DELETE(m_pSC);
	m_pCanvas->DestroyGLandCUDA();

	// Save file history to configuration.
	m_pFileHistory->Save(*wxConfig::Get());
	SAFE_DELETE(m_pFileHistory);

	// Save settings in configuration.
	m_pSettings->Save();
	SAFE_DELETE(m_pSettings);

	event.Skip(); // Continue processing as we want to close.
}

bool MainFrame::ReinitializeCore(bool bUpdate/* = true*/)
{
	if(!m_pSC)
		return false;
	wxSize szScreen = m_pCanvas->GetScreenSize();

	// Kill old instance if required.
	SAFE_DELETE(m_pRTCore);

	// Recreate...
	m_pRTCore = new RTCore(m_pSettings);
	if(!m_pRTCore->Initialize(szScreen.GetWidth(), m_pSC))
		return false;
	if(m_bCanRender)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();

	return true;
}

bool MainFrame::WantUpdate() const
{
	if(!m_bCanRender)
		return false;

	if(m_RenderCommand == IDM_TEST_DISCREPANCY)
		return true;
	else if(m_RenderCommand == IDM_DISPLAYERROR)
		return m_pRTCore != NULL;
	else if(m_bSingleFrame && !m_bSingleDone)
		return m_pRTCore != NULL; // Need to compute single frame.
	else if(!m_bSingleFrame)
		return m_pRTCore != NULL; // Always refresh.
	else
		return false;
}

void MainFrame::Render(uchar4* d_buffer)
{
	wxSize szScreen = m_pCanvas->GetScreenSize();
	int oldRenderCommand = m_RenderCommand;
	m_RenderCommand = -1;

	if(oldRenderCommand == -1)
	{
		if(!m_pRTCore->RenderScene(d_buffer))
		{
			MNError("Rendering failed.");
			UnloadScene();
		}
	}
	else if(oldRenderCommand == IDM_TEST_DISCREPANCY)
	{
		KernelIMGTestDiscrepancy(1000000, szScreen.GetWidth(), szScreen.GetHeight(), d_buffer);
	}
	else if(oldRenderCommand == IDM_DISPLAYERROR)
	{
		ILuint handleImage, srcW, srcH;

		ilGenImages(1, &handleImage);
		ilBindImage(handleImage);
		if(!ilLoadImage(m_strRefImage.c_str()))
		{
			MNError("Failed to load reference image.");
			return;
		}
		// Read out image width/height.
		srcW = ilGetInteger(IL_IMAGE_WIDTH);
		srcH = ilGetInteger(IL_IMAGE_HEIGHT);

		if(srcW != szScreen.GetWidth() || srcH != szScreen.GetHeight())
		{
			MNError("Illegal reference image format. Need %d x %d pixels.", 
				szScreen.GetWidth(), szScreen.GetHeight());
			ilDeleteImages(1, &handleImage);
			return;
		}
	
		// We need the image data in IL_RGBA format, IL_UNSIGNED_BYTE type.
		uchar4* h_image = new uchar4[srcW*srcH];
		ilCopyPixels(0, 0, 0, srcW, srcH, 1, 
			IL_RGBA, IL_UNSIGNED_BYTE, (void*)h_image);

		ilDeleteImages(1, &handleImage);

		MNCudaMemory<uchar4> d_imgReference(srcW*srcH);
		mncudaSafeCallNoSync(cudaMemcpy(d_imgReference, h_image, srcW*srcH*sizeof(uchar4), 
			cudaMemcpyHostToDevice));

		// Now read current image and copy it to the screen buffer.
		m_pCanvas->GetCurrentImage(h_image);
		mncudaSafeCallNoSync(cudaMemcpy(d_buffer, h_image, srcW*srcH*sizeof(uchar4), 
			cudaMemcpyHostToDevice));

		SAFE_DELETE_ARRAY(h_image);

		KernelIMGGenerateErrorImage(d_buffer, d_imgReference, srcW*srcH, m_dErrorScale);
	}

	m_bSingleDone = true;
}

void MainFrame::OnButtonLoadExample(wxCommandEvent& event)
{
	if(LoadFromFile(_("")))
		ReinitializeCore();
}

void MainFrame::OnButtonSceneConfig(wxCommandEvent& event)
{
	// wxDialogs can be created on stack.
	SceneConfigDialog dlg(this, m_pSC);
	dlg.CenterOnParent();
	if(wxID_OK == dlg.ShowModal() && dlg.IsConfigModified())
		ReinitializeCore();	
}

bool MainFrame::LoadFromFile(const wxString& strModelFile)
{
	// First be sure old scene is unloaded.
	UnloadScene();

	m_pSC = new SceneConfig();
	if(!m_pSC->LoadFromFile(std::string(strModelFile.c_str())))
	{
		MNError("Loading scene from file failed.\nScene: \"%s\".", strModelFile.c_str());
		SAFE_DELETE(m_pSC);
		return false;
	}

	// Store file information in config. Also update the menu.
	if(!strModelFile.IsEmpty())
		m_pFileHistory->AddFileToHistory(strModelFile);
	return true;
}

void MainFrame::UnloadScene()
{
	if(!m_pSC)
		return; // Nothing to unload.

	// Also destroy core.
	SAFE_DELETE(m_pRTCore);
	SAFE_DELETE(m_pSC);
}

void MainFrame::OnButtonLoad(wxCommandEvent& event)
{
	wxString strWildcard;
	strWildcard += _("Model Files|*.obj;*.3ds;*.lwo;*.lws;*.ply;*.dae;*.xml;*.dxf;*.nff;*.smd;*.vta;");
	strWildcard += _("*.md1;*.md2;*.md3;*.md5mesh;*.x;*.raw;*.ac;*.irrmesh;*.irr;*.mdl;*.mesh.xml;*.ms3d|");
	strWildcard += _("Wavefront Object (*.obj)|*.obj|");
	strWildcard += _("3D Studio Max 3DS (*.3ds)|*.3ds|");
	strWildcard += _("LightWave (*.lwo,*.lws)|*.lwo;*.lws|");
	strWildcard += _("Stanford Polygon Library (*.ply)|*.ply|");
	strWildcard += _("Collada (*.dae,*.xml)|*.dae;*.xml|");
	strWildcard += _("AutoCAD DXF (*.dxf)|*.dxf|");
	strWildcard += _("Neutral File Format (*.nff)|*.nff|");
	strWildcard += _("Valve Model (*.smd,*.vta)|*.smd;*.vta|");
	strWildcard += _("Quake Model (*.md1,*.md2,*.md3)|*.md1;*.md2;*.md3|");
	strWildcard += _("Doom 3 (*.md5mesh)|*.md5mesh|");
	strWildcard += _("DirectX X (*.x)|*.x|");
	strWildcard += _("Raw Triangles (*.raw)|*.raw|");
	strWildcard += _("AC3D (*.ac)|*.ac|");
	strWildcard += _("Irrlicht (*.irrmesh,*.irr)|*.irrmesh;*.irr|");
	strWildcard += _("3D GameStudio Model (*.mdl)|*.mdl|");
	strWildcard += _("Ogre (*.mesh.xml)|*.mesh.xml|");
	strWildcard += _("Milkshape 3D (*.ms3d)|*.ms3d|");
	strWildcard += _("All files (*.*)|*.*");

	wxFileDialog* pFD = new wxFileDialog(this, _("Select a model to load!"), wxEmptyString, wxEmptyString,
							strWildcard, wxFD_OPEN);
	if(pFD->ShowModal() == wxID_OK)
	{
		if(LoadFromFile(pFD->GetPath()))
			ReinitializeCore();
	}
}

void MainFrame::OnButtonRenderMode(wxCommandEvent& event)
{
	if(event.GetId() == IDB_RENDERMODE)
		m_bSingleFrame = m_pButRenderMode->GetValue();
	else if(event.GetId() == IDM_TOGGLE_RENDERMODE)
		m_bSingleFrame = !m_pButRenderMode->GetValue();
	
	m_bSingleDone = false;
	m_pCanvas->Refresh();
	UpdateControls();
}

void MainFrame::OnSaveImage(wxCommandEvent& event)
{
	if(!m_pCanvas || !m_bSingleFrame)
		return;

	wxString strWildcard;
	strWildcard += _("Portable Network Graphics (*.png)|*.png|");
	strWildcard += _("Windows Bitmap (*.bmp)|*.bmp|");
	strWildcard += _("Jpeg (*.jpg)|*.jpg|");
	strWildcard += _("All files (*.*)|*.*");

	// Show file dialog to get target file name.
	wxFileDialog* pFD = new wxFileDialog(this, _("Select destination for image"), wxEmptyString, wxEmptyString,
		strWildcard, wxFD_SAVE|wxFD_DEFAULT_STYLE|wxFD_OVERWRITE_PROMPT);
	if(pFD->ShowModal() != wxID_OK)
		return;

	ILuint handleImage;

	// Generate and bind image to handle.
	ilGenImages(1, &handleImage);
	ilBindImage(handleImage);

	// Set pixels from screen buffer.
	wxSize szScreen = m_pCanvas->GetScreenSize();
	uchar4* pData = new uchar4[szScreen.GetWidth()*szScreen.GetHeight()];
	m_pCanvas->GetCurrentImage(pData);
	ilTexImage(szScreen.GetWidth(), szScreen.GetHeight(), 1, 4,
		IL_RGBA, IL_UNSIGNED_BYTE, pData);
	SAFE_DELETE_ARRAY(pData);

	// Remove alpha channel.
	if(!ilConvertImage(IL_RGB, IL_UNSIGNED_BYTE))
	{
		MNError("Failed to save image. Conversion failed.");
		ilDeleteImages(1, &handleImage);
		return;
	}

	// Save image to chosen file.
	ilEnable(IL_FILE_OVERWRITE);
	if(!ilSaveImage(pFD->GetPath().c_str()))
		MNError("Failed to save image: %s.", pFD->GetPath().c_str());

	ilDeleteImages(1, &handleImage);
}

void MainFrame::OnDisplayError(wxCommandEvent& event)
{
	if(!m_pRTCore || !m_pCanvas || !m_bSingleFrame || m_RenderCommand != -1)
		return;

	// First, get the reference image path.
	wxFileDialog* pFD = new wxFileDialog(this, _("Select reference image"), wxEmptyString, m_strRefImage,
			_("Portable Network Graphics (*.png)|*.png|Windows Bitmap (*.bmp)|*.bmp"), wxFD_OPEN);
	if(pFD->ShowModal() != wxID_OK)
		return;
	m_strRefImage = pFD->GetPath();

	// Now request scale factor.
	wxTextEntryDialog* pTED = new wxTextEntryDialog(this, _("Enter scale factor for the absolute error."),
		_("Enter scale factor"), _("4.0"));
	if(pTED->ShowModal() != wxID_OK)
		return;

	if(!pTED->GetValue().ToDouble(&m_dErrorScale))
	{
		MNError("Failed to parse error scale value.");
		return;
	}

	m_RenderCommand = IDM_DISPLAYERROR;
	m_pCanvas->Refresh();
}

void MainFrame::OnMRUFile(wxCommandEvent& event)
{
	MNAssert(m_pFileHistory);
	wxString strFile(m_pFileHistory->GetHistoryFile(event.GetId() - wxID_FILE1));
	if(LoadFromFile(strFile))
		ReinitializeCore();
}

void MainFrame::OnButtonExit(wxCommandEvent& event)
{
    Close(true);
}

void MainFrame::OnShowLog(wxCommandEvent& event)
{
	m_pWndLog->Show();
	m_pWndLog->GetFrame()->SetFocus();
}

void MainFrame::OnShowStats(wxCommandEvent& event)
{
	StatsDialog* pWndStats = new StatsDialog(this);
	pWndStats->CenterOnParent();
	pWndStats->ShowModal();
}

void MainFrame::OnBenchmark(wxCommandEvent& event)
{
	if(event.GetId() == IDM_BENCHMARK_KD)
	{
		m_bCanRender = false;

		uint warmup = 10;
		uint runs = 200;
		float tTotal, tAvg;

		// Initialize core, but do not update view.
		wxString strTestScene = _("Test Scenes/MNSimpleDragon.obj");
		if(!LoadFromFile(strTestScene))
		{
			MNError("Benchmark failed: Test scene \"%s\" not found.", strTestScene.c_str());
			m_bCanRender = true;
			return;
		}
		ReinitializeCore(false);

		ProgressListenerWx progLst(this, _("kd-Tree Construction Benchmark"), _("Benchmark in progress..."));

		bool bAborted = m_pRTCore->BenchmarkKDTreeCon(warmup, runs, &tTotal, &tAvg, &progLst);
		SAFE_DELETE(m_pRTCore);

		if(bAborted)
			return;

		wxString strRes;
		strRes += wxString::Format("Warmup Runs:  %5d\n", warmup);
		strRes += wxString::Format("Runs:	      %5d\n", runs);
		strRes += wxString("\n");
		strRes += wxString::Format("Total Time:   %9.3f ms\n", tTotal);
		strRes += wxString::Format("Average Time: %9.3f ms\n", tAvg);

		BenchResultDialog* dlgRes = new BenchResultDialog(this, _("kd-Tree Construction Benchmark"), strRes);
		dlgRes->CenterOnParent();
		dlgRes->ShowModal();

		m_bCanRender = true;
		m_pCanvas->Refresh();
		UpdateControls();
	}
	else if(event.GetId() == IDM_BENCHMARK_RT)
	{
		wxMessageBox(_("Not available yet."));
	}
}

void MainFrame::OnTest(wxCommandEvent& event)
{
	if(event.GetId() == IDM_TEST_DISCREPANCY)
	{
		m_RenderCommand = IDM_TEST_DISCREPANCY;
		m_bSingleFrame = true;
		m_pCanvas->Refresh();
	}
}

void MainFrame::OnCameraMode(wxCommandEvent& event)
{
	if(event.GetId() == IDM_CAM_AROUNDAT)
		m_pCanvas->SetCameraMode(CUDACanvas::CamMode_RotAroundLookAt);
	else if(event.GetId() == IDM_CAM_WASD)
		m_pCanvas->SetCameraMode(CUDACanvas::CamMode_WASD);
	UpdateControls();
}

void MainFrame::OnToggleDynamicScene(wxCommandEvent& event)
{
	m_pSettings->SetDynamicScene(!m_pSettings->GetDynamicScene());
}

void MainFrame::OnToggleDirectRT(wxCommandEvent& event)
{
	m_pSettings->SetEnableDirectRT(!m_pSettings->GetEnableDirectRT());

	// Recompute picture for mode change.
	if(m_pRTCore)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();
}

void MainFrame::OnToggleShadowRays(wxCommandEvent& event)
{
	m_pSettings->SetEnableShadowRays(!m_pSettings->GetEnableShadowRays());

	// Recompute picture for mode change.
	if(m_pRTCore)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();
}

void MainFrame::OnToggleReflect(wxCommandEvent& event)
{
	m_pSettings->SetEnableSpecReflect(!m_pSettings->GetEnableSpecReflect());

	// Recompute picture for mode change.
	if(m_pRTCore)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();
}

void MainFrame::OnToggleTransmit(wxCommandEvent& event)
{
	m_pSettings->SetEnableSpecTransmit(!m_pSettings->GetEnableSpecTransmit());

	// Recompute picture for mode change.
	if(m_pRTCore)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();
}

void MainFrame::OnSettings(wxCommandEvent& event)
{
	MNRTConfigDialog dlg(this, m_pSettings);
	dlg.CenterOnParent();
	if(wxID_OK == dlg.ShowModal() && dlg.IsConfigModified())
		ReinitializeCore();
}

void MainFrame::OnViewModeChange(wxCommandEvent& event)
{
	switch(event.GetId())
	{
	case IDM_VIEWMODE_RESULT:
		m_pSettings->SetViewMode(MNRTView_Result);
		break;
	case IDM_VIEWMODE_INITIALSAMPLES:
		m_pSettings->SetViewMode(MNRTView_InitialSamples);
		break;
	case IDM_VIEWMODE_CLUSTER:
		m_pSettings->SetViewMode(MNRTView_Cluster);
		break;
	case IDM_VIEWMODE_CLUSTERCTR:
		m_pSettings->SetViewMode(MNRTView_ClusterCenters);
		break;
	default:
		MNError("Unknown view mode.");
		return;
	}

	// Recompute picture for mode change.
	if(m_pRTCore)
	{
		m_bSingleDone = false;
		m_pCanvas->Refresh();
	}
	UpdateControls();
}

void MainFrame::OnPMModeChange(wxCommandEvent& event)
{
	switch(event.GetId())
	{
	case IDM_PMAP_DISABLED:
		m_pSettings->SetPhotonMapMode(PhotonMap_Disabled);
		break;
	case IDM_PMAP_VISUALIZE:
		m_pSettings->SetPhotonMapMode(PhotonMap_Visualize);
		break;
	case IDM_PMAP_FULLFG:
		m_pSettings->SetPhotonMapMode(PhotonMap_FullFinalGather);
		break;
	case IDM_PMAP_ADAPTIVEBESTFIT:
		m_pSettings->SetPhotonMapMode(PhotonMap_AdaptiveSamplesBestFit);
		break;
	case IDM_PMAP_ADAPTIVEWANG:
		m_pSettings->SetPhotonMapMode(PhotonMap_AdaptiveSamplesWang);
		break;
	default:
		MNError("Unknown photon mapping mode.");
		return;
	}

	ReinitializeCore();
}

void MainFrame::OnPMToggleICut(wxCommandEvent& event)
{
	m_pSettings->SetUseIllumCuts(!m_pSettings->GetUseIllumCuts());

	// Reinit core to recompute photon maps.
	ReinitializeCore();
}

void MainFrame::OnPMToggleICutUseLeafs(wxCommandEvent& event)
{
	m_pSettings->SetICutUseLeafs(!m_pSettings->GetICutUseLeafs());

	// Reinit core to recompute photon maps.
	ReinitializeCore();
}

void MainFrame::OnHelp(wxCommandEvent& event)
{
	wxString strLicenses = wxString("file://") + wxGetCwd() + wxString("/Help/MNRT_Documentation.html");
	wxLaunchDefaultBrowser(strLicenses);
}

void MainFrame::OnContact(wxCommandEvent& event)
{
	wxLaunchDefaultBrowser(_("http://www.maneumann.com"));
}

void MainFrame::OnToggleErrorChecks(wxCommandEvent& event)
{
	mncudaEnableErrorChecks(!mncudaIsErrorChecking());
	UpdateControls();
}

void MainFrame::OnAbout(wxCommandEvent& event)
{
	AboutDialog dlg(this);
	dlg.CenterOnParent();
	dlg.ShowModal();
}

void MainFrame::ActivateUpdateLoop(bool bActivate)
{
    if(bActivate && !m_bUpdateLoopActive)
    {
        Connect(wxID_ANY, wxEVT_IDLE, wxIdleEventHandler(MainFrame::OnIdle));
        m_bUpdateLoopActive = true;
    }
    else if(!bActivate && m_bUpdateLoopActive)
    {
        Disconnect(wxEVT_IDLE, wxIdleEventHandler(MainFrame::OnIdle));
        m_bUpdateLoopActive = false;
    }
}


void MainFrame::OnIdle(wxIdleEvent& event)
{
	if(m_bUpdateLoopActive)
	{
		if(m_bProfile)
		{
			if(!m_pRTCore && m_pCanvas->IsCUDAInited())
				ReinitializeCore();
			else if(m_bSingleDone)
				Close(true);
		}
		else
		{
			if(!m_bSingleFrame)
				m_pCanvas->Refresh();
			UpdateControls();
		}

        event.RequestMore();
    }
}

void MainFrame::OnTimer(wxTimerEvent & event)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	pool.UpdatePool();

	// Update memory usage.
	float mb = pool.GetAssignedSize() / (1024.0*1024.0);
	wxString strMem = wxString::Format("Used: %.1lf MB", mb);
	SetStatusText(strMem, 1);

	// Update timings.
	float fps, last_s;
	m_pCanvas->GetTimings(&fps, &last_s);

	wxString strTime;
	if(m_bSingleFrame)
	{
		if(last_s == FLT_MAX)
			strTime = wxString("N/A");
		else
			strTime = wxString::Format("%.2f s (Last)", last_s);
	}
	else
	{
		if(fps == FLT_MAX)
			strTime = wxString("N/A");
		else
			strTime = wxString::Format("%3.1f fps", fps);
	}
	SetStatusText(strTime, 2);
}

CameraModel* MainFrame::GetCamera()
{
	if(m_pSC)
		return m_pSC->GetCamera();
	else
		return NULL;
}