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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file	wxWidgets\MainFrame.h
///
/// \brief	Declares the MainFrame class.
/// \author	Mathias Neumann
/// \date	03.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_MAINFRAME_H__
#define __MN_MAINFRAME_H__

#pragma once

#include <wx/wx.h>
#include <vector_types.h>

class RTCore;
class CUDACanvas;
class wxFileHistory;
class wxToggleButton;
class MNRTSettingsWx;
class SceneConfig;
class CameraModel;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MainFrame
///
/// \brief	Main frame for MNRT. Provides user interface.
///
/// \author	Mathias Neumann
/// \date	03.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MainFrame : public wxFrame
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MainFrame(const wxString& title, const wxSize& sizeScreen, bool bProfile = false)
	///
	/// \brief	Constructs the main frame using given parameters. 
	///
	/// \author	Mathias Neumann
	/// \date	03.10.2010
	///
	/// \param	title		The title string. 
	/// \param	sizeScreen	Screen size to use. Should be quadratic and power of 2, see QuadTreeSP::
	/// 					Initialize(). 
	/// \param	bProfile	Pass \c true to enable profile mode for CUDA profiler.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MainFrame(const wxString& title, const wxSize& sizeScreen, bool bProfile = false);
	virtual ~MainFrame(void);

// Attributes
private:
	wxBoxSizer* m_pSizerBox;
	CUDACanvas* m_pCanvas;
	wxPanel* m_pPanelCtrls;
	wxMenu* m_pMenuFile;
	wxMenu* m_pMenuMRU;
	wxMenu* m_pMenuSettings;
	wxMenu* m_pMenuViewMode;
	wxMenu* m_pMenuPMode;
	wxMenu* m_pMenuHelp;
	wxLogWindow* m_pWndLog;
	wxFileHistory* m_pFileHistory;
	wxTimer* m_pTimerPool;
	wxButton* m_pButSceneConfig;
	wxToggleButton* m_pButRenderMode;

	// Chosen CUDA device ID.
	int m_nCUDADeviceID;
	// Whether we are in profiler mode.
	bool m_bProfile;
	// Single frame mode (computes one picture, that is redrawn every time).
	bool m_bSingleFrame;
	// Do we have a single frame?
	bool m_bSingleDone;
	// This can be used to disable rendering.
	bool m_bCanRender;
	// Current scene configuration.
	SceneConfig* m_pSC;
	// Raytracing core.
	RTCore* m_pRTCore;
	// MNRT settings.
	MNRTSettingsWx* m_pSettings;
	// Update loop active?
	bool m_bUpdateLoopActive;

	// Current reference image path.
	wxString m_strRefImage;
	// Current scale factor.
	double m_dErrorScale;
	// Last render command.
	int m_RenderCommand;

// Implementation
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool ReinitializeCore(bool bUpdate = true)
	///
	/// \brief	Reinitializes the core (RTCore object) of MNRT.
	///
	/// \author	Mathias Neumann
	/// \date	05.10.2010
	///
	/// \param	bUpdate	Pass \c true to update the screen after reinitialization.
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool ReinitializeCore(bool bUpdate = true);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Render(uchar4* d_buffer)
	///
	/// \brief	Renders the scene loaded.
	///
	///			This method should be called by the CUDACanvas object.
	///
	/// \author	Mathias Neumann
	/// \date	04.10.2010
	///
	/// \param [in,out]	d_buffer	The screen buffer to render the image to.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Render(uchar4* d_buffer);

	/// Returns the current camera.
	CameraModel* GetCamera();
	/// Returns core object.
	RTCore* GetRTCore() { return m_pRTCore; }
	/// Returns scene configuration or \c NULL, if none.
	SceneConfig* GetSceneConfig() { return m_pSC; }
	/// Returns the ID of the chosen CUDA device.
	int GetCUDADeviceID() const { return m_nCUDADeviceID; }

	/// Returns \c true if we want to update the displayed image.
	bool WantUpdate() const;

private:
	void OnButtonLoad(wxCommandEvent& event);
	void OnSaveImage(wxCommandEvent& event);
	void OnDisplayError(wxCommandEvent& event);
	void OnShowLog(wxCommandEvent& event);
	void OnShowStats(wxCommandEvent& event);
	void OnMRUFile(wxCommandEvent& event);
	void OnButtonExit(wxCommandEvent& event);

	void OnBenchmark(wxCommandEvent& event);
	void OnTest(wxCommandEvent& event);

	void OnCameraMode(wxCommandEvent& event);
	void OnToggleDynamicScene(wxCommandEvent& event);
	void OnToggleDirectRT(wxCommandEvent& event);
	void OnToggleShadowRays(wxCommandEvent& event);
	void OnToggleReflect(wxCommandEvent& event);
	void OnToggleTransmit(wxCommandEvent& event);
	void OnSettings(wxCommandEvent& event);

	void OnViewModeChange(wxCommandEvent& event);
	void OnPMModeChange(wxCommandEvent& event);
	void OnPMToggleICut(wxCommandEvent& event);
	void OnPMToggleICutUseLeafs(wxCommandEvent& event);

	void OnButtonLoadExample(wxCommandEvent& event);
	void OnButtonRenderMode(wxCommandEvent& event);
	void OnButtonSceneConfig(wxCommandEvent& event);

	void OnHelp(wxCommandEvent& event);
	void OnContact(wxCommandEvent& event);
	void OnToggleErrorChecks(wxCommandEvent& event);
	void OnAbout(wxCommandEvent& event);

	void OnClose(wxCloseEvent& event);
	void OnIdle(wxIdleEvent& event);
	void OnTimer(wxTimerEvent& event);


private:
	bool CheckForCUDA();
	void ActivateUpdateLoop(bool bActivate);

	void CreateMenuBar();
	void CreateControlBar();

	void UpdateControls();

	bool LoadFromFile(const wxString& strModelFile);
	void UnloadScene();

    DECLARE_EVENT_TABLE()
};

#endif //__MN_MAINFRAME_H__