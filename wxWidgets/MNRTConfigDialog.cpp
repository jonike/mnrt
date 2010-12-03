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

#include "MNRTConfigDialog.h"
#include <wx/propgrid/propgrid.h>
#include <wx/statline.h>
#include "../MNRTSettings.h"
#include "../MNUtilities.h"


BEGIN_EVENT_TABLE(MNRTConfigDialog, wxDialog)
	EVT_BUTTON(wxID_DEFAULT, MNRTConfigDialog::OnRestoreDefaults)
	EVT_PG_CHANGING(wxID_ANY, MNRTConfigDialog::OnPropertyGridChanging)
	EVT_PG_CHANGED(wxID_ANY, MNRTConfigDialog::OnPropertyGridChanged)
END_EVENT_TABLE()

MNRTConfigDialog::MNRTConfigDialog(wxWindow* pParent, MNRTSettings* pSet)
	: wxDialog(pParent, wxID_ANY, wxT("MNRT Configuration"), wxDefaultPosition, wxDefaultSize,
			wxDEFAULT_DIALOG_STYLE)
{
	MNAssert(pSet);
	m_pSet = pSet;

	m_pPG = new wxPropertyGrid(this, -1, wxDefaultPosition, wxDefaultSize,
		wxPG_SPLITTER_AUTO_CENTER|wxPG_BOLD_MODIFIED);
	m_pPG->SetMinSize(wxSize(400, 400));
	FillGrid();

	wxPanel* pPanelButtons = new wxPanel(this);
	wxBoxSizer* pSizerButtons = new wxBoxSizer(wxHORIZONTAL);
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_DEFAULT, _("Restore Defaults")), wxSizerFlags().Border(wxALL));
	pSizerButtons->Add(new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVERTICAL), 
		wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerButtons->AddStretchSpacer();
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_OK), wxSizerFlags().Border(wxALL));
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_CANCEL), wxSizerFlags().Border(wxALL));
	pPanelButtons->SetSizer(pSizerButtons);

	wxSizer* pSizerBox = new wxBoxSizer(wxVERTICAL);
	pSizerBox->Add(m_pPG, wxSizerFlags().Expand());
	pSizerBox->Add(pPanelButtons, wxSizerFlags().Expand());
	pSizerBox->AddStretchSpacer();

	SetSizer(pSizerBox);
	pSizerBox->SetSizeHints(this);
}

MNRTConfigDialog::~MNRTConfigDialog(void)
{
}

void MNRTConfigDialog::FillGrid()
{
	wxPGProperty* pID;

	// Ray Tracing
	m_pPG->Append(new wxPropertyCategory(wxT("Ray Tracing")));
	m_pPG->Append(new wxBoolProperty(wxT("Direct Lighting"), wxPG_LABEL, m_pSet->GetEnableDirectRT()));
	m_pPG->Append(new wxBoolProperty(wxT("Trace Shadow Rays"), wxPG_LABEL, m_pSet->GetEnableShadowRays()));
	m_pPG->Append(new wxBoolProperty(wxT("Specular Reflection"), wxPG_LABEL, m_pSet->GetEnableSpecReflect()));
	m_pPG->Append(new wxBoolProperty(wxT("Specular Transmission"), wxPG_LABEL, m_pSet->GetEnableSpecTransmit()));
	pID = m_pPG->Append(new wxStringProperty(wxT("Area Light Samples"), wxT("AreaSamples"), wxT("<composed>")));
	m_pPG->AppendIn(pID, new wxUIntProperty(wxT("X"), wxPG_LABEL, m_pSet->GetAreaLightSamplesX()));
	m_pPG->AppendIn(pID, new wxUIntProperty(wxT("Y"), wxPG_LABEL, m_pSet->GetAreaLightSamplesY()));

	wxArrayString pmapModes;
    pmapModes.Add(wxT("Disabled"));
    pmapModes.Add(wxT("Visualize Photons"));
    pmapModes.Add(wxT("Full Final Gathering"));
	pmapModes.Add(wxT("Adaptive FG + Best Fit"));
	pmapModes.Add(wxT("Adaptive FG + Interpolation"));

	wxArrayInt pmapModesInt;
	pmapModesInt.Add((int)PhotonMap_Disabled);
	pmapModesInt.Add((int)PhotonMap_Visualize);
	pmapModesInt.Add((int)PhotonMap_FullFinalGather);
	pmapModesInt.Add((int)PhotonMap_AdaptiveSamplesBestFit);
	pmapModesInt.Add((int)PhotonMap_AdaptiveSamplesWang);

	// Photon Mapping
	m_pPG->Append(new wxPropertyCategory(wxT("Photon Mapping")));
	m_pPG->Append(new wxEnumProperty(wxT("Mode"), wxT("PMap.Mode"), 
		pmapModes, pmapModesInt, (int)m_pSet->GetPhotonMapMode()));
	m_pPG->Append(new wxUIntProperty(wxT("Max. Photon Bounces"), 
		wxPG_LABEL, m_pSet->GetMaxPhotonBounces()));
	m_pPG->Append(new wxUIntProperty(wxT("Target Count (Global)"), 
		wxPG_LABEL, m_pSet->GetTargetCountGlobal()));
	m_pPG->Append(new wxUIntProperty(wxT("Target Count (Caustics)"), 
		wxPG_LABEL, m_pSet->GetTargetCountCaustics()));
	m_pPG->Append(new wxUIntProperty(wxT("k for kNN Search (Global)"), 
		wxPG_LABEL, m_pSet->GetKinKNNSearchGlobal()));
	m_pPG->Append(new wxUIntProperty(wxT("k for kNN Search (Caustics)"), 
		wxPG_LABEL, m_pSet->GetKinKNNSearchCaustics()));
	m_pPG->Append(new wxUIntProperty(wxT("kNN Refinement Iterations"), 
		wxPG_LABEL, m_pSet->GetKNNRefineIters()));

	// Final Gathering
	m_pPG->Append(new wxPropertyCategory(wxT("Final Gathering")));
	pID = m_pPG->Append(new wxStringProperty(wxT("Final Gathering Rays"), 
		wxT("FGRays"), wxT("<composed>")));
	m_pPG->AppendIn(pID, new wxUIntProperty(wxT("X"), wxPG_LABEL, m_pSet->GetFinalGatherRaysX()));
	m_pPG->AppendIn(pID, new wxUIntProperty(wxT("Y"), wxPG_LABEL, m_pSet->GetFinalGatherRaysY()));
	m_pPG->Append(new wxFloatProperty(wxT("Geometric Variation Alpha"), 
		wxPG_LABEL, m_pSet->GetGeoVarAlpha()));
	m_pPG->Append(new wxFloatProperty(wxT("Geometric Variation Propagation"), 
		wxPG_LABEL, m_pSet->GetGeoVarPropagation()));
	m_pPG->Append(new wxUIntProperty(wxT("k-Means Iterations (Max.)"), 
		wxPG_LABEL, m_pSet->GetKMeansItersMax()));
	m_pPG->Append(new wxBoolProperty(wxT("Illumination Cuts"), 
		wxPG_LABEL, m_pSet->GetUseIllumCuts()));
	m_pPG->Append(new wxBoolProperty(wxT("ICut: Use Leafs as Cut Nodes"), 
		wxPG_LABEL, m_pSet->GetICutUseLeafs()));
	m_pPG->Append(new wxUIntProperty(wxT("ICut: Node level for E_min"), 
		wxPG_LABEL, m_pSet->GetICutLevelEmin()));
	m_pPG->Append(new wxUIntProperty(wxT("ICut: Refinement iterations"), 
		wxPG_LABEL, m_pSet->GetICutRefineIters()));
	m_pPG->Append(new wxFloatProperty(wxT("ICut: Required Accuracy"), 
		wxPG_LABEL, m_pSet->GetICutAccuracy()));

	// Update states.
	UpdateProperties();
}

void MNRTConfigDialog::UpdateProperties()
{
	PhotonMapMode mode = (PhotonMapMode)m_pPG->GetPropertyValueAsInt(wxT("PMap.Mode"));
	bool bPMap = mode != PhotonMap_Disabled;
	bool bICut = m_pPG->GetPropertyValueAsBool(wxT("Illumination Cuts"));

	m_pPG->EnableProperty(wxT("Max. Photon Bounces"), bPMap);
	m_pPG->EnableProperty(wxT("Target Count (Global)"), bPMap);
	m_pPG->EnableProperty(wxT("Target Count (Caustics)"), bPMap);
	m_pPG->EnableProperty(wxT("k for kNN Search (Global)"), bPMap);
	m_pPG->EnableProperty(wxT("k for kNN Search (Caustics)"), bPMap);
	m_pPG->EnableProperty(wxT("kNN Refinement Iterations"), bPMap);

	m_pPG->EnableProperty(wxT("FGRays.X"), bPMap);
	m_pPG->EnableProperty(wxT("FGRays.Y"), bPMap);
	m_pPG->EnableProperty(wxT("Geometric Variation Alpha"), bPMap);
	m_pPG->EnableProperty(wxT("Geometric Variation Propagation"), bPMap);
	m_pPG->EnableProperty(wxT("k-Means Iterations (Max.)"), bPMap);
	m_pPG->EnableProperty(wxT("Illumination Cuts"), bPMap);
	m_pPG->EnableProperty(wxT("ICut: Use Leafs as Cut Nodes"), bPMap && bICut);
	m_pPG->EnableProperty(wxT("ICut: Node level for E_min"), bPMap && bICut);
	m_pPG->EnableProperty(wxT("ICut: Refinement iterations"), bPMap && bICut);
	m_pPG->EnableProperty(wxT("ICut: Required Accuracy"), bPMap && bICut);
}

void MNRTConfigDialog::OnPropertyGridChanging(wxPropertyGridEvent& event)
{
    wxPGProperty* p = event.GetProperty();
	event.SetValidationFailureBehavior(wxPG_VFB_STAY_IN_PROPERTY|wxPG_VFB_BEEP|wxPG_VFB_MARK_CELL);

	// Make sure value is not unspecified.
	wxVariant pendingValue = event.GetValue();
    if(!pendingValue.IsNull())
	{
		// Ensure the geometric variation is at least zero.
		if(p->GetName() == wxT("Geometric Variation Alpha"))
		{
			if(pendingValue.GetDouble() < 0.f)
				event.Veto();
		}
	}
}

void MNRTConfigDialog::OnPropertyGridChanged(wxPropertyGridEvent& event)
{
    wxPGProperty* p = event.GetProperty();
    if (!p) // Might be NULL.
        return;

	UpdateProperties();
}

bool MNRTConfigDialog::Validate()
{
	return m_pPG->EditorValidate();
}

bool MNRTConfigDialog::TransferDataFromWindow()
{
	m_pSet->SetEnableDirectRT(m_pPG->GetPropertyValueAsBool(wxT("Direct Lighting")));
	m_pSet->SetEnableShadowRays(m_pPG->GetPropertyValueAsBool(wxT("Trace Shadow Rays")));
	m_pSet->SetEnableSpecReflect(m_pPG->GetPropertyValueAsBool(wxT("Specular Reflection")));
	m_pSet->SetEnableSpecTransmit(m_pPG->GetPropertyValueAsBool(wxT("Specular Transmission")));
	m_pSet->SetAreaLightSamplesX(m_pPG->GetPropertyValueAsLong(wxT("AreaSamples.X")));
	m_pSet->SetAreaLightSamplesY(m_pPG->GetPropertyValueAsLong(wxT("AreaSamples.Y")));

	m_pSet->SetPhotonMapMode((PhotonMapMode)m_pPG->GetPropertyValueAsLong(wxT("PMap.Mode")));
	m_pSet->SetMaxPhotonBounces(m_pPG->GetPropertyValueAsLong(wxT("Max. Photon Bounces")));
	m_pSet->SetTargetCountGlobal(m_pPG->GetPropertyValueAsLong(wxT("Target Count (Global)")));
	m_pSet->SetTargetCountCaustics(m_pPG->GetPropertyValueAsLong(wxT("Target Count (Caustics)")));
	m_pSet->SetKinKNNSearchGlobal(m_pPG->GetPropertyValueAsLong(wxT("k for kNN Search (Global)")));
	m_pSet->SetKinKNNSearchCaustics(m_pPG->GetPropertyValueAsLong(wxT("k for kNN Search (Caustics)")));
	m_pSet->SetKNNRefineIters(m_pPG->GetPropertyValueAsLong(wxT("kNN Refinement Iterations")));

	m_pSet->SetFinalGatherRaysX(m_pPG->GetPropertyValueAsLong(wxT("FGRays.X")));
	m_pSet->SetFinalGatherRaysY(m_pPG->GetPropertyValueAsLong(wxT("FGRays.Y")));
	m_pSet->SetGeoVarAlpha(m_pPG->GetPropertyValueAsDouble(wxT("Geometric Variation Alpha")));
	m_pSet->SetGeoVarPropagation(m_pPG->GetPropertyValueAsDouble(wxT("Geometric Variation Propagation")));
	m_pSet->SetKMeansItersMax(m_pPG->GetPropertyValueAsLong(wxT("k-Means Iterations (Max.)")));
	m_pSet->SetUseIllumCuts(m_pPG->GetPropertyValueAsBool(wxT("Illumination Cuts")));
	m_pSet->SetICutUseLeafs(m_pPG->GetPropertyValueAsBool(wxT("ICut: Use Leafs as Cut Nodes")));
	m_pSet->SetICutLevelEmin(m_pPG->GetPropertyValueAsLong(wxT("ICut: Node level for E_min")));
	m_pSet->SetICutRefineIters(m_pPG->GetPropertyValueAsLong(wxT("ICut: Refinement iterations")));
	m_pSet->SetICutAccuracy(m_pPG->GetPropertyValueAsDouble(wxT("ICut: Required Accuracy")));

	return true;
}

bool MNRTConfigDialog::IsConfigModified() const
{
	return m_pPG->IsAnyModified();
}

void MNRTConfigDialog::OnRestoreDefaults(wxCommandEvent& event)
{
	if(wxNO == wxMessageBox(_("Are you sure?"), _("Restore Defaults"), wxYES_NO|wxICON_QUESTION))
		return;

	// Need to improve this by using some kind of vector for the settings to avoid redundant code...

	// Use ChangePropertyValue to generate events and enable modification display.
	m_pPG->ChangePropertyValue(wxT("Direct Lighting"), wxVariant(m_pSet->GetEnableDirectRTDef()));
	m_pPG->ChangePropertyValue(wxT("Trace Shadow Rays"), wxVariant(m_pSet->GetEnableShadowRaysDef()));
	m_pPG->ChangePropertyValue(wxT("Specular Reflection"), wxVariant(m_pSet->GetEnableSpecReflectDef()));
	m_pPG->ChangePropertyValue(wxT("Specular Transmission"), wxVariant(m_pSet->GetEnableSpecTransmitDef()));
	m_pPG->ChangePropertyValue(wxT("AreaSamples.X"), wxVariant((long)m_pSet->GetAreaLightSamplesXDef()));
	m_pPG->ChangePropertyValue(wxT("AreaSamples.Y"), wxVariant((long)m_pSet->GetAreaLightSamplesYDef()));

	m_pPG->ChangePropertyValue(wxT("PMap.Mode"), wxVariant(m_pSet->GetPhotonMapModeDef()));
	m_pPG->ChangePropertyValue(wxT("Max. Photon Bounces"), wxVariant((long)m_pSet->GetMaxPhotonBouncesDef()));
	m_pPG->ChangePropertyValue(wxT("Target Count (Global)"), wxVariant((long)m_pSet->GetTargetCountGlobalDef()));
	m_pPG->ChangePropertyValue(wxT("Target Count (Caustics)"), wxVariant((long)m_pSet->GetTargetCountCausticsDef()));
	m_pPG->ChangePropertyValue(wxT("k for kNN Search (Global)"), wxVariant((long)m_pSet->GetKinKNNSearchGlobalDef()));
	m_pPG->ChangePropertyValue(wxT("k for kNN Search (Caustics)"), wxVariant((long)m_pSet->GetKinKNNSearchCausticsDef()));
	m_pPG->ChangePropertyValue(wxT("kNN Refinement Iterations"), wxVariant((long)m_pSet->GetKNNRefineItersDef()));

	m_pPG->ChangePropertyValue(wxT("FGRays.X"), wxVariant((long)m_pSet->GetFinalGatherRaysXDef()));
	m_pPG->ChangePropertyValue(wxT("FGRays.Y"), wxVariant((long)m_pSet->GetFinalGatherRaysYDef()));
	m_pPG->ChangePropertyValue(wxT("Geometric Variation Alpha"), wxVariant((double)m_pSet->GetGeoVarAlphaDef()));
	m_pPG->ChangePropertyValue(wxT("Geometric Variation Propagation"), wxVariant((double)m_pSet->GetGeoVarPropagationDef()));
	m_pPG->ChangePropertyValue(wxT("k-Means Iterations (Max.)"), wxVariant((long)m_pSet->GetKMeansItersMaxDef()));
	m_pPG->ChangePropertyValue(wxT("Illumination Cuts"), wxVariant(m_pSet->GetUseIllumCutsDef()));
	m_pPG->ChangePropertyValue(wxT("ICut: Use Leafs as Cut Nodes"), wxVariant(m_pSet->GetICutUseLeafsDef()));
	m_pPG->ChangePropertyValue(wxT("ICut: Node level for E_min"), wxVariant((long)m_pSet->GetICutLevelEminDef()));
	m_pPG->ChangePropertyValue(wxT("ICut: Refinement iterations"), wxVariant((long)m_pSet->GetICutRefineItersDef()));
	m_pPG->ChangePropertyValue(wxT("ICut: Required Accuracy"), wxVariant((double)m_pSet->GetICutAccuracyDef()));
}