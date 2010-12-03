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

#include "SceneConfigDialog.h"
#include <wx/propgrid/propgrid.h>
#include "../SceneConfig.h"
#include "../BasicScene.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTOR PROPERTY FOR PROPERTY GRID
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_IGNORE

WX_PG_DECLARE_VARIANT_DATA(MNVector3VariantData, MNVector3, wxPG_NO_DECL)
WX_PG_IMPLEMENT_VARIANT_DATA(MNVector3VariantData, MNVector3)

class wxVectorProperty : public wxPGProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxVectorProperty)

public:
    wxVectorProperty(const wxString& label = wxPG_LABEL, const wxString& name = wxPG_LABEL,
                     const MNVector3& value = MNVector3(),
					 const wxString& comp1 = wxT("X"), const wxString& comp2 = wxT("Y"), 
					 const wxString& comp3 = wxT("Z"));
    virtual ~wxVectorProperty();

    WX_PG_DECLARE_PARENTAL_METHODS()
};

WX_PG_IMPLEMENT_PROPERTY_CLASS(wxVectorProperty, wxPGProperty, MNVector3, const MNVector3&, TextCtrl)

wxVectorProperty::wxVectorProperty(const wxString& label,
		const wxString& name, const MNVector3& value, 
		const wxString& comp1, const wxString& comp2, const wxString& comp3)
    : wxPGProperty(label,name)
{
    SetValue(MNVector3ToVariant(value));
    AddPrivateChild( new wxFloatProperty(comp1,wxPG_LABEL,value.x) );
    AddPrivateChild( new wxFloatProperty(comp2,wxPG_LABEL,value.y) );
    AddPrivateChild( new wxFloatProperty(comp3,wxPG_LABEL,value.z) );
}

wxVectorProperty::~wxVectorProperty() { }

void wxVectorProperty::RefreshChildren()
{
    if (!GetCount()) 
		return;
    MNVector3& vector = MNVector3FromVariant(m_value);
    Item(0)->SetValue( vector.x );
    Item(1)->SetValue( vector.y );
    Item(2)->SetValue( vector.z );
}

void wxVectorProperty::ChildChanged( wxVariant& thisValue, int childIndex, wxVariant& childValue ) const
{
    MNVector3& vector = MNVector3FromVariant(thisValue);
    switch(childIndex)
    {
        case 0: vector.x = childValue.GetDouble(); break;
        case 1: vector.y = childValue.GetDouble(); break;
        case 2: vector.z = childValue.GetDouble(); break;
    }
}

#endif // DOXYGEN_IGNORE

////////////////////////////////////////////////////////////////////////////////////////////////////
// SCENE CONFIG DIALOG
////////////////////////////////////////////////////////////////////////////////////////////////////

BEGIN_EVENT_TABLE(SceneConfigDialog, wxDialog)
	EVT_PG_CHANGING(wxID_ANY, SceneConfigDialog::OnPropertyGridChanging)
	EVT_PG_CHANGED(wxID_ANY, SceneConfigDialog::OnPropertyGridChanged)
END_EVENT_TABLE()

SceneConfigDialog::SceneConfigDialog(wxWindow* pParent, SceneConfig* pSceneConfig)
	: wxDialog(pParent, wxID_ANY, wxT("Scene Configuration"), wxDefaultPosition, wxDefaultSize,
			wxDEFAULT_DIALOG_STYLE)
{
	MNAssert(pSceneConfig);
	m_pSC = pSceneConfig;

	m_pPG = new wxPropertyGrid(this, -1, wxDefaultPosition, wxDefaultSize,
		wxPG_SPLITTER_AUTO_CENTER|wxPG_BOLD_MODIFIED);
	m_pPG->SetMinSize(wxSize(400, 550));
	FillGrid();

	wxPanel* pPanelButtons = new wxPanel(this);
	wxBoxSizer* pSizerButtons = new wxBoxSizer(wxHORIZONTAL);
	pSizerButtons->AddStretchSpacer();
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_OK, _("OK")), wxSizerFlags().Border(wxALL));
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_CANCEL, _("Cancel")), wxSizerFlags().Border(wxALL));
	pPanelButtons->SetSizer(pSizerButtons);

	wxSizer* pSizerBox = new wxBoxSizer(wxVERTICAL);
	pSizerBox->Add(m_pPG, wxSizerFlags().Expand());
	pSizerBox->Add(pPanelButtons, wxSizerFlags().Expand());
	pSizerBox->AddStretchSpacer();

	SetSizer(pSizerBox);
	pSizerBox->SetSizeHints(this);
}

SceneConfigDialog::~SceneConfigDialog(void)
{
}

void SceneConfigDialog::FillGrid()
{
	wxPGProperty* pID, pID2;

	// Scene information (read-only).
	pID = m_pPG->Append(new wxPropertyCategory(wxT("Scene Information")));
	m_pPG->Append(new wxStringProperty(wxT("Scene Name"), 
		wxPG_LABEL, wxFileNameFromPath(m_pSC->GetScenePath())));
	m_pPG->DisableProperty(pID);

	pID = m_pPG->Append(new wxUIntProperty(wxT("Triangle Count"), 
		wxPG_LABEL, m_pSC->GetScene()->GetNumTris()));
	m_pPG->DisableProperty(pID);

	MNBBox bounds = m_pSC->GetSceneBounds();
	MNVector3 vMin(bounds.ptMin.x, bounds.ptMin.y, bounds.ptMin.z);
	pID = m_pPG->Append(new wxVectorProperty(wxT("AABB Minimum"), wxPG_LABEL, vMin));
	m_pPG->DisableProperty(pID);
	MNVector3 vMax(bounds.ptMax.x, bounds.ptMax.y, bounds.ptMax.z);
	pID = m_pPG->Append(new wxVectorProperty(wxT("AABB Maximum"), wxPG_LABEL, vMax));
	m_pPG->DisableProperty(pID);
	pID = m_pPG->Append(new wxVectorProperty(wxT("AABB Extent"), wxPG_LABEL, vMax - vMin));
	m_pPG->DisableProperty(pID);

	// General scene parameters
	m_pPG->Append(new wxPropertyCategory(wxT("General Parameters")));
	pID = m_pPG->Append(new wxBoolProperty(wxT("Construct Caustics Photon Map"), 
		wxPG_LABEL, m_pSC->GetHasSpecular()));
	//m_pPG->SetPropertyAttribute(pID, wxPG_BOOL_USE_CHECKBOX, wxVariant(1));

	// Camera parameters
	CameraModel* pCamera = m_pSC->GetCamera();
	m_pPG->Append(new wxPropertyCategory(wxT("Camera")));
	m_pPG->Append(new wxVectorProperty(wxT("Eye Position"), wxPG_LABEL, *(MNVector3*)&pCamera->GetEye()));
	m_pPG->Append(new wxVectorProperty(wxT("Look At Position"), wxPG_LABEL, *(MNVector3*)&pCamera->GetLookAt()));
	m_pPG->Append(new wxVectorProperty(wxT("Up Vector"), wxPG_LABEL, pCamera->GetUp()));

	// Light parameters
	LightData& light = m_pSC->GetLight();

	wxArrayString lightTypes;
    lightTypes.Add(wxT("Point"));
    lightTypes.Add(wxT("Directional"));
    lightTypes.Add(wxT("Area (disc)"));
	lightTypes.Add(wxT("Area (rectangle)"));

	wxArrayInt lightTypesInt;
	lightTypesInt.Add((int)Light_Point);
	lightTypesInt.Add((int)Light_Directional);
	lightTypesInt.Add((int)Light_AreaDisc);
	lightTypesInt.Add((int)Light_AreaRect);

	m_pPG->Append(new wxPropertyCategory(wxT("Light")));
	m_pPG->Append(new wxEnumProperty(wxT("Type"), wxT("Light.Type"), lightTypes, lightTypesInt, (int)light.type));
	m_pPG->Append(new wxVectorProperty(wxT("Position"), wxT("Light.Position"), *(MNVector3*)&light.position));
	m_pPG->Append(new wxVectorProperty(wxT("Direction"), wxT("Light.Direction"), *(MNVector3*)&light.direction));
	m_pPG->Append(new wxVectorProperty(wxT("Emitted Radiance"), wxT("Light.Emitted Radiance"), *(MNVector3*)&light.L_emit, 
		wxT("R"), wxT("G"), wxT("B")));
	m_pPG->Append(new wxVectorProperty(wxT("Rectangle Vector 1"), wxT("Light.Rectangle Vector 1"), *(MNVector3*)&light.areaV1));
	m_pPG->Append(new wxVectorProperty(wxT("Rectangle Vector 2"), wxT("Light.Rectangle Vector 2"), *(MNVector3*)&light.areaV2));
	m_pPG->Append(new wxFloatProperty(wxT("Disc Radius"), wxT("Light.Disc Radius"), light.areaRadius));

	// Algorithmic Parameters
	m_pPG->Append(new wxPropertyCategory(wxT("Algorithmic Parameters")));
	m_pPG->Append(new wxIntProperty(wxT("Target Count (Global)"), 
		wxPG_LABEL, m_pSC->GetTargetCountGlobal()));
	m_pPG->Append(new wxIntProperty(wxT("Target Count (Caustics)"), 
		wxPG_LABEL, m_pSC->GetTargetCountCaustics()));
	m_pPG->Append(new wxFloatProperty(wxT("Ray Epsilon"), 
		wxPG_LABEL, m_pSC->GetRayEpsilon()));
	m_pPG->Append(new wxFloatProperty(wxT("kNN Search Radius"), 
		wxPG_LABEL, m_pSC->GetRadiusPMapMax()));
	m_pPG->Append(new wxFloatProperty(wxT("k-Means-Algorithm Search Radius"), 
		wxPG_LABEL, m_pSC->GetRadiusWangKMeansMax()));
	m_pPG->Append(new wxFloatProperty(wxT("Illumination Sample Search Radius"), 
		wxPG_LABEL, m_pSC->GetRadiusWangInterpolMax()));
	m_pPG->Append(new wxUIntProperty(wxT("Adaptive FG Initial Samples"), 
		wxPG_LABEL, m_pSC->GetWangInitialSamples()));

	// Update states.
	UpdateProperties();
}

void SceneConfigDialog::OnPropertyGridChanging(wxPropertyGridEvent& event)
{
    wxPGProperty* p = event.GetProperty();
	event.SetValidationFailureBehavior(wxPG_VFB_STAY_IN_PROPERTY|wxPG_VFB_BEEP|wxPG_VFB_MARK_CELL);

	// Make sure value is not unspecified.
	wxVariant pendingValue = event.GetValue();
    if(!pendingValue.IsNull())
	{
		// Ensure ray epsilon is larger or equal zero.
		if(p->GetName() == wxT("Ray Epsilon"))
			if(pendingValue.GetDouble() < 0.f)
				event.Veto();
		// Ensure the radii are larger than zero.
		else if(p->GetName() == wxT("Photon Mapping Search Radius") ||
			p->GetName() == wxT("k-Means-Algorithm Search Radius") ||
			p->GetName() == wxT("Illumination Sample Search Radius"))
		{
			if(pendingValue.GetDouble() <= 0.f)
				event.Veto();
		}
	}
}

void SceneConfigDialog::OnPropertyGridChanged(wxPropertyGridEvent& event)
{
    wxPGProperty* p = event.GetProperty();
    if (!p) // Might be NULL.
        return;

	if(p->GetName() == wxT("Light.Type"))
		UpdateProperties();
}

void SceneConfigDialog::UpdateProperties()
{
	LightType typeLight = (LightType)m_pPG->GetPropertyValueAsInt(wxT("Light.Type"));
	bool isAreaRect = (typeLight == Light_AreaRect);
	bool isAreaDisc = (typeLight == Light_AreaDisc);
	bool isPoint = (typeLight == Light_Point);
	bool isDirectional = (typeLight == Light_Directional);

	// Update state of properties according to light type.
	m_pPG->EnableProperty(wxT("Light.Rectangle Vector 1"), isAreaRect);
	m_pPG->EnableProperty(wxT("Light.Rectangle Vector 2"), isAreaRect);
	m_pPG->EnableProperty(wxT("Light.Disc Radius"), isAreaDisc);
	m_pPG->EnableProperty(wxT("Light.Position"), !isDirectional);
	if(isPoint)
		m_pPG->SetPropertyLabel(wxT("Light.Emitted Radiance"), wxT("Emitted Intensity"));
	else
		m_pPG->SetPropertyLabel(wxT("Light.Emitted Radiance"), wxT("Emitted Radiance"));
}


bool SceneConfigDialog::Validate()
{
	return m_pPG->EditorValidate();
}

bool SceneConfigDialog::TransferDataFromWindow()
{
	m_pSC->SetHasSpecular(m_pPG->GetPropertyValueAsBool(wxT("Construct Caustics Photon Map")));

	CameraModel* pCamera = m_pSC->GetCamera();
	MNPoint3 ptEye = *(MNPoint3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Eye Position")));
	MNPoint3 ptLookAt = *(MNPoint3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Look At Position")));
	MNVector3 vUp = MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Up Vector")));
	pCamera->LookAt(ptEye, ptLookAt, vUp);

	// Just read in, even if the light type doesn't support all properties.
	LightData& light = m_pSC->GetLight();
	light.type = (LightType)m_pPG->GetPropertyValueAsInt(wxT("Light.Type"));
	light.position = *(float3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Light.Position")));
	light.direction = *(float3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Light.Direction")));
	light.L_emit = *(float3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Light.Emitted Radiance")));
	light.areaV1 = *(float3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Light.Rectangle Vector 1")));
	light.areaV2 = *(float3*)&MNVector3FromVariant(m_pPG->GetPropertyValue(wxT("Light.Rectangle Vector 2")));
	light.areaRadius = m_pPG->GetPropertyValueAsDouble(wxT("Light.Disc Radius"));

	m_pSC->SetTargetCountGlobal(m_pPG->GetPropertyValueAsInt(wxT("Target Count (Global)")));
	m_pSC->SetTargetCountCaustics(m_pPG->GetPropertyValueAsInt(wxT("Target Count (Caustics)")));
	m_pSC->SetRayEpsilon(m_pPG->GetPropertyValueAsDouble(wxT("Ray Epsilon")));
	m_pSC->SetRadiusPMapMax(m_pPG->GetPropertyValueAsDouble(wxT("kNN Search Radius")));
	m_pSC->SetRadiusWangKMeansMax(m_pPG->GetPropertyValueAsDouble(wxT("k-Means-Algorithm Search Radius")));
	m_pSC->SetRadiusWangInterpolMax(m_pPG->GetPropertyValueAsDouble(wxT("Illumination Sample Search Radius")));
	m_pSC->SetWangInitialSamples(m_pPG->GetPropertyValueAsLong(wxT("Adaptive FG Initial Samples")));
	return true;
}

bool SceneConfigDialog::IsConfigModified() const
{
	return m_pPG->IsAnyModified();
}