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

#include "StatsDialog.h"
#include "../MNCudaMemPool.h"
#include "../MNStatContainer.h"
#include <wx/ffile.h>
#include <wx/filename.h>
#include <wx/tglbtn.h>

enum
{
	IDB_UPDATE = 1,
	IDB_RESET,
	IDB_ENABLETIMERS,
};

BEGIN_EVENT_TABLE(StatsDialog, wxDialog)
	EVT_BUTTON(IDB_UPDATE,  StatsDialog::OnButtonUpdate)
	EVT_BUTTON(IDB_RESET,  StatsDialog::OnButtonReset)
	EVT_TOGGLEBUTTON(IDB_ENABLETIMERS,  StatsDialog::OnButtonEnableTimers)
END_EVENT_TABLE()

StatsDialog::StatsDialog(wxWindow* pParent)
			: wxDialog(pParent, wxID_ANY, _("Statistics"), wxDefaultPosition, wxDefaultSize)
{
	MNStatContainer& stats = MNStatContainer::GetInstance();

	m_pText = new wxTextCtrl(this, -1, _(""), wxDefaultPosition, wxSize(470, 400), wxTE_MULTILINE|wxTE_READONLY);
	m_pText->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
	
	wxPanel* pPanelButtons = new wxPanel(this);

	m_pButTimers = new wxToggleButton(pPanelButtons, IDB_ENABLETIMERS, _("Enable Timers"));
	m_pButTimers->SetValue(stats.GetTimersEnabled());

	wxBoxSizer* pSizerButtons = new wxBoxSizer(wxHORIZONTAL);
	pSizerButtons->Add(new wxButton(pPanelButtons, IDB_RESET, _("Reset")), wxSizerFlags().Border(wxALL));
	pSizerButtons->Add(m_pButTimers, wxSizerFlags().Border(wxALL));
	pSizerButtons->AddStretchSpacer();
	pSizerButtons->Add(new wxButton(pPanelButtons, IDB_UPDATE, _("Update")), wxSizerFlags().Border(wxALL));

	pPanelButtons->SetSizer(pSizerButtons);

	wxBoxSizer* pSizerBox = new wxBoxSizer(wxVERTICAL);
	pSizerBox->Add(m_pText, wxSizerFlags().Center());
	pSizerBox->Add(pPanelButtons, wxSizerFlags().Expand());

	SetSizer(pSizerBox);
	pSizerBox->SetSizeHints(this);
	UpdateStats();
}

StatsDialog::~StatsDialog(void)
{
}

void StatsDialog::OnButtonUpdate(wxCommandEvent& event)
{
	UpdateStats();
}

void StatsDialog::OnButtonReset(wxCommandEvent& event)
{
	MNStatContainer& stats = MNStatContainer::GetInstance();
	stats.Reset();

	UpdateStats();
}

void StatsDialog::OnButtonEnableTimers(wxCommandEvent& event)
{
	MNStatContainer& stats = MNStatContainer::GetInstance();
	bool newValue = !stats.GetTimersEnabled();
	stats.SetTimersEnabled(newValue);
	m_pButTimers->SetValue(newValue);

	if(newValue && wxYES == wxMessageBox(_("Do you want to reset all statistics to enable correct timing?"), 
									_("Statistics reset recommended"), wxYES_NO|wxICON_QUESTION))
		stats.Reset();

	UpdateStats();
}

void StatsDialog::UpdateStats()
{
	// Destructor closes file.
	wxFFile file;
	wxString strTmpPath = wxFileName::CreateTempFileName(_("MNRT"), &file);
	if(strTmpPath.IsEmpty())
	{
		MNError("Failed to create temporary file.");
		m_pText->ChangeValue(_(""));
		return;
	}
	
	// General statistics.
	MNStatContainer& stats = MNStatContainer::GetInstance();
	stats.Print(file.fp());

	file.Write(_("\n"));

	// Memory pool statistics.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	pool.PrintState(file.fp());

	wxString strStats;
	file.Seek(0);
	file.ReadAll(&strStats);

	// There seems to be no way to automatically scroll back to the last position
	// under Windows. See http://wiki.wxwidgets.org/WxTextCtrl
	m_pText->ChangeValue(strStats);
}