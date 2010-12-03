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

#include "BenchResultDialog.h"

BEGIN_EVENT_TABLE(BenchResultDialog, wxDialog)
END_EVENT_TABLE()

BenchResultDialog::BenchResultDialog(wxWindow* pParent, const wxString& strTitle, const wxString& strResult)
			: wxDialog(pParent, wxID_ANY, strTitle, wxDefaultPosition, wxDefaultSize)
{
	wxBoxSizer* pSizerBox = new wxBoxSizer(wxVERTICAL);

	wxStaticText* pTitle = new wxStaticText(this, -1, _("Result"));
	pTitle->SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));

	wxTextCtrl* pText = new wxTextCtrl(this, -1, strResult, 
		wxDefaultPosition, wxSize(300, 150), wxTE_MULTILINE|wxTE_READONLY);
	pText->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));

	wxPanel* pPanelButtons = new wxPanel(this);

	wxBoxSizer* pSizerButtons = new wxBoxSizer(wxHORIZONTAL);
	pSizerButtons->AddStretchSpacer();
	pSizerButtons->Add(new wxButton(pPanelButtons, wxID_OK, _("Close")), wxSizerFlags().Border(wxALL));
	pSizerButtons->AddStretchSpacer();

	pPanelButtons->SetSizer(pSizerButtons);

	pSizerBox->Add(pTitle, wxSizerFlags().Center().Border());
	pSizerBox->Add(pText, wxSizerFlags().Center().DoubleBorder());
	pSizerBox->Add(pPanelButtons, wxSizerFlags().Expand());

	SetSizer(pSizerBox);
	pSizerBox->SetSizeHints(this);
}

BenchResultDialog::~BenchResultDialog(void)
{
}
