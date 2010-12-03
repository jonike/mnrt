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

#include "ProgressListenerWx.h"
#include "../MNUtilities.h"
#include <wx/progdlg.h>

ProgressListenerWx::ProgressListenerWx(wxWindow* pParent, const wxString& strTitle, const wxString& strMsg)
{
	m_pParent = pParent;
	m_strTitle = strTitle;
	m_strMessage = strMsg;
	m_pDlg = NULL;
}

ProgressListenerWx::~ProgressListenerWx(void)
{
}

void ProgressListenerWx::SetMaximum(int maxValue)
{
	// Now we can initialize and show the progress dialog as we have a maximum value.
	m_pDlg = new wxProgressDialog(m_strTitle, m_strMessage, maxValue, m_pParent, 
		wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_REMAINING_TIME | wxPD_ELAPSED_TIME | wxPD_CAN_ABORT);
}

bool ProgressListenerWx::Update(int newValue, const std::string& strNewMessage/* = ""*/)
{
	if(m_pDlg == NULL)
		MNFatal("Progress dialog initialization missing.");

	bool result = m_pDlg->Update(newValue, wxString(strNewMessage));
	if(!result)
	{
		// Just destroy for now. Asking would be too much.
		m_pDlg->Destroy();
	}

	// Yields control to pending messages in the windowing system. Also disables controls to
	// avoid illegal user input.
	wxSafeYield();

	return result;
}