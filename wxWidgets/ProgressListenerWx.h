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
/// \file	wxWidgets\ProgressListenerWx.h
///
/// \brief	Declares the ProgressListenerWx class. 
/// \author	Mathias Neumann
/// \date	05.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_PROGRESSLISTENERWX_H__
#define __MN_PROGRESSLISTENERWX_H__

#pragma once

#include "../ProgressListener.h"
#include <wx/wx.h>

class wxProgressDialog;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	ProgressListenerWx
///
/// \brief	A wxWidgets implementation of the ProgressListener interface of MNRT.
///
/// \author	Mathias Neumann
/// \date	05.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class ProgressListenerWx : public ProgressListener
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	ProgressListenerWx(wxWindow* pParent, const wxString& strTitle, const wxString& strMsg)
	///
	/// \brief	Constructs the progress listener. 
	///
	/// \author	Mathias Neumann
	/// \date	05.10.2010
	///
	/// \param [in]		pParent	The parent window. 
	/// \param	strTitle		Title string for dialog. 
	/// \param	strMsg			Message string to show within progress dialog. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	ProgressListenerWx(wxWindow* pParent, const wxString& strTitle, const wxString& strMsg);
	virtual ~ProgressListenerWx(void);

private:
	wxWindow* m_pParent;
	wxString m_strTitle;
	wxString m_strMessage;
	wxProgressDialog* m_pDlg;

public:
	virtual void SetMaximum(int maxValue);
	virtual bool Update(int newValue, const std::string& strNewMessage = "");
};

#endif // __MN_PROGRESSLISTENERWX_H__