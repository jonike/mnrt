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
/// \file	wxWidgets\StatsDialog.h
///
/// \brief	Declares the StatsDialog class. 
/// \author	Mathias Neumann
/// \date	04.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_STATSDIALOG_H__
#define __MN_STATSDIALOG_H__

#pragma once

#include <wx/wx.h>

class wxToggleButton;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	StatsDialog
///
/// \brief	Simple dialog that shows execution statistics for MNRT.
///
/// \author	Mathias Neumann
/// \date	04.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class StatsDialog : public wxDialog
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	StatsDialog(wxWindow* pParent)
	///
	/// \brief	Constructs the dialog. 
	///
	/// \author	Mathias Neumann
	/// \date	04.10.2010
	///
	/// \param [in]		pParent	The parent window. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	StatsDialog(wxWindow* pParent);
	virtual ~StatsDialog(void);

private:
	wxTextCtrl* m_pText;
	wxToggleButton* m_pButTimers;

private:
	void UpdateStats();

	void OnButtonUpdate(wxCommandEvent& event);
	void OnButtonReset(wxCommandEvent& event);
	void OnButtonEnableTimers(wxCommandEvent& event);

	DECLARE_EVENT_TABLE()
};

#endif // __MN_STATSDIALOG_H__