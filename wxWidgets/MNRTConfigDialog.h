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
/// \file	wxWidgets\MNRTConfigDialog.h
///
/// \brief	Declares the MNRTConfigDialog class. 
/// \author	Mathias Neumann
/// \date	11.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_MNRTCONFIGDIALOG_H__
#define __MN_MNRTCONFIGDIALOG_H__

#pragma once

#include <wx/wx.h>
#include <wx/propgrid/propgrid.h>

class MNRTSettings;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNRTConfigDialog
///
/// \brief	Provides a setting dialog to change MNRT's global configuration. 
///
///			This dialog includes all settings which aren't considered as scene dependend. Dialog is
///			implemented with the help of the \c wxPropertyGrid class. Check
///			http://wxpropgrid.sourceforge.net/cgi-bin/index for more information on that
///			wxWidgets class.
///
/// \author	Mathias Neumann
/// \date	11.10.2010
/// \see	MNRTSettings, SceneConfigDialog, SceneConfig
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNRTConfigDialog : public wxDialog
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNRTConfigDialog(wxWindow* pParent, MNRTSettings* pSet)
	///
	/// \brief	Constructs the dialog for the given MNRTSettings object. It is used to show the
	/// 		settings to change. 
	///
	/// \author	Mathias Neumann
	/// \date	11.10.2010
	///
	/// \param [in]		pParent	The parent window. 
	/// \param [in]		pSet	The settings object. May not be \c NULL. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNRTConfigDialog(wxWindow* pParent, MNRTSettings* pSet);
	virtual ~MNRTConfigDialog(void);

private:
	MNRTSettings* m_pSet;

	wxPropertyGrid* m_pPG;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool IsConfigModified() const
	///
	/// \brief	Query if the configuration was modified. Call this right after the dialog was closed.  
	///
	/// \author	Mathias Neumann
	/// \date	11.10.2010
	///
	/// \return	\c true if configuration was modified, \c false if not. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool IsConfigModified() const;

private:
	virtual bool Validate();
	virtual bool TransferDataFromWindow();
private:
	void OnRestoreDefaults(wxCommandEvent& event);

	void OnPropertyGridChanging(wxPropertyGridEvent& event);
	void OnPropertyGridChanged(wxPropertyGridEvent& event);

private:
	void FillGrid();
	void UpdateProperties();

	DECLARE_EVENT_TABLE()
};

#endif // __MN_MNRTCONFIGDIALOG_H__