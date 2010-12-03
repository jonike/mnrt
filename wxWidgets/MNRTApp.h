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
/// \file	wxWidgets\MNRTApp.h
///
/// \brief	Declares the MNRTApp class. 
/// \author	Mathias Neumann
/// \date	03.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	UI			User Interface
/// 
/// \brief	Components related to the user interface of MNRT.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MN_MNRTAPP_H_
#define _MN_MNRTAPP_H_

#pragma once

#include <wx/wx.h>
#include <wx/cmdline.h>

class MainFrame;
class wxFFile;

/// Application title.
#define MNRT_APPLICATION	"MNRT - Global illumination on the GPU using CUDA"
/// Application version.
#define MNRT_VERSION		"1.00"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNRTApp
///
/// \brief	Application class for MNRT.
///
///			Manages command line argument parsing and creates the main frame. As command line
///			parameters, currently we only support a switch (-p or --profile) to start a profiler
///			run. This can be helpful in conjunction with the CUDA profiler.
///
/// \author	Mathias Neumann
/// \date	03.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNRTApp : public wxApp
{
private:
	// The main frame.
	MainFrame* m_pFrame;

	// Log file.
	wxFFile* m_pFileLog;

	// Profile mode?
	bool m_bProfile;

private:
    virtual bool OnInit();
	virtual int OnExit();
	virtual int OnRun();
	virtual void OnInitCmdLine(wxCmdLineParser& parser);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
};

// Adds wxGetApp().
//DECLARE_APP(MNRTApp);


#endif //_MN_MNRTAPP_H_