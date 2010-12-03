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



#include "MNRTApp.h"
#include "../MNUtilities.h"
#include "MainFrame.h"
#include <wx/config.h>
#include <wx/ffile.h>

// DevIL includes to load/save images.
#include <IL/il.h>

// We currently require that *width = height* and both are *power of 2* for simple quad tree construction!
#define MNRT_SCREEN_W		512
#define MNRT_SCREEN_H		512


// Command line switches/options/...
// See http://wiki.wxwidgets.org/Command-Line_Arguments
// and http://docs.wxwidgets.org/stable/wx_wxcmdlineparser.html#wxcmdlineparser
static const wxCmdLineEntryDesc f_cmdLineDesc [] =
{
     { wxCMD_LINE_SWITCH, wxT("p"), wxT("profile"), wxT("Performs profiler run"),
          wxCMD_LINE_VAL_NONE, wxCMD_LINE_PARAM_OPTIONAL },
 
     { wxCMD_LINE_NONE }
};

// Use IMPLEMENT_APP_CONSOLE() for /SUBSYSTEM:CONSOLE.
IMPLEMENT_APP(MNRTApp)
//IMPLEMENT_APP_CONSOLE(MNRTApp)


bool MNRTApp::OnInit()
{
	if(!wxApp::OnInit())
		return false;
	SetAppName("MNRT");
	SetVendorName("Mathias Neumann");

	// Set up file logger.
	m_pFileLog = new wxFFile(_("MNRT.log"), _("w+"));
	if(!m_pFileLog->IsOpened())
		MNFatal("Failed to open log file \"MNRT.log\".");
	wxLog::SetActiveTarget(new wxLogStderr(m_pFileLog->fp()));

	// Init DevIL base library.
	ilInit();
	// We want all images to be loaded in a consistent manner.
	ilEnable(IL_ORIGIN_SET);

	// Create global configuration.
	wxConfig::Set(new wxConfig(_("MNRT")));

	// Create main window.
	wxString strTitle = wxT(MNRT_APPLICATION);
    m_pFrame = new MainFrame(strTitle, wxSize(MNRT_SCREEN_W, MNRT_SCREEN_H), m_bProfile);
    m_pFrame->Show(true);
	m_pFrame->CenterOnScreen();
	SetTopWindow(m_pFrame);

    return true;
}

int MNRTApp::OnRun()
{
	return wxApp::OnRun();
}

int MNRTApp::OnExit()
{
	wxLog::FlushActive();
	if(m_pFileLog)
		m_pFileLog->Flush();
	SAFE_DELETE(m_pFileLog);
	return 0;
}

void MNRTApp::OnInitCmdLine(wxCmdLineParser& parser)
{
    parser.SetDesc(f_cmdLineDesc);

    // Only "-switch" allowed.
    parser.SetSwitchChars(wxT("-"));
}
 
bool MNRTApp::OnCmdLineParsed(wxCmdLineParser& parser)
{
	// Shall we do a profiler run?
    m_bProfile = parser.Found(wxT("p"));
  
    return true;
}