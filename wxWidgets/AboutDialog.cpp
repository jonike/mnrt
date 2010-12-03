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

#include "AboutDialog.h"
#include "MNRTApp.h"
#include <wx/statline.h>

BEGIN_EVENT_TABLE(AboutDialog, wxDialog)
	EVT_HTML_LINK_CLICKED(wxID_ANY, OnLinkEvent)
END_EVENT_TABLE()

AboutDialog::AboutDialog(wxWindow* pParent)
			: wxDialog(pParent, wxID_ANY, wxT("About MNRT"), wxDefaultPosition, wxDefaultSize)
{
	wxBoxSizer* pSizerBox = new wxBoxSizer(wxVERTICAL);

	// Button bar.
	wxBoxSizer* pSizerButtons = new wxBoxSizer(wxHORIZONTAL);
	pSizerButtons->AddStretchSpacer();
	pSizerButtons->Add(new wxButton(this, wxID_OK, _("OK")), wxSizerFlags().Border(wxALL));

	// Main sizer.
	pSizerBox->Add(CreateTitle(), wxSizerFlags().Center().Border());
	pSizerBox->Add(new wxStaticLine(this), wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerBox->AddSpacer(10);
	pSizerBox->Add(CreateInfo(), wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerBox->AddSpacer(10);
	pSizerBox->Add(new wxStaticLine(this), wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerBox->Add(CreateCopyrightPanel(), wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerBox->Add(new wxStaticLine(this), wxSizerFlags().Expand().DoubleHorzBorder());
	pSizerBox->Add(pSizerButtons, wxSizerFlags().Expand());

	SetSizer(pSizerBox);
	pSizerBox->SetSizeHints(this);
}

AboutDialog::~AboutDialog(void)
{
}

wxWindow* AboutDialog::CreateTitle()
{
	wxString strWndBk = GetBackgroundColour().GetAsString(wxC2S_HTML_SYNTAX);

	wxString strVer;
	strVer += wxT("<font size=\"-1\" color=\"#444\">Version ");
	strVer += wxT(MNRT_VERSION);
	strVer += wxT("</font>");

	wxString str;
	str += wxString::Format("<html><body bgcolor=\"%s\">", strWndBk);
	str += wxT("<center>");
	str += wxT("<font size=\"+2\">MNRT</font><br />");
	str += wxT("<font size=\"+1\" color=\"#888\">GPU-based Global Illumination using CUDA</font><br />");
	str += strVer;
	str += wxT("</center>");
	str += wxT("</body></html>");

	wxHtmlWindow* pHTML = new wxHtmlWindow(this, wxID_ANY, 
		wxDefaultPosition, wxSize(450, 300), wxHW_SCROLLBAR_NEVER);
    pHTML->SetBorders(0);
    pHTML->SetPage(str);
    pHTML->SetSize(pHTML->GetInternalRepresentation()->GetWidth(),
                   pHTML->GetInternalRepresentation()->GetHeight());

	return pHTML;
}

wxWindow* AboutDialog::CreateInfo()
{
	wxString strWndBk = GetBackgroundColour().GetAsString(wxC2S_HTML_SYNTAX);

	wxString str;
	str += wxString::Format("<html><body bgcolor=\"%s\">", strWndBk);
	str += wxT("<p>Demonstrative application for the system described in my thesis:</p><br><br>");
	str += wxT("<center><table border=\"0\" width=\"80%\">");
	str += wxT("<tr><td align=\"left\">GPU-basierte globale Beleuchtung mit CUDA in Echtzeit</td></tr>");
	str += wxT("<tr><td align=\"left\">Mathias Neumann</td></tr>");
	str += wxT("<tr><td align=\"left\">Diplomarbeit, FernUniversität in Hagen, 2010</td></tr>");
	str += wxT("</table></center>");
	str += wxT("</body></html>");

	wxHtmlWindow* pHTML = new wxHtmlWindow(this, wxID_ANY, 
		wxDefaultPosition, wxSize(450, 400), wxHW_SCROLLBAR_NEVER);
    pHTML->SetBorders(0);
    pHTML->SetPage(str);
    pHTML->SetSize(pHTML->GetInternalRepresentation()->GetWidth(),
                   pHTML->GetInternalRepresentation()->GetHeight());

	return pHTML;
}

wxWindow* AboutDialog::CreateCopyrightPanel()
{
	wxPanel* pPanelCR = new wxPanel(this);
	wxBoxSizer* pSizer = new wxBoxSizer(wxVERTICAL);

	wxString strWndBk = GetBackgroundColour().GetAsString(wxC2S_HTML_SYNTAX);
	wxString strLicenses = wxString("file://") + wxGetCwd() + wxString("/Help/MNRT_License.html");

	// This is somewhat stupid HTML, however wxHtmlWindow doesn't seem to support
	// a lot of features.
	wxString strCP;
	strCP += wxString::Format("<html><body bgcolor=\"%s\">", strWndBk);
	strCP += wxT("<table border=\"0\" width=\"100%\"><tr>");
	strCP += wxT("<td align=\"left\">Copyright &copy; Mathias Neumann 2010</td>");
	strCP += wxT("<td align=\"right\"><a href=\"http://www.maneumann.com\">www.maneumann.com</a></td>");
	strCP += wxT("</tr></table><br><br>");
	strCP += wxT("<table border=\"0\" width=\"100%\"><tr>");
	strCP += wxT("<tr><td align=\"left\">Used libraries: ");
	strCP += wxT("<a href=\"http://assimp.sourceforge.net\">ASSIMP</a>, ");
	strCP += wxT("<a href=\"http://code.google.com/p/cudpp/\">CUDPP</a>, ");
	strCP += wxT("<a href=\"http://openil.sourceforge.net/\">DevIL</a>, ");
	strCP += wxT("<a href=\"http://www.wxwidgets.org/\">wxWidgets</a>, ");
	strCP += wxT("<a href=\"http://wxpropgrid.sourceforge.net/cgi-bin/index\">wxPropertyGrid</a></td></tr>");
	strCP += wxString::Format("<tr><td align=\"left\">Check the <a href=\"%s\">license</a> document for details.</td></tr>", strLicenses);
	strCP += wxT("</table>");
	strCP += wxT("</body></html>");

	wxHtmlWindow* pHTML = new wxHtmlWindow(pPanelCR, wxID_ANY, 
		wxDefaultPosition, wxSize(450, 300), wxHW_SCROLLBAR_NEVER);
    pHTML->SetBorders(0);
    pHTML->SetPage(strCP);
    pHTML->SetSize(pHTML->GetInternalRepresentation()->GetWidth(),
                   pHTML->GetInternalRepresentation()->GetHeight());

	pSizer->Add(pHTML, wxSizerFlags().Border(wxALL).Expand());

	pPanelCR->SetSizer(pSizer);
	return pPanelCR;
}

void AboutDialog::OnLinkEvent(wxHtmlLinkEvent& event)
{
	const wxHtmlLinkInfo& info = event.GetLinkInfo();
	wxLaunchDefaultBrowser(info.GetHref());
}