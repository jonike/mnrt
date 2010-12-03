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

#include "MNRTSettingsWx.h"
#include "../MNUtilities.h"
#include <wx/config.h>

MNRTSettingsWx::MNRTSettingsWx()
{
}

MNRTSettingsWx::~MNRTSettingsWx()
{
}


bool MNRTSettingsWx::LoadItem(const std::string& strKey, bool* val, bool defaultVal)
{
	return wxConfig::Get()->Read(wxString(strKey), val, defaultVal);
}

bool MNRTSettingsWx::SaveItem(const std::string& strKey, bool val)
{
	return wxConfig::Get()->Write(wxString(strKey), val);
}

bool MNRTSettingsWx::LoadItem(const std::string& strKey, long* val, long defaultVal)
{
	return wxConfig::Get()->Read(wxString(strKey), val, defaultVal);
}

bool MNRTSettingsWx::SaveItem(const std::string& strKey, long val)
{
	return wxConfig::Get()->Write(wxString(strKey), val);
}

bool MNRTSettingsWx::LoadItem(const std::string& strKey, unsigned int* val, unsigned int defaultVal)
{
	MNAssert(val);
	long temp;
	bool result = wxConfig::Get()->Read(wxString(strKey), (long*)&temp, defaultVal);
	// Ensure we have a positive value.
	if(temp < 0)
		*val = defaultVal;
	else
		*val = temp;
	return result;
}

bool MNRTSettingsWx::SaveItem(const std::string& strKey, unsigned int val)
{
	return wxConfig::Get()->Write(wxString(strKey), (long)val);
}

bool MNRTSettingsWx::LoadItem(const std::string& strKey, double* val, double defaultVal)
{
	return wxConfig::Get()->Read(wxString(strKey), val, defaultVal);
}

bool MNRTSettingsWx::SaveItem(const std::string& strKey, double val)
{
	return wxConfig::Get()->Write(wxString(strKey), val);
}

bool MNRTSettingsWx::LoadItem(const std::string& strKey, float* val, float defaultVal)
{
	MNAssert(val);
	double temp;
	bool result = wxConfig::Get()->Read(wxString(strKey), &temp, defaultVal);
	*val = float(std::max(std::min(temp, (double)MN_INFINITY), -(double)MN_INFINITY));

	return result;
}

bool MNRTSettingsWx::SaveItem(const std::string& strKey, float val)
{
	return wxConfig::Get()->Write(wxString(strKey), val);
}