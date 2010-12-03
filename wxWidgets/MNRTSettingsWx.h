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
/// \file	wxWidgets\MNRTSettingsWx.h
///
/// \brief	Declares the MNRTSettingsWx class. 
/// \author	Mathias Neumann
/// \date	08.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_MNRTSETTINGSWX_H__
#define __MN_MNRTSETTINGSWX_H__

#pragma once

#include "../MNRTSettings.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNRTSettingsWx
///
/// \brief	Implementation of MNRTSettings for wxWidgets.
/// 		
/// 		Uses the \c wxConfig class of wxWidgets to write the settings to the application's
/// 		configuration (e.g. ini file or registry). 
///
/// \author	Mathias Neumann
/// \date	08.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNRTSettingsWx : public MNRTSettings
{
public:
	/// Default constructor.
	MNRTSettingsWx();
	virtual ~MNRTSettingsWx();

public:
	virtual bool LoadItem(const std::string& strKey, bool* val, bool defaultVal);
	virtual bool SaveItem(const std::string& strKey, bool val);
	virtual bool LoadItem(const std::string& strKey, long* val, long defaultVal);
	virtual bool SaveItem(const std::string& strKey, long val);
	virtual bool LoadItem(const std::string& strKey, unsigned int* val, unsigned int defaultVal);
	virtual bool SaveItem(const std::string& strKey, unsigned int val);
	virtual bool LoadItem(const std::string& strKey, double* val, double defaultVal);
	virtual bool SaveItem(const std::string& strKey, double val);
	virtual bool LoadItem(const std::string& strKey, float* val, float defaultVal);
	virtual bool SaveItem(const std::string& strKey, float val);
};

#endif // __MN_MNRTSETTINGSWX_H__