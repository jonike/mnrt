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
/// \file	MNRT\ProgressListener.h
///
/// \brief	Declares the ProgressListener class. 
/// \author	Mathias Neumann
/// \date	05.10.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_PROGRESSLISTENER_H__
#define __MN_PROGRESSLISTENER_H__

#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	ProgressListener
///
/// \brief	Progress listener interface.
///
///			Leaves open which UI is used by providing only an interface.
///
/// \author	Mathias Neumann
/// \date	05.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class ProgressListener
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void SetMaximum(int maxValue)
	///
	/// \brief	Sets the maximum value for the progress listener.
	///
	/// \author	Mathias Neumann. 
	/// \date	05.10.2010. 
	///
	/// \param	maxValue	The maximum value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void SetMaximum(int maxValue) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool Update(int newValue, const std::string& strNewMessage = "") = 0
	///
	/// \brief	Updates the progress to a new value. 
	///
	/// \author	Mathias Neumann. 
	/// \date	05.10.2010. 
	///
	/// \param	newValue		The new value. 
	/// \param	strNewMessage	New progress message. 
	///
	/// \return	\c true if the user wants to continue the progress, else \c false (abort button hit). 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool Update(int newValue, const std::string& strNewMessage = "") = 0;
};


#endif // __MN_PROGRESSLISTENER_H__