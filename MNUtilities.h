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
/// \file	MNRT\MNUtilities.h
///
/// \brief	Declares error reporting utilities.
/// \ingroup	cpuutil
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	cpuutil	CPU-based General Utilities
/// 
/// \brief	General, CPU-based utility components of MNRT. 
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_UTILITIES_H__
#define __MN_UTILITIES_H__

#pragma once

#include <assert.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void MNFatal(const char *strFormat, ...)
///
/// \brief	Fatal error reporting procedure. Shows error to the user and quits the program.
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	strFormat	An error string for the user.
////////////////////////////////////////////////////////////////////////////////////////////////////
void MNFatal(const char *strFormat, ...);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void MNError(const char *strFormat, ...)
///
/// \brief	Error reporting procedure. Shows error to the user, but does not quit the program. 
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	strFormat	An error string for the user. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void MNError(const char *strFormat, ...);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void MNWarning(const char *strFormat, ...)
///
/// \brief	Warning reporting procedure. Does not quit the application.
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	strFormat	An warning string for the user.
////////////////////////////////////////////////////////////////////////////////////////////////////
void MNWarning(const char *strFormat, ...);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void MNMessage(const char *strFormat, ...)
///
/// \brief	Reports a message string to the user. Does not quit or interrupt the application	
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	strFormat	The message string. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void MNMessage(const char *strFormat, ...);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNAssert(expr)
///
/// \brief	A custom assertion macro that uses MNFatal().
///
/// \author	Mathias Neumann
/// \date	09.02.2010
///
/// \param	expr	The asserted expression.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef _DEBUG
#define MNAssert(expr) \
			if((expr) == false) \
			{ MNFatal("Failed assertion at %s (line %d).\n", __FILE__, __LINE__); }
#else
#define MNAssert(expr) \
			((expr) ? ((void)0) : \
			assert(expr))
#endif // _DEBUG


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	#define SAFE_DELETE(p)
///
/// \brief	Performs a safe deletion on a given pointer to make the dynamic memory available
///			again. Safe because it is checked if the pointer \a p is \c NULL. The deletion is
///			performed only when \a p isn't \c NULL. In any case, thereafter \a p is set to \c NULL.
///
/// \author	Mathias Neumann
/// \date	09.02.2010
///
/// \param	p	A non-array pointer. Can be \c NULL.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define SAFE_DELETE(p)  { if(p) delete (p); (p) = NULL; }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	SAFE_DELETE_ARRAY(p)
///
/// \brief	Performs a safe deletion on a pointer to an array to make the dynamic memory
/// 		available again. Safe because it is checked if the pointer \a p is \c NULL. The
/// 		deletion is performed only when \a p isn't \c NULL. In any case, thereafter \a p is
/// 		set to \c NULL. 
///
/// \author	Mathias Neumann. 
/// \date	09.02.2010. 
///
/// \remarks	Mathias Neumann, 18.10.2010. 
///
/// \param	p	The array pointer to delete. Can be \c NULL.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define SAFE_DELETE_ARRAY(p)  { if(p) delete [] (p); (p) = NULL; }


#endif // __MN_UTILITIES_H__