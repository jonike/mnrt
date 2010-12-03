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
/// \file	kd-tree\KDTreeListener.h
///
/// \brief	Declares the KDTreeListener class.
/// \author	Mathias Neumann
/// \date	23.07.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_KDTREELISTENER_H__
#define __MN_KDTREELISTENER_H__

#pragma once

struct KDFinalNodeList;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	KDTreeListener
///
/// \brief	Interface for kd-tree construction listeners.
/// 		
/// 		This interface can be used to intercept the kd-tree construction process at given
/// 		places. Currently there is only a way to gain access to the final node list before
/// 		reorganization. 
///
/// \author	Mathias Neumann
/// \date	23.07.2010 \see	KDTreeGPU
////////////////////////////////////////////////////////////////////////////////////////////////////
class KDTreeListener
{
public:
	/// Default constructor.
	KDTreeListener(void) {};
	virtual ~KDTreeListener(void) {};

// Listener methods
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void OnFinalNodeList(KDFinalNodeList* pNodesFinal) = 0
	///
	/// \brief	Is called when the final node list is available, but before node reordering.
	/// 		
	/// 		Can be used to perform additional computations to generate misc data with the help of
	/// 		the final node list. For example, node bounds are still explicitly available here.
	/// 		The listener \em may \em not change the final node node list. 
	///
	/// \author	Mathias Neumann. 
	/// \date	23.07.2010. 
	///
	/// \param [in]		pNodesFinal	The final node list. Implementing classes may not change this
	///								node list.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void OnFinalNodeList(KDFinalNodeList* pNodesFinal) = 0;
};


#endif // __MN_KDTREELISTENER_H__