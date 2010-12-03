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
/// \file	Geometry\MNRay.h
///
/// \brief	Declares the MNRay class. 
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_MNRAY_H__
#define __MN_MNRAY_H__

#pragma once

#include "MNPoint3.h"
#include "MNVector3.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNRay
///
/// \brief	Ray description. 
///
///			A ray consists of a ray origin \a o and a ray direction \a d. We
/// 		assume that the direction is normalized, that is of length 1. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNRay
{
public:
	/// Default constructor. Initializes ray with ray origin (0, 0, 0) and ray direction (1, 0, 0).
	MNRay(void)
		: o(0.f, 0.f, 0.f), d(1.f, 0.f, 0.f)
	{
	}
	/// Constructor that generates ray from given origin and direction.
	MNRay(const MNPoint3& origin, const MNVector3& direction);
	~MNRay(void);

public:
	/// Ray origin.
    MNPoint3 o;
    /// Ray direction (normalized).
    MNVector3 d;

// Operators
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNPoint3 operator()(float t) const
	///
	/// \brief	Function operator. Takes a. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	t	The parameter. Normally larger or equal zero.
	///
	/// \return	A point "on" the ray at distance \a t from #o.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNPoint3 operator()(float t) const
	{
		return o + d * t;
	}

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool IsOnRay(MNPoint3 pt)
	///
	/// \brief	Tests if the given point is part of the ray.
	/// 		
	/// 		This helps to filter out points that lie on the opposite side of the ray. The method
	/// 		uses a small error frame for the ray parameter. This is required due to computation
	/// 		errors and might result to bad behaviour. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	pt	The point to test. 
	///
	/// \return	True if \a pt is on the ray. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool IsOnRay(MNPoint3 pt);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNPoint3 GetPoint(float t)
	///
	/// \brief	Returns the ray point for \a t as parameter. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	t	The parameter. 
	///
	/// \return	The point "on" the ray at distance \a t from #o. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNPoint3 GetPoint(float t);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float Distance(MNPoint3 pt)
	///
	/// \brief	Computes the distance of \a pt from the ray origin #o. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	pt	The point. 
	///
	/// \return	The distance from the ray origin to the given point. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float Distance(MNPoint3 pt) { return ::Distance(pt, o); }
};


#endif // __MN_MNRAY_H__