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
/// \file	Geometry\MNPlane.h
///
/// \brief	Declares the MNPlane class.
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_MNPLANE_H__
#define __MN_MNPLANE_H__

#pragma once

#include "MNPoint3.h"
#include "MNNormal3.h"
#include "MNRay.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNPlane
///
/// \brief	Three-dimensional plane representation.
///
///			A plane is defined by a point #m_ptPlane, a normal #m_N and the parameter #m_D, where
///			Dot(#m_N, #m_ptPlane) + #m_D = 0.
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNPlane
{
public:
	/// Constructs a plane from a given point \a pt and normal \a n.
	MNPlane(const MNPoint3& pt, const MNNormal3& n);
	~MNPlane(void);

private:
	/// A point on the plane.
	MNPoint3 m_ptPlane;
	/// Normal vector of the plane. Might be unnormalized.
	MNNormal3 m_N;
	/// Plane equation parameter.
    float m_D;

public:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \fn	bool Intersect(const MNRay& ray, MNPoint3& ptIntersect)
    ///
    /// \brief	Checks if the given ray intersects this plane. Returns the intersection point (if
    /// 		any) in the provided point. 
    ///
    /// \author	Mathias Neumann
    /// \date	06.02.2010
    ///
    /// \param	ray					The ray. 
    /// \param [out]	ptIntersect	The intersection point. Only valid when \c true is returned. 
    ///
    /// \return	\c true if there is an intersection and \a ptIntersect contains this intersection.
    /// 		Else false. 
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    bool Intersect(const MNRay& ray, MNPoint3& ptIntersect);
};

#endif // __MN_MNPLANE_H__