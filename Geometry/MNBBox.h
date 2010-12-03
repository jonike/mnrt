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
/// \file	Geometry\MNBBox.h
///
/// \brief	Declares the MNBBox class and two global functions. 
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_BBOX_H__
#define __MN_BBOX_H__

#pragma once

#include "../MNMath.h"
#include "MNPoint3.h"
#include <algorithm> // min/max

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNBBox
///
/// \brief	Axis-aligned bounding box (AABB) class. 
///
///			Modelled after \ref lit_pharr "[Pharr and Humphreys 2004]".
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNBBox
{
public:
	/// Default constructor. Creates AABB with minimum in (ininity, infinity, infinity) and 
	/// maximum in (-ininity, -ininity, -ininity).
	MNBBox(void)
	{
		ptMin = MNPoint3( MN_INFINITY,  MN_INFINITY,  MN_INFINITY);
		ptMax = MNPoint3(-MN_INFINITY, -MN_INFINITY, -MN_INFINITY);
	}
	/// Construction from a single point. This creates a degenerate AABB with both minimum
	/// and maximum equal \a p.
	MNBBox(const MNPoint3& p)
		: ptMin(p), ptMax(p)
	{
	}
	/// Construction from two points \a p1 and \a p2. Created AABB is the minimum AABB that
	/// contains both points.
	MNBBox(const MNPoint3& p1, const MNPoint3& p2)
	{
		ptMin = MNPoint3(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
		ptMax = MNPoint3(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
	}
	~MNBBox(void) {}

// Data members
public:
	/// Minimum point.
	MNPoint3 ptMin;
	/// Maximum point.
	MNPoint3 ptMax;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Overlaps(const MNBBox& b) const
	///
	/// \brief	Overlapping test. Checks if there is an intersection between the given bounding box
	/// 		and this bounding box. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	b	The other bounding box. 
	///
	/// \return	\c true if there is an intersection, else \c false. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Overlaps(const MNBBox& b) const 
	{
		bool x = (ptMax.x >= b.ptMin.x) && (ptMin.x <= b.ptMax.x);
		bool y = (ptMax.y >= b.ptMin.y) && (ptMin.y <= b.ptMax.y);
		bool z = (ptMax.z >= b.ptMin.z) && (ptMin.z <= b.ptMax.z);
		return x && y && z;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Contains(const MNPoint3& p) const
	///
	/// \brief	Determines if this bounding box contains a given point. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	p	The point to test. 
	///
	/// \return	\c true if the point is within the AABB, else \c false. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Contains(const MNPoint3& p) const
	{
		return (ptMin.x <= p.x && p.x <= ptMax.x &&
				ptMin.y <= p.y && p.y <= ptMax.y &&
				ptMin.z <= p.z && p.z <= ptMax.z);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float Volume() const
	///
	/// \brief	Computes the volume of the AABB. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \return	The volume. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float Volume() const
	{
		MNVector3 diag = ptMax - ptMin;
		return diag.x * diag.y * diag.z;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	int MaxExtentAxis() const
	///
	/// \brief	Returns the axis where the AABB has the maximum extent. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \return	The axis index, where 0 stands for x-axis, 1 for y-axis and 2 for z-axis. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	int MaxExtentAxis() const
	{
		MNVector3 diag = ptMax - ptMin;
		if(diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if(diag.y > diag.z)
			return 1;
		else
			return 2;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNPoint3 GetCenter() const
	///
	/// \brief	Gets the center of the AABB. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \return	The center point. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNPoint3 GetCenter() const
	{
		return ptMin + 0.5f * (ptMax - ptMin);
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	MNBBox Union(const MNBBox& box, const MNPoint3& p)
///
/// \brief	Calculates the "union" of the given bounding box \a box with a point \a p. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	box	The AABB. 
/// \param	p	The point. 
///
/// \return	The union of AABB and point. That is the minimum AABB that contains \a box and \a p. 
////////////////////////////////////////////////////////////////////////////////////////////////////
MNBBox Union(const MNBBox& box, const MNPoint3& p);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	MNBBox Union(const MNBBox& box1, const MNBBox& box2)
///
/// \brief	Calculates the "union" of two bounding boxes. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	box1	The first AABB. 
/// \param	box2	The second AABB. 
///
/// \return	The union of the two AABBs. That is the minimum AABB that contains both AABBs \a box1
/// 		and \a box2. 
////////////////////////////////////////////////////////////////////////////////////////////////////
MNBBox Union(const MNBBox& box1, const MNBBox& box2);

#endif // __MN_BBOX_H__