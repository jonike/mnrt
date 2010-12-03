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
/// \file	Geometry\MNPoint3.h
///
/// \brief	Declares the MNPoint3 class and some global utility functions. 
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_POINT3_H__
#define __MN_POINT3_H__

#pragma once

#include <math.h>
#include "MNVector3.h"
#include "MNNormal3.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNPoint3
///
/// \brief	Three-dimensional floating point points class.
/// 		
/// 		Modelled after \ref lit_pharr "[Pharr and Humphreys 2004]".
/// 		
/// 		\warning Do not add members or change member order as the current order ensures
/// 				 casting to CUDA compatible structs works. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNPoint3
{
public:
	/// Default constructor.
	MNPoint3(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	~MNPoint3(void) {}

// Data members
public:
	/// x-coordinate of point.
	float x;
	/// y-coordinate of point.
	float y;
	/// z-coordinate of point.
	float z;

// Operators
public:
	/// Point addition operator.
	MNPoint3 operator+(const MNVector3& v) const
	{
		return MNPoint3(x + v.x, y + v.y, z + v.z);
	}
	/// Point addition assignment operator.
	MNPoint3& operator+=(const MNVector3& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	/// Point subtraction operator.
	MNPoint3 operator-(const MNVector3& v) const
	{
		return MNPoint3(x - v.x, y - v.y, z - v.z);
	}
	/// Point subtraction assignment operator.
	MNPoint3& operator-=(const MNVector3& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	/// Point scaling (by scalar) operator.
	MNPoint3 operator*(float f) const
	{
		return MNPoint3(f*x, f*y, f*z);
	}
	/// Point scaling (by scalar) assignment operator.
	MNPoint3& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Point division (by scalar) operator.
	MNPoint3 operator/(float f) const
	{
		MNAssert(f != 0);
		float inv = 1.f / f;
		return MNPoint3(x*inv, y*inv, z*inv);
	}
	/// Point division (by scalar) assignment operator.
	MNPoint3& operator/=(float f)
	{
		MNAssert(f != 0);
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	/// Vector generation operator. Creates an MNVector3 by subtracting the
	/// given point \a p from this point and returning the result.
	MNVector3 operator-(const MNPoint3& p) const
	{
		return MNVector3(x - p.x, y - p.y, z - p.z);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float operator[](int i) const
	///
	/// \brief	Component-wise access operator.
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	i	Component index. Has to be 0, 1 or 2.
	///
	/// \return	The component (as constant value).
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float operator[](int i) const
	{
		MNAssert(i >= 0 && i < 3);
		return (&x)[i];
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float& operator[](int i)
	///
	/// \brief	Component-wise access operator.
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	i	Component index. Has to be 0, 1 or 2.
	///
	/// \return	The component (as reference).
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float& operator[](int i)
	{
		MNAssert(i >= 0 && i < 3);
		return (&x)[i];
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float DistanceSquared(const MNPoint3& p1, const MNPoint3& p2)
///
/// \brief	Computes the squared distance between two points and therefore avoids the square root
/// 		operation. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	p1	The first point. 
/// \param	p2	The second point. 
///
/// \return	Squared distance between \a p1 and \a p2. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float DistanceSquared(const MNPoint3& p1, const MNPoint3& p2)
{
	return (p2 - p1).LengthSquared();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Distance(const MNPoint3& p1, const MNPoint3& p2)
///
/// \brief	Computes the distance between two points \a p1 and \a p2. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	p1	The first point.
/// \param	p2	The second point.
///
/// \return	Distance between the points. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Distance(const MNPoint3& p1, const MNPoint3& p2)
{
	return (p2 - p1).Length();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNPoint3& p, const MNNormal3& n)
///
/// \brief	Dot product between point and normal (for plane calculations).
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	p	The point. 
/// \param	n	The normal. 
///
/// \return	The dot product between the two "vectors". 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNPoint3& p, const MNNormal3& n)
{
	return p.x * n.x + p.y * n.y + p.z * n.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNNormal3& n, const MNPoint3& p)
///
/// \brief	Dot product between normal and point (for plane calculations).
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n	The normal. 
/// \param	p	The point. 
///
/// \return	The dot product between the two "vectors".
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNNormal3& n, const MNPoint3& p)
{
	return Dot(p, n);
}

#endif // __MN_POINT3_H__