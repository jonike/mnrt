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
/// \file	Geometry\MNVector3.h
///
/// \brief	Declares the MNVector3 class and global utility functions for vectors. 
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_VECTOR3_H__
#define __MN_VECTOR3_H__

#pragma once

#include <math.h>
#include "../MNUtilities.h"

class MNNormal3;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNVector3
///
/// \brief	Three-dimensional floating point vector class. 
///
///			Modelled after \ref lit_pharr "[Pharr and Humphreys 2004]".
///
/// 		\warning Do not add members or change member order as the current order ensures
/// 				 casting to CUDA compatible structs works. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNVector3
{
public:
	/// Default constructor.
	MNVector3(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	/// Explicit conversion from a normal.
	explicit MNVector3(const MNNormal3& n);
	~MNVector3(void) {}

// Data members
public:
	/// x-coordinate of vector.
	float x;
	/// y-coordinate of vector.
	float y;
	/// z-coordinate of vector.
	float z;

// Operators
public:
	/// Vector addition operator.
	MNVector3 operator+(const MNVector3& v) const
	{
		return MNVector3(x + v.x, y + v.y, z + v.z);
	}
	/// Vector addition assignment operator.
	MNVector3& operator+=(const MNVector3& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	/// Vector subtraction operator.
	MNVector3 operator-(const MNVector3& v) const
	{
		return MNVector3(x - v.x, y - v.y, z - v.z);
	}
	/// Vector subtraction assignment operator.
	MNVector3& operator-=(const MNVector3& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	/// Vector negation operator.
	MNVector3 operator-() const 
	{
		return MNVector3(-x, -y, -z);
	}

	/// Vector scaling (by scalar) operator.
	MNVector3 operator*(float f) const
	{
		return MNVector3(f*x, f*y, f*z);
	}
	/// Vector scaling (by scalar) assignment operator.
	MNVector3& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Vector division (by scalar) operator.
	MNVector3 operator/(float f) const
	{
		MNAssert(f != 0);
		float inv = 1.f / f;
		return MNVector3(x*inv, y*inv, z*inv);
	}
	/// Vector division (by scalar) assignment operator.
	MNVector3& operator/=(float f)
	{
		MNAssert(f != 0);
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
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

public:
	/// Computes the squared length of the vector. Might be used to avoid the square root operation.
	float LengthSquared() const { return x*x + y*y + z*z; }

	/// Computes the length of this vector. Includes square root operation.
	float Length() const { return sqrtf(LengthSquared()); }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline MNVector3 operator*(float f, const MNVector3& v)
///
/// \brief	Scaling of a vector \a v by a scalar \a f from the left side.
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	f	The scalar. 
/// \param	v	The vector. 
///
/// \return	The scaled vector.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline MNVector3 operator*(float f, const MNVector3& v)
{
	return v * f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline MNVector3 Normalize(const MNVector3& v)
///
/// \brief	Normalizes the given vector \a v and returns the result.
/// 		
/// 		\warning This is no inplace operation! 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v	The vector. 
///
/// \return	Normalized vector. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline MNVector3 Normalize(const MNVector3& v)
{
	return v / v.Length();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNVector3& v1, const MNVector3& v2)
///
/// \brief	Calculates the dot product (inner product, scalar product) between two vectors \a v1
/// 		and \a v2. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v1	The first vector. 
/// \param	v2	The second vector. 
///
/// \return	Dot product between the two vectors. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNVector3& v1, const MNVector3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float AbsDot(const MNVector3& v1, const MNVector3& v2)
///
/// \brief	Computes the dot product of \a v1 and \a v2 and takes the absolute value of the
/// 		result. It is therefore a composition of \a fabsf() and ::Dot. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v1	The first vector. 
/// \param	v2	The second vector. 
///
/// \return	Absolute dot produkt of the two vectors. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float AbsDot(const MNVector3& v1, const MNVector3& v2)
{
	return fabsf(Dot(v1, v2));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline MNVector3 Cross(const MNVector3& v1, const MNVector3& v2)
///
/// \brief	Determines the cross product of two vectors /a v1 and /a v2.
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v1	The first vector. 
/// \param	v2	The second vector. 
///
/// \return	The cross product. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline MNVector3 Cross(const MNVector3& v1, const MNVector3& v2)
{
	return MNVector3((v1.y * v2.z) - (v1.z * v2.y),
					 (v1.z * v2.x) - (v1.x * v2.z),
					 (v1.x * v2.y) - (v1.y * v2.x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline void CoordinateSystem(const MNVector3& v1, MNVector3* pV2, MNVector3* pV3)
///
/// \brief	Coordinate system construction from a single vector. Two additional vectors are
/// 		generated to create a valid orthogonal coordinate system with \a v1 and the two
/// 		generated vectors \a pV2 and \a pV3 as coordinate axes. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v1			Input vector. 
/// \param [out]	pV2	Second generated vector of the coordinate system. 
/// \param [out]	pV3	Third generated vector of the coordinate system. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline void CoordinateSystem(const MNVector3& v1, MNVector3* pV2, MNVector3* pV3)
{
	// Compute second vector by zeroing one component and swapping the others.
	if(fabsf(v1.x) > fabsf(v1.y))
	{
		float invLength = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
		*pV2 = MNVector3(-v1.z * invLength, 0.f, v1.x * invLength);
	}
	else
	{
		float invLength = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
		*pV2 = MNVector3(0.f, v1.z * invLength, -v1.y * invLength);
	}
	*pV3 = Cross(v1, *pV2);
}

#endif // __MN_VECTOR3_H__