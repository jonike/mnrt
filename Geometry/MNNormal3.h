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
/// \file	Geometry\MNNormal3.h
///
/// \brief	Declares the MNNormal3 class and global utility functions.
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_NORMAL3_H__
#define __MN_NORMAL3_H__

#pragma once

#include <math.h>
#include "../MNUtilities.h"
#include "MNVector3.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNNormal3
///
/// \brief	Three-dimensional floating point normal class.
/// 		
/// 		Modelled after \ref lit_pharr "[Pharr and Humphreys 2004]". We
/// 		distinguish between normals and vectors to emphasize the differences between both. 
///
/// 		\warning Do not add members or change member order as the current order ensures
/// 				 casting to CUDA compatible structs works. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNNormal3
{
public:
	/// Default constructor.
	MNNormal3(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	/// Explicit conversion from a vector. 
	explicit MNNormal3(const MNVector3& v)	
		: x(v.x), y(v.y), z(v.z)
	{
	}
	~MNNormal3(void) {}

// Data members
public:
	/// x-coordinate of normal.
	float x;
	/// y-coordinate of normal.
	float y;
	/// z-coordinate of normal.
	float z;

// Operators
public:
	/// Normal addition operator.
	MNNormal3 operator+(const MNNormal3& n) const
	{
		return MNNormal3(x + n.x, y + n.y, z + n.z);
	}
	/// Normal addition assignment operator.
	MNNormal3& operator+=(const MNNormal3& n)
	{
		x += n.x; y += n.y; z += n.z;
		return *this;
	}

	/// Normal subtraction operator.
	MNNormal3 operator-(const MNNormal3& n) const
	{
		return MNNormal3(x - n.x, y - n.y, z - n.z);
	}
	/// Normal subtraction assignment operator.
	MNNormal3& operator-=(const MNNormal3& n)
	{
		x -= n.x; y -= n.y; z -= n.z;
		return *this;
	}

	/// Normal negation operator.
	MNNormal3 operator-() const 
	{
		return MNNormal3(-x, -y, -z);
	}

	/// Normal scaling (by scalar) operator.
	MNNormal3 operator*(float f) const
	{
		return MNNormal3(f*x, f*y, f*z);
	}
	/// Normal scaling (by scalar) assignment operator.
	MNNormal3& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Normal division (by scalar) operator.
	MNNormal3 operator/(float f) const
	{
		MNAssert(f != 0);
		float inv = 1.f / f;
		return MNNormal3(x*inv, y*inv, z*inv);
	}
	/// Normal division (by scalar) assignment operator.
	MNNormal3& operator/=(float f)
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
	/// Computes the squared length of the normal. Might be used to avoid the square root operation.
	float LengthSquared() const { return x*x + y*y + z*z; }

	/// Computes the length of this normal. Includes square root operation.
	float Length() const { return sqrtf(LengthSquared()); }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline MNNormal3 operator*(float f, const MNNormal3& n)
///
/// \brief	Scaling of a normal \a n by a scalar \a f from the left side. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	f	The scalar. 
/// \param	n	The normal. 
///
/// \return	The scaled normal. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline MNNormal3 operator*(float f, const MNNormal3& n)
{
	return n * f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline MNNormal3 Normalize(const MNNormal3& n)
///
/// \brief	Normalizes the given normal \a n and returns the result.
/// 		
/// 		\warning This is no inplace operation! 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n	The normal. 
///
/// \return	Normalized normal. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline MNNormal3 Normalize(const MNNormal3& n)
{
	return n / n.Length();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNNormal3& n1, const MNNormal3& n2)
///
/// \brief	Calculates the dot product (inner product, scalar product) between two normals \a n1
/// 		and \a n2. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n1	The first normal. 
/// \param	n2	The second normal. 
///
/// \return	Dot product between the two normals. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNNormal3& n1, const MNNormal3& n2)
{
	return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNNormal3& n, const MNVector3& v)
///
/// \brief	Calculates the dot product (inner product, scalar product) between a normal \a n
/// 		and a vector \a v. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n	The normal. 
/// \param	v	The vector. 
///
/// \return	Dot product between normal and vector.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNNormal3& n, const MNVector3& v)
{
	return n.x * v.x + v.y * n.y + v.z * n.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Dot(const MNVector3& v, const MNNormal3& n)
///
/// \brief	Calculates the dot product (inner product, scalar product) between a vector \a v and
/// 		a normal \a n. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v	The vector. 
/// \param	n	The normal. 
///
/// \return	Dot product between vector and normal. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Dot(const MNVector3& v, const MNNormal3& n)
{
	return Dot(n, v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float AbsDot(const MNNormal3& n1, const MNNormal3& n2)
///
/// \brief	Computes the dot product of \a n1 and \a n2 and takes the absolute value of the
/// 		result. It is therefore a composition of \a fabsf() and ::Dot.  
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n1	The first normal. 
/// \param	n2	The second normal. 
///
/// \return	The absolute dot product. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float AbsDot(const MNNormal3& n1, const MNNormal3& n2)
{
	return fabsf(Dot(n1, n2));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float AbsDot(const MNNormal3& n, const MNVector3& v)
///
/// \brief	Computes the dot product of \a n and \a v and takes the absolute value of the
/// 		result. It is therefore a composition of \a fabsf() and ::Dot. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	n	The normal. 
/// \param	v	The vector. 
///
/// \return	The absolute dot product. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float AbsDot(const MNNormal3& n, const MNVector3& v)
{
	return fabsf(Dot(n, v));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float AbsDot(const MNVector3& v, const MNNormal3& n)
///
/// \brief	Computes the dot product of \a v and \a n and takes the absolute value of the result.
/// 		It is therefore a composition of \a fabsf() and ::Dot. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010
///
/// \param	v	The vector. 
/// \param	n	The normal. 
///
/// \return	The absolute dot product. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float AbsDot(const MNVector3& v, const MNNormal3& n)
{
	return fabsf(Dot(n, v));
}

#endif // __MN_NORMAL3_H__