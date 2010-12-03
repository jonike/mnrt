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
/// \file	MNRT\MNMath.h
///
/// \brief	Declares mathematical constants and functions not provided by standard library.
/// \author	Mathias Neumann
/// \date	07.02.2010
/// \ingroup	cpuutil
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_MATH_H__
#define __MN_MATH_H__

#pragma once

#include <math.h>
#include <float.h>	// For FLT_MAX


/// Floating-point value of PI.
#define MN_PI			3.14159265358979323846f
/// Floating-point inverse of PI.
#define MN_INV_PI		0.31830988618379067154f
/// Floating-point inverse of 2*PI.
#define MN_INV_TWOPI	0.15915494309189533577f

/// Floating point infinity.
#define MN_INFINITY		FLT_MAX

/// \brief	Unsigned 32-bit integer type.
///
///			It is important that this is 32-bit wide as some operations, e.g. CUDPP primitives, do
///			not support wider types.
typedef unsigned __int32 uint;
/// Unsigned char type.
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Clamp(float val, float low, float high)
///
/// \brief	Clamps the given value to [\a low, \a high].
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	val		The value. 
/// \param	low		The lower bound. 
/// \param	high	The higher bound.
///
/// \return	The clamped value. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Clamp(float val, float low, float high)
{
	if(val <= low)
		return low;
	else if(val >= high)
		return high;
	else
		return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline int Mod(int a, int b)
///
/// \brief	Computes the remainder of \a a / \a b. 
///
///			This function also works with negative numbers, where
///         the standard % operator is undefined. Implementation taken from
///			\ref lit_pharr "[Pharr and Humphreys 2004]".
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	a	The divident. 
/// \param	b	The divisor.
///
/// \return	The remainder of a / b. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline int Mod(int a, int b)
{
	int n = int(a/b);
	a -= n*b;
	if(a < 0)
		a += b;
	return a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Radians(float degrees)
///
/// \brief	Conversion from degrees to radians.
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	degrees	The angle in degrees. 
///
/// \return	The angle in radians. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Radians(float degrees)
{
	return (MN_PI / 180.f) * degrees;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Degrees(float radians)
///
/// \brief	Conversion from radians to degrees.
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	radians	The angle in radians. 
///
/// \return	The angle in degrees. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Degrees(float radians)
{
	return (180.f / MN_PI) * radians;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline float Log2(float x)
///
/// \brief	Logarithm of \a x to base 2. Not available in standard library.
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	x	The input. 
///
/// \return	Base-2 logarithm of x. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline float Log2(float x)
{
	static float invLog2 = 1.f / logf(2.f);
	return logf(x) * invLog2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline int Log2Int(float x)
///
/// \brief	Computes the integral base-2 logarithm of \a x.
///
///			Exploits IEEE floating-point layout to compute the integral logarithm of x without
///         computing a real logarithm. Implementation taken from
///			\ref lit_pharr "[Pharr and Humphreys 2004]".
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	x	The value to take the logarithm from. 
///
/// \return	Integral logarithm of \a x to base 2. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline int Log2Int(float x)
{
	return ((*(int*)&x) >> 23) - 127;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline bool IsPowerOf2(int x)
///
/// \brief	Checks if \a x is power of 2.
/// 		
/// 		Implementation taken from \ref lit_pharr "[Pharr and Humphreys 2004]". 
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	x	The value to check. 
///
/// \return	\c true if power of 2, \c false if not. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline bool IsPowerOf2(int x)
{
	return (x & (x - 1)) == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline uint RoundUpPow2(uint x)
///
/// \brief	Rounds \a x up to the next power of 2.
/// 		
/// 		Implementation taken from \ref lit_pharr "[Pharr and Humphreys 2004]". 
///
/// \author	Mathias Neumann
/// \date	07.02.2010
///
/// \param	x	The value to round. 
///
/// \return	The next power of two greater or equal \a x. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline uint RoundUpPow2(uint x)
{
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x+1;
}

#endif // __MN_MATH_H__