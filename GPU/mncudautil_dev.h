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
/// \file	GPU\mncudautil_dev.h
///
/// \brief	Contains device functions used for mncudautil.cu.
///
///	\note	This file can be included in multiple cu-files!
///
/// \author	Mathias Neumann
/// \date	01.04.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_CUDAUTIL_DEV_H__
#define __MN_CUDAUTIL_DEV_H__

#include "MNMath.h"
#include "MNCudaUtil.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __device__ T dev_ReduceSimple(T* s_data, uint count, MNCudaOP op)
///
/// \brief	Very simple reduction device function.
/// 		
/// 		Usually called on data stored in shared memory. It is assumed that there are enough
/// 		threads so that each thread can handle one data element. 
///
/// \author	Mathias Neumann
/// \date	18.02.2010
///
/// \tparam	T	Type of elements.
///
/// \param [in,out]	s_data	The data to reduce. Usually located in shared memory. Array is
/// 						changed as it is used for the algorithm. 
/// \param	count			Number of elements in \a s_data. 
/// \param	op				The reduction operation. 
///
/// \return	The reduced value. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__device__ T dev_ReduceSimple(T* s_data, uint count, MNCudaOP op)
{
	uint tid = threadIdx.x;
	for(uint s=blockDim.x/2; s>0; s>>=1) 
	{
		if(tid < s)
		{
			if(op == MNCuda_ADD)
				s_data[tid] = s_data[tid] + s_data[tid + s];
			else if(op == MNCuda_MIN)
				s_data[tid] = min(s_data[tid], s_data[tid + s]);
			else if(op == MNCuda_MAX)
				s_data[tid] = max(s_data[tid], s_data[tid + s]);
		}
		__syncthreads();
	}

	return s_data[0];
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, uint blockSize, class opTraits> __device__ T dev_ReduceFast(T* s_data)
///
/// \brief	Fast parallel reduction using loop-unrolling and sync-free operations within warps. 
///
///			Implemented after NVIDIA's CUDA SDK example.
///
/// \author	Mathias Neumann
/// \date	24.10.2010
///
/// \tparam	T			Type of elements. \c float4 or other structs are not supported for \c T because
/// 					we use \c volatile \c T to avoid compiler optimizations when doing work within a
/// 					warp. This is required for Fermi GPUs. 
/// \tparam blockSize	Thread block size to use. Maximum is 512. Should be power of two.
/// \tparam	opTraits	Operator to use. See ReduceOperatorTraits.
///
/// \param [in,out]	s_data	The data to reduce. Array size is given by template parameter. However,
///							reduction of smaller arrays is possible by padding the array with
///							identity elements. Note that the operation uses the input aray, hence
///							it is changed!
///
/// \return	The reduced value.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, uint blockSize, class opTraits>
__device__ T dev_ReduceFast(T* s_data)
{
#ifdef __DEVICE_EMULATION__
	// Device emulation does not like no-sync within warps.
	return dev_ReduceSimple(s_data, blockSize, op);
#else
	uint tid = threadIdx.x;

	if(blockSize >= 512)
	{
		if (tid < 256)
			s_data[tid] = opTraits::op(s_data[tid], s_data[tid + 256]);
		__syncthreads();
	}

	if(blockSize >= 256)
	{
		if (tid < 128)
			s_data[tid] = opTraits::op(s_data[tid], s_data[tid + 128]);
		__syncthreads();
	}

	if(blockSize >= 128)
	{
		if (tid < 64) 
			s_data[tid] = opTraits::op(s_data[tid], s_data[tid + 64]);
		__syncthreads();
	}

	// Instructions are SIMD synchronous within a warp (32 threads). Therefore no
	// synchronization is required here.
	if(tid < 32)
	{
		// Use volatile to avoid compiler optimizations that reorder the store operations.
		volatile T* sv_data = s_data;

		if(blockSize >= 64)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 32]);
		if(blockSize >= 32)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 16]);
		if(blockSize >= 16)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 8]);
		if(blockSize >= 8)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 4]);
		if(blockSize >= 4)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 2]);
		if(blockSize >= 2)
			sv_data[tid] = opTraits::op(sv_data[tid], sv_data[tid + 1]);
	}

	// Need a sync here since only then all threads will return the same value.
	__syncthreads();
	return s_data[0];
#endif // NOT __DEVICE_EMULATION__
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ uint dev_CountBits(uint value)
///
/// \brief	Counts the 1-bits using parallel bit counting.
///
///			Used algorithm is described in http://infolab.stanford.edu/~manku/bitcount/bitcount.c.
///
/// \author	Mathias Neumann
/// \date	26.02.2010
///
/// \param	value	Input value.
///
/// \return	Number of bits set in \a value. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ uint dev_CountBits(uint value)
{
#ifndef DOXYGEN_IGNORE
#define TWO(c) (0x1u << (c))
#define MASK(c) (((unsigned long long int)(-1)) / (TWO(TWO(c)) + 1u))
#define COUNT(x,c) ((x) & MASK(c)) + (((x) >> (TWO(c))) & MASK(c))
#endif // DOXYGEN_IGNORE

	value = COUNT(value, 0);
    value = COUNT(value, 1);
    value = COUNT(value, 2);
    value = COUNT(value, 3);
    value = COUNT(value, 4);
    // value = COUNT(value, 5); Shift count too large warning when compiling...
    return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ uint dev_CountBits(unsigned long long int value)
///
/// \brief	Counts the 1-bits using parallel bit counting.
///
///			See ::dev_CountBits().
///
/// \author	Mathias Neumann
/// \date	26.02.2010
///
/// \param	value	Input value.
///
/// \return	Number of bits set in \a value.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ uint dev_CountBits(unsigned long long int value)
{
	uint result = dev_CountBits((uint)(value & 0x00000000FFFFFFFF));
	result += dev_CountBits((uint)((value & 0xFFFFFFFF00000000) >> 32));
    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float dev_RadicalInverse(int n, int base)
///
/// \brief	Computes the radical inverse function for the given parameters.
/// 		
/// 		The radical inverse function Phi_{base}(n) converts a nonnegative integer n to a
/// 		floating-point value in [0, 1) by reflecting the digits about the decimal point. That
/// 		is, if n = sum(d_i * base^{i}), then Phi_{base}(n) = 0.d_{1}d_{2}...d_{m}.
/// 		
/// 		This function is useful to compute several low-discrepancy sequences, for example the
/// 		Halton or Hammersley sequences. See \ref lit_pharr "[Pharr and Humphreys 2004]",
/// 		p. 319. 
///
/// \author	Mathias Neumann
/// \date	07.04.2010
///
/// \param	n		The nonnegative integer value. 
/// \param	base	The base. 
///
/// \return	The radical inverse of \a n to base \a base, Phi_{\a base}(\a n). 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_RadicalInverse(int n, int base)
{
	float value = 0.f;
	float invBase = 1.f / base;
	float invBase_i = invBase;

	while(n > 0)
	{
		// Compute next digit of radical inverse.
		int nByBase = n / base;
		int d_i = n - nByBase*base; // n % base
		value += d_i * invBase_i;
		n = nByBase;
		invBase_i *= invBase;
	}

	return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ uint dev_RandomLCG(uint xn)
///
/// \brief	Linear congruential random number generator (RNG).
/// 		
/// 		Computes (a * xn + b) % 2^32 to generate new numbers. a and b are chosen as in
/// 		Numerical Recipes, see http://en.wikipedia.org/wiki/Linear_congruential_generator. 
///
/// \author	Mathias Neumann
/// \date	11.04.2010
///
/// \param	xn	Input number. 
///
/// \return	The next number generated by the LCG. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ uint dev_RandomLCG(uint xn)
{
	unsigned long res = 1664525UL * ((unsigned long)xn) + 1013904223;
	return (uint)res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_RandomLCGUniform(uint xn)
///
/// \brief	Generates uniform random numbers using the LCG.
///
///			See ::dev_RandomLCG().
///
/// \author	Mathias Neumann
/// \date	11.04.2010
///
/// \param	xn	Input number.
///
/// \return	Uniform random number in [0, 1]. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_RandomLCGUniform(uint xn)
{
	return dev_RandomLCG(xn) / 4294967296.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float dev_DistanceSquared(float3 p, float3 q)
///
/// \brief	Computes the squared distance of two points. 
///
/// \author	Mathias Neumann
/// \date	08.04.2010
///
/// \param	p	First point. 
/// \param	q	Second point. 
///
/// \return	The squared distance. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_DistanceSquared(float3 p, float3 q)
{
	float3 diff = p - q;
    return dot(diff, diff);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_Spherical2Direction(float phi, float theta)
///
/// \brief	Converts spherical coordinates (phi, theta) to normalized direction (x, y, z).
///
/// \author	Mathias Neumann
/// \date	06.07.2010
///
/// \param	phi		The azimuthal angle, 0 <= phi < 2PI. 
/// \param	theta	The polar angle, 0 <= theta <= PI. 
///
/// \return	Normalized direction. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_Spherical2Direction(float phi, float theta)
{
	float3 dir;
	float sinTheta = sinf(theta);
	dir.x = sinTheta * cosf(phi);
	dir.y = sinTheta * sinf(phi);
	dir.z = cosf(theta);
	return dir;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float2 dev_Direction2Spherical(const float3& dir)
///
/// \brief	Converts normalized direction (x, y, z) to spherical coordinates (phi, theta).
///
/// \author	Mathias Neumann
/// \date	06.07.2010
///
/// \param	dir	Normalized direction to convert.
///
/// \return	Spherical coordinates (azimuthal, polar). 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float2 dev_Direction2Spherical(const float3& dir)
{
	float2 spherical;
	spherical.x = atan2f(dir.y, dir.x);
	if(spherical.x < 0.f)
		spherical.x += 2.f*MN_PI;
	spherical.y = acosf(dir.z);
	return spherical;
}


//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __MN_CUDAUTIL_DEV_H__