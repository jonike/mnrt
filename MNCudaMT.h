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
/// \file	MNRT\MNCudaMT.h
///
/// \brief	Declares the MNCudaMT class.
/// \author	Mathias Neumann
/// \date	11.04.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_CUDA_MT_H__
#define __MN_CUDA_MT_H__

#pragma once

#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNCudaMT
///
/// \brief	CUDA Mersenne Twister implementation. Hides NVIDIA's implementation from CUDA SDK 3.0.
///
///			Class is designed as singleton and might need optimizations for when used from
///			multiple CPU-threads.
///
/// \see	http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/MersenneTwister/doc/MersenneTwister.pdf
///
/// \author	Mathias Neumann
/// \date	11.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNCudaMT
{
	// Singleton. Hide constructors.
private:
	MNCudaMT(void);
	MNCudaMT(const MNCudaMT& other);

public:
	~MNCudaMT(void);

private:
	/// Keeps track of the initialization state.
	bool m_bInited;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNCudaMemPool& GetInstance()
	///
	/// \brief	Returns the only instance.
	///
	/// \warning Not thread-safe!
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	///
	/// \return	The instance. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNCudaMT& GetInstance();


public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Init(const char *fname) }
	///
	/// \brief	Initialises this object from a configuration file.
	///
	/// \author	Mathias Neumann
	/// \date	11.04.2010
	///
	/// \param	fname	File path of the configuration file. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Init(const char *fname);

	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Seed(unsigned int seed)
	///
	/// \brief	Seeds Mersenne Twister for current GPU context. 
	///
	/// \author	Mathias Neumann
	/// \date	11.04.2010
	///
	/// \param	seed	The seed value. 
	///
	/// \return	\c true if it succeeds, \c false if it fails.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Seed(unsigned int seed);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Generate(float* d_outRand, int count)
	///
	/// \brief	Performs Mersenne Twister RNG to generate a predefined number of uniform random
	/// 		numbers for use in other kernels. 
	///
	/// \author	Mathias Neumann
	/// \date	11.04.2010
	/// \see	GetAlignedCount()
	///
	/// \param [out]	d_outRand	The generated uniform random numbers. Device memory provided by
	///								caller.
	/// \param	count				Number of randoms to generate. Use GetAlignedCount() to get
	///								an appropriate aligned count.
	///
	/// \return	\c cudaSuccess if it succeeds, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Generate(float* d_outRand, int count);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	unsigned int GetAlignedCount(unsigned int count)
	///
	/// \brief	Computes the aligned random number count for a given requested random number count.
	/// 		Should be used before calling Generate(). 
	///
	/// \author	Mathias Neumann
	/// \date	25.04.2010 
	///	\see	Generate()
	///
	/// \param	count	Number of randoms to generate. Unaligned in most cases. 
	///
	/// \return	The aligned count. Fits the needs of the mersenne twister implementation. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int GetAlignedCount(unsigned int count);
};

#endif // __MN_CUDA_MT_H__