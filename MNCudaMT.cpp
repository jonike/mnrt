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

#include "MNCudaMT.h"
#include "MNCudaUtil.h"
#include "MNStatContainer.h"
#include "3rdParty/MT/MersenneTwister.h"

// See 3rdParty/MT/MersenneTwister_kernel.cu
extern "C"
bool MersenneTwisterGPUInit(const char *fname);
extern "C"
void MersenneTwisterGPUSeed(unsigned int seed);
extern "C"
void MersenneTwisterGPU(float* d_outRand, int nPerRNG);

MNCudaMT::MNCudaMT(void)
{
	m_bInited = false;
}

MNCudaMT::~MNCudaMT(void)
{
}

MNCudaMT& MNCudaMT::GetInstance()
{
	static MNCudaMT cudaMT;
	return cudaMT;
}

bool MNCudaMT::Init(const char *fname)
{
	bool res = MersenneTwisterGPUInit(fname);
	m_bInited = res;
	return res;
}

bool MNCudaMT::Seed(unsigned int seed)
{
	if(!m_bInited)
		return false;

	MersenneTwisterGPUSeed(seed);

	static StatCounter& ctrSeed = StatCounter::Create("General", "Mersenne Twister (new seed)");
	++ctrSeed;

	return true;
}

cudaError_t MNCudaMT::Generate(float* d_outRand, int count)
{
	if(!m_bInited)
		return cudaErrorUnknown;

	// Check if count is OK.
	if(count != GetAlignedCount(count))
		return cudaErrorInvalidValue;
	int nPerRNG = count / MT_RNG_COUNT;

	MersenneTwisterGPU(d_outRand, nPerRNG);

	static StatCounter& ctrRandom = StatCounter::Create("General", "Mersenne Twister (generated randoms)");
	ctrRandom += count;

	return cudaSuccess;
}

unsigned int MNCudaMT::GetAlignedCount(unsigned int count)
{
	// Taken from SDK 3.0 sample.
	unsigned int numPerRNG = MNCUDA_DIVUP(count, MT_RNG_COUNT);
	unsigned int numAligned = MNCUDA_ALIGN_EX(numPerRNG, 2);

	return numAligned * MT_RNG_COUNT;
}