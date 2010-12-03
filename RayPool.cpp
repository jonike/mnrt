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

#include "RayPool.h"
#include "CameraModel.h"
#include "MNCudaUtil.h"
#include "MNCudaMemPool.h"

using namespace std;

extern "C"
void RTCleanupRayGenKernels();
extern "C"
void KernelRTPrimaryKernel(CameraModel* pCamera, 
						   uint idxSampleX, uint samplesPerPixelX, uint idxSampleY, uint samplesPerPixelY,
						   RayChunk& outChunk);
extern "C"
void KernelRTReflectedKernel(RayChunk& chunkSrc, ShadingPoints& shadingPts,
						     TriangleData& triData, RayChunk& outChunk, uint* d_outIsValid);
extern "C"
void KernelRTTransmittedKernel(RayChunk& chunkSrc, ShadingPoints& shadingPts,
						       TriangleData& triData, RayChunk& outChunk, uint* d_outIsValid);



////////////////////////////////////////////////////////////////////////////////////////////////////
// RayChunk implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

void RayChunk::Initialize(uint _maxRays)
{
	maxRays = _maxRays;
	numRays = 0;
	rekDepth = 0;

	// Allocate device memory.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Request((void**)&d_origins, maxRays*sizeof(float4), "Ray pool", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_dirs, maxRays*sizeof(float4), "Ray pool", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_pixels, maxRays*sizeof(uint), "Ray pool"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_influences, maxRays*sizeof(float4), "Ray pool", 256));
}

void RayChunk::Destroy(void)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_origins));
	mncudaSafeCallNoSync(pool.Release(d_dirs));
	mncudaSafeCallNoSync(pool.Release(d_pixels));
	mncudaSafeCallNoSync(pool.Release(d_influences));
	maxRays = 0;
}

uint RayChunk::Compact(uint* d_isValid)
{
	MNCudaMemory<uint> d_srcAddr(numRays);
	uint countNew = mncudaGenCompactAddresses(d_isValid, numRays, d_srcAddr);

	if(countNew == 0)
	{
		numRays = 0;
		return 0; // Nothing to do.
	}
	if(countNew == numRays)
		return numRays;

	// Now move source data to destination data.
	mncudaCompactInplace(d_origins, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_dirs, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_pixels, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_influences, d_srcAddr, numRays, countNew);

	// Update count.
	numRays = countNew;

	return countNew;
}

void RayChunk::CompactSrcAddr(uint* d_srcAddr, uint countNew)
{
	if(countNew == 0)
	{
		numRays = 0; // Update count!
		return;
	}

	// Move source data to destination data inplace.
	mncudaCompactInplace(d_origins, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_dirs, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_pixels, d_srcAddr, numRays, countNew);
	mncudaCompactInplace(d_influences, d_srcAddr, numRays, countNew);

	// Update count.
	numRays = countNew;
}

void RayChunk::SetFrom(const RayChunk& other, uint* d_srcAddr, uint numSrcAddr)
{
	if(numSrcAddr == 0)
	{
		numRays = 0;
		return;
	}

	mncudaSetFromAddress(d_origins, d_srcAddr, other.d_origins, numSrcAddr);
	mncudaSetFromAddress(d_dirs, d_srcAddr, other.d_dirs, numSrcAddr);
	mncudaSetFromAddress(d_pixels, d_srcAddr, other.d_pixels, numSrcAddr);
	mncudaSetFromAddress(d_influences, d_srcAddr, other.d_influences, numSrcAddr);

	// Update count.
	numRays = numSrcAddr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RayPool implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

RayPool::RayPool(void)
{
	m_pChunkTask = NULL;
	m_maxRaysPerChunk = 256*1024;
	m_nSamplesPerPixelX = 1;
	m_nSamplesPerPixelY = 1;
	m_bInited = false;
}

RayPool::~RayPool(void)
{
	Destroy();
}

void RayPool::Initialize(uint maxRays, uint numSamplesPerPixelX/* = 1*/, uint numSamplesPerPixelY/* = 1*/)
{
	MNAssert(numSamplesPerPixelX > 0 && numSamplesPerPixelY > 0);
	
	m_maxRaysPerChunk = maxRays;
	m_nSamplesPerPixelX = numSamplesPerPixelX;
	m_nSamplesPerPixelY = numSamplesPerPixelY;

	m_bInited = true;
}

void RayPool::Destroy()
{
	if(!m_bInited)
		return;

	RTCleanupRayGenKernels();

	if(isChunkActive())
		MNFatal("Ray pool destroyed while chunk active.");
	m_pChunkTask = NULL;

	for(uint i=0; i<m_RayChunks.size(); i++)
	{
		m_RayChunks[i]->Destroy();
		SAFE_DELETE(m_RayChunks[i]);
	}
	m_RayChunks.clear();
	m_bInited = false;
}

void RayPool::AddPrimaryRays(CameraModel* pCamera, bool useMultiSample)
{
	MNAssert(pCamera);
	uint nScreenW = pCamera->GetScreenWidth();
	uint nScreenH = pCamera->GetScreenHeight();

	// Ensure that all rays for one sample can fit into a single ray chunk.
	// NOTE: This is required for adaptive sample seeding as we would get notable separators in the
	//       final image when subdividing the screen's pixels into sections. The reason is that
	//       the clustering results don't fit together and pixels on both sides of the separator
	//       are calculated using different interpolation points.
	//       Therefore I'm abandoning the choise of putting all samples for a given pixel into the
	//		 same ray chunk to improve cache performance for primary rays for the sake of using
	//		 multisampling in combination with adaptive sample seeding.
	if(nScreenW*nScreenH > m_maxRaysPerChunk)
		MNFatal("Ray chunks too small for given screen resolution (max: %d; need: %d).", 
			m_maxRaysPerChunk, nScreenW*nScreenH);

	uint samplesPerPixelX = 1, samplesPerPixelY = 1;
	if(useMultiSample)
	{
		samplesPerPixelX = m_nSamplesPerPixelX;
		samplesPerPixelY = m_nSamplesPerPixelY;
	}

	// Create as many chunks as required.
	for(uint x=0; x<samplesPerPixelX; x++)
	{
		for(uint y=0; y<samplesPerPixelY; y++)
		{
			RayChunk* pChunk = FindAllocChunk();

			// Generates all rays for this sample index together.
			KernelRTPrimaryKernel(pCamera, x, samplesPerPixelX, y, samplesPerPixelY, *pChunk);
		}
	}
}

bool RayPool::GenerateChildRays(RayChunk* pChunk, ShadingPoints& shadingPts, TriangleData& triData,
								bool bGenReflect, bool bGenTransmit)
{
	if(!isChunkActive() || pChunk != m_pChunkTask)
		MNFatal("Error: Failed to generate child rays. Given parent chunk not active.");

	// Go parallely through both rays and intersections since they have the same order.
	// Do not mix reflected and transmitted rays for better cache performance.
	float fReflect = 0.1f;
	float fTransmit = 0.0f;
	const char maxRekDepth = 1;

	if(pChunk->rekDepth >= maxRekDepth)
		return false;

	// First reflected rays.
	if(bGenReflect)
	{
		MNCudaMemory<uint> d_isValid(pChunk->numRays);

		// Get a chunk for allocation.
		RayChunk* pChunkNew = FindAllocChunk();
		KernelRTReflectedKernel(*pChunk, shadingPts, triData, *pChunkNew, d_isValid);

		// Compact to remove low influence rays.
		pChunkNew->Compact(d_isValid);

		// Ignore small chunks.
		if(pChunkNew->numRays < 32)
		{
			pChunkNew->numRays = 0;
			pChunkNew->rekDepth = 0;
		}
	}

	if(bGenTransmit)
	{
		MNCudaMemory<uint> d_isValid(pChunk->numRays);

		// Get a chunk for allocation.
		RayChunk* pChunkNew = FindAllocChunk();
		KernelRTTransmittedKernel(*pChunk, shadingPts, triData, *pChunkNew, d_isValid);

		// Compact to remove low influence rays.
		pChunkNew->Compact(d_isValid);

		// Ignore small chunks.
		if(pChunkNew->numRays < 32)
		{
			pChunkNew->numRays = 0;
			pChunkNew->rekDepth = 0;
		}
	}

	return true;
}

RayChunk* RayPool::AllocateNewChunk(uint maxRays)
{
	MNAssert(maxRays > 0);

	RayChunk* pChunk = new RayChunk();
	pChunk->Initialize(maxRays);
	
	// Add chunk.
	m_RayChunks.push_back(pChunk);

	MNMessage("Ray pool: new chunk.");

	return pChunk;
}

RayChunk* RayPool::FindAllocChunk()
{
	// Try to use an old chunk first.
	for(size_t i=0; i<m_RayChunks.size(); i++)
	{
		RayChunk* pChunk = m_RayChunks[i];
		if(pChunk != m_pChunkTask && pChunk->numRays == 0)
			return pChunk;
	}

	if(m_RayChunks.size() > 64)
		MNFatal("Ray pool error - too many chunks.");

	// Allocate new chunk...
	return AllocateNewChunk(m_maxRaysPerChunk);
}

void RayPool::Clear()
{
	// This will only work when there is no active task.
	if(isChunkActive())
		MNFatal("Trying to clear ray pool while chunk is active.");

	// Just reset chunk indices.
	for(uint i=0; i<m_RayChunks.size(); i++)
	{
		RayChunk* pChunk = m_RayChunks[i];
		pChunk->numRays = 0;
	}
}

RayChunk* RayPool::GetNextChunk()
{
	// This will only work if there is no other task active.
	if(isChunkActive())
		MNFatal("Trying to get next ray chunk while chunk is active.");

	// Get best chunk.
	RayChunk* pChunk = FindProcessingChunk();
	if(!pChunk)
		return NULL;

	MNAssert(pChunk->numRays > 0);

	// Store the chunk to remember where the rays are in host memory. This
	// is used later when we generate child rays.
	m_pChunkTask = pChunk;

	//MNMessage("Generated Task: %d rays.", pChunk->numRays);

	return m_pChunkTask;
}

RayChunk* RayPool::FindProcessingChunk()
{
	// Use a chunk with the highest possible recursion depth. This should avoid
	// allocating to many chunks since it avoids following too many paths in the
	// recursion tree.

	RayChunk* pChunkBest = NULL;
	int maxDepth = -1;
	for(uint i=0; i<m_RayChunks.size(); i++)
	{
		RayChunk* pChunk = m_RayChunks[i];
		
		int depth = (int)pChunk->rekDepth;
		if(depth > maxDepth && pChunk->numRays > 0)
		{
			maxDepth = depth;
			pChunkBest = pChunk;
		}
	}

	return pChunkBest;
}

void RayPool::FinalizeChunk(RayChunk* pChunk)
{
	if(!isChunkActive() || pChunk != m_pChunkTask)
		MNFatal("Failed to finalize chunk. Chunk unknown.");

	// Chunk done. Reset indices.
	m_pChunkTask->rekDepth = 0;
	m_pChunkTask->numRays = 0;

	// Forget about that task.
	m_pChunkTask = NULL;
}