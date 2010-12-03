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

#include "MNCudaMemPool.h"
#include <cutil_inline.h>
#include "MNStatContainer.h"
#include <string>
#include <ctime>


//#define MN_MPOOL_MEASURETIME

/// Time in seconds until a chunk is obsolete.
#define MN_MPOOL_TIME2OBSOLETE	5

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// MNCudaMemPool::MemoryChunk implementation
////////////////////////////////////////////////////////////////////////////////
MNCudaMemPool::MemoryChunk::MemoryChunk(size_t _size, bool pinnedHost, cudaError_t& outError)
{
	sizeChunk = _size;
	isHostPinned = pinnedHost;
	isRemoveable = true;

	if(isHostPinned)
	{
		// Allocate pinned (non-pageable) host memory.
		outError = cudaMallocHost(&d_buffer, sizeChunk);
		if(outError != cudaSuccess)
			return;
	}
	else
	{
		// Allocate device memory.
		outError = cudaMalloc(&d_buffer, sizeChunk);
		if(outError != cudaSuccess)
			return;

		MNMessage("MNCudaMemPool - New device chunk: %.3lf MBytes.", (double)sizeChunk / (1024.0*1024.0));
	}

	// At first everything is free.
	lstAssigned.clear();
}

MNCudaMemPool::MemoryChunk::~MemoryChunk()
{
	if(isHostPinned)
		cudaFreeHost(d_buffer);
	else
	{
		MNMessage("MNCudaMemPool - Device chunk released: %.3lf MBytes.", (double)sizeChunk / (1024.0*1024.0));
		cudaFree(d_buffer);
	}
}

void* MNCudaMemPool::MemoryChunk::Request(size_t size_bytes, size_t alignment, const std::string& strCategory)
{
	MNAssert(size_bytes > 0);

	// Find first fit.
	list<AssignedSegment>::iterator iter;
	size_t offset, freeTo;
	AssignedSegment* pPrev = NULL;
	for(iter=lstAssigned.begin(); iter!=lstAssigned.end(); ++iter) 
	{
		AssignedSegment* pCur = (AssignedSegment*)&(*iter);

		// Check space *before* current segment.
		offset = 0;
		if(pPrev != NULL)
			offset = MNCUDA_ALIGN_BYTES(pPrev->offset + pPrev->size_bytes, alignment);
		freeTo = pCur->offset;

		if(freeTo > offset && freeTo - offset >= size_bytes)
		{
			// Found fit. To keep order, place fit right before assigned segment.
			void* buf = ((unsigned char*)d_buffer + offset);
			lstAssigned.insert(iter, AssignedSegment(offset, size_bytes, buf, strCategory));
			tLastUse = time(NULL);
			//MNMessage("MNCudaMemPool - Segment assigned: offset %d, size %d.\n", offset, size_bytes);
			return buf;
		}

		pPrev = pCur;
	}

	// Now check space after the last segment or from the beginning (if no segments).
	offset = 0;
	if(lstAssigned.size())
	{
		AssignedSegment* pLast = &lstAssigned.back();
		offset = MNCUDA_ALIGN_BYTES(pLast->offset + pLast->size_bytes, alignment);
	}
	freeTo = sizeChunk;
	if(freeTo > offset && freeTo - offset >= size_bytes)
	{
		// Found fit. Just attach at end.
		void* buf = ((unsigned char*)d_buffer + offset);
		lstAssigned.push_back(AssignedSegment(offset, size_bytes, buf, strCategory));
		tLastUse = time(NULL);
		//MNMessage("MNCudaMemPool - Segment assigned: offset %d, size %d.\n", offset, size_bytes);
		return buf;
	}


	// Nothing found.
	return NULL;
}

size_t MNCudaMemPool::MemoryChunk::Release(void* d_buffer)
{
	list<AssignedSegment>::iterator iter;
	for(iter=lstAssigned.begin(); iter!=lstAssigned.end(); ++iter) 
	{
		AssignedSegment* pCur = (AssignedSegment*)&(*iter);

		if(pCur->d_buffer == d_buffer)
		{
			StatCounter& ctrRelCat = StatCounter::Create("Memory", "Cat: " + pCur->strCategory + " (Release)");
			ctrRelCat += pCur->size_bytes;

			//MNMessage("MNCudaMemPool - Segment released: offset %d, size %d.\n", pCur->offset, pCur->size_bytes);
			size_t bytesFreed = pCur->size_bytes;
			lstAssigned.erase(iter);
			tLastUse = time(NULL);
			return bytesFreed;
		}
	}

	return 0;
}

size_t MNCudaMemPool::MemoryChunk::GetAssignedSize() const
{
	size_t size = 0;

	list<AssignedSegment>::const_iterator iter;
	for(iter=lstAssigned.begin(); iter!=lstAssigned.end(); ++iter) 
	{
		AssignedSegment* pCur = (AssignedSegment*)&(*iter);
		size += pCur->size_bytes;
	}

	return size;
}

bool MNCudaMemPool::MemoryChunk::IsObsolete(time_t tCurrent) const
{
	return isRemoveable && lstAssigned.size() == 0 && (tCurrent - tLastUse >= MN_MPOOL_TIME2OBSOLETE);
}

bool MNCudaMemPool::MemoryChunk::Test(FILE* stream) const
{
	size_t lastEnd = 0;
	list<AssignedSegment>::const_iterator iter;
	for(iter=lstAssigned.begin(); iter!=lstAssigned.end(); ++iter) 
	{
		AssignedSegment* pCur = (AssignedSegment*)&(*iter);

		size_t thisStart = pCur->offset;

		// Test if segments are disjoint.
		if(thisStart < lastEnd)
			return false;

		// Test if offset is aligned to 64 bytes (at least).
		if(MNCUDA_ALIGN_EX(pCur->offset, 64) != pCur->offset)
			fprintf(stream, "              - Segment not aligned to 64 bytes.\n");

		lastEnd = pCur->offset + pCur->size_bytes;
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
// MNCudaMemPool implementation
////////////////////////////////////////////////////////////////////////////////
MNCudaMemPool::MNCudaMemPool(void)
{
	m_bInited = false;
	m_nBytesAssigned = 0;
	m_tLastObsoleteCheck = time(NULL);

	m_pHostChunk = NULL;
}

MNCudaMemPool::~MNCudaMemPool(void)
{
	Free();
}

MNCudaMemPool& MNCudaMemPool::GetInstance()
{
	static MNCudaMemPool pool;
	return pool;
}

cudaError_t MNCudaMemPool::Initialize(size_t sizeInitial_bytes, size_t sizePinnedHost_bytes)
{
	if(m_bInited)
		return cudaErrorInitializationError;
	cudaError_t err = cudaSuccess;

	// Get texture alignment requirement (bytes).
	cudaDeviceProp props;
	int curDevice;
	err = cudaGetDevice(&curDevice);
	if(err != cudaSuccess)
		return err;
	err = cudaGetDeviceProperties(&props, curDevice);
	m_texAlignment = props.textureAlignment;

	// Store initial chunk size, but do not allocate yet.
	m_sizeInitial_bytes = sizeInitial_bytes;

	// Allocate pinned host memory chunk if required.
	m_pHostChunk = NULL;
	if(sizePinnedHost_bytes > 0)
	{
		m_pHostChunk = new MemoryChunk(sizePinnedHost_bytes, true, err);
		m_pHostChunk->SetRemoveable(false);
		if(err != cudaSuccess)
		{
			SAFE_DELETE(m_pHostChunk);
			return err;
		}
	}

	m_bInited = true;
	return err;
}

cudaError_t MNCudaMemPool::Request(void** d_buffer, size_t size_bytes, 
								   const std::string& strCategory/* = "General"*/, size_t alignment /*= 64*/)
{
	if(!m_bInited)
		return cudaErrorNotReady;
	if(size_bytes == 0)
		return cudaErrorInvalidValue;
	cudaError_t err = cudaSuccess;

	static StatCounter& ctrReq = StatCounter::Create("Memory", "Requested bytes (total)");
	static StatCounter& ctrReqMax = StatCounter::Create("Memory", "Requested bytes (maximum)");
	static StatCounter& ctrReqSingleMax = StatCounter::Create("Memory", "Single request (maximum)");
	StatCounter& ctrReqCat = StatCounter::Create("Memory", "Cat: " + strCategory + " (Request)");
	ctrReqCat += size_bytes;
	ctrReq += size_bytes;
	ctrReqSingleMax.Max(size_bytes);

#ifdef MN_MPOOL_MEASURETIME
	static StatTimer& timerRequest = StatTimer::Create("Memory", "Request processing time", true);
	mncudaSafeCallNoSync(timerRequest.Start());
#endif

	while(true)
	{
		// Find a free range. For now, just use the first fit.
		list<MemoryChunk*>::iterator iter;
		bool done = false;
		for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
		{
			MemoryChunk* pChunk = *iter;

			// Try to use this chunk.
			void* buf = pChunk->Request(size_bytes, alignment, strCategory);
			if(buf != NULL)
			{
				// Found space.
				*d_buffer = buf;
				m_nBytesAssigned += size_bytes;
				ctrReqMax.Max(m_nBytesAssigned);
				err = cudaSuccess;
				done = true;
				break;
			}
		}
		
		if(done)
			break;

		// When getting here, there is no more space left in our chunks.
		// Therefore allocate a new device memory chunk.

		// Get the amout of free memory first. Ensure we still have enough memory left.
		// Not available in device emu mode!
		size_t free, total;
		CUresult res = cuMemGetInfo(&free, &total);
		if(res != CUDA_SUCCESS)
		{
			err = cudaErrorUnknown;
			break;
		}
		if(free < size_bytes)
		{
			err = cudaErrorMemoryValueTooLarge;
			break;
		}
		// Avoid allocating too much memory by reserving 100 MB for other use.
		const size_t reserved = 100*1024*1024;
		size_t freeForUs = 0;
		if(free > reserved)
			freeForUs = free - reserved;
		
		// Use a maximum chunk size. Doubling the chunks does not lead to good results as it
		// would fill the whole memory in a few steps. This chunk size can only be enlarged
		// if a given request needs more memory.
		size_t sizeNew;
		if(m_DevChunks.size() == 0)
			sizeNew = m_sizeInitial_bytes;
		else
		{
			MemoryChunk* pLast = m_DevChunks.back();
			sizeNew = std::min(freeForUs, std::max(std::min(pLast->sizeChunk*2, (size_t)100*1024*1024), size_bytes));
		}

		if(freeForUs == 0)
		{
			// No more memory available for us.
			MNFatal("MNCudaMemPool - OUT OF MEMORY (left: %.2f MByte, reserved: %.2f MByte).",
				free / (1024.f*1024.f), reserved / (1024.f*1024.f));
		}

		err = AllocChunk(sizeNew);
		if(err != cudaSuccess)
			break;

		// Use the new chunk.
		MemoryChunk* pNew = m_DevChunks.back();
		void* buf = pNew->Request(size_bytes, alignment, strCategory);
		if(buf != NULL)
		{
			// Found space.
			*d_buffer = buf;
			m_nBytesAssigned += size_bytes;
			ctrReqMax.Max(m_nBytesAssigned);
			err = cudaSuccess;
			break;
		}

		err = cudaErrorMemoryValueTooLarge;
		break;
	}

#ifdef MN_MPOOL_MEASURETIME
	mncudaSafeCallNoSync(timerRequest.Stop());
#endif
	return err;
}

cudaError_t MNCudaMemPool::RequestTexture(void** d_buffer, size_t size_bytes, 
										  const std::string& strCategory/* = "General"*/)
{
	return Request(d_buffer, size_bytes, strCategory, m_texAlignment);
}

cudaError_t MNCudaMemPool::Release(void* d_buffer)
{
	if(!m_bInited)
	{
		// Nothing to do. Pool destroyed before. All memory free'd.
		return cudaSuccess;
	}
#ifdef MN_MPOOL_MEASURETIME
	static StatTimer& timerRelease = StatTimer::Create("Memory", "Release processing time", true);
	mncudaSafeCallNoSync(timerRelease.Start());
#endif
	cudaError_t err = cudaErrorInvalidDevicePointer;

	// Find the associated segment.
	list<MemoryChunk*>::iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
	{
		MemoryChunk* pChunk = *iter;
		size_t freed = pChunk->Release(d_buffer);
		if(freed)
		{
			m_nBytesAssigned -= freed;
			err = cudaSuccess;
			break;
		}
	}

	// Kill obsolete (unused) memory chunks.
	KillObsoleteChunks();

#ifdef MN_MPOOL_MEASURETIME
	mncudaSafeCallNoSync(timerRelease.Stop());
#endif
	return err;
}

cudaError_t MNCudaMemPool::RequestHost(void** h_buffer, size_t size_bytes)
{
	if(!m_bInited)
		return cudaErrorNotReady;
	if(size_bytes == 0)
		return cudaErrorInvalidValue;
	if(!m_pHostChunk)
		return cudaErrorMemoryAllocation;
	cudaError_t err = cudaSuccess;

	StatCounter& ctrReqCat = StatCounter::Create("Memory", "Cat: Pinned Host (Request)");
	ctrReqCat += size_bytes;

#ifdef MN_MPOOL_MEASURETIME
	static StatTimer& timerRequestPin = StatTimer::Create("Memory", "Request processing time (pinned)", true);
	mncudaSafeCallNoSync(timerRequestPin.Start());
#endif

	// Find a free range.
	void* buf = m_pHostChunk->Request(size_bytes, 64, "Pinned Host");
	if(buf == NULL)
		err = cudaErrorMemoryValueTooLarge;
	else
	{
		// Found space.
		*h_buffer = buf;
		err = cudaSuccess;
	}	

#ifdef MN_MPOOL_MEASURETIME
	mncudaSafeCallNoSync(timerRequestPin.Stop());
#endif
	return err;
}

cudaError_t MNCudaMemPool::ReleaseHost(void* h_buffer)
{
	if(!m_bInited)
	{
		// Nothing to do. Pool destroyed before. All memory free'd.
		return cudaSuccess;
	}

	cudaError_t err = cudaErrorInvalidDevicePointer;
	if(!m_pHostChunk)
		return err;

#ifdef MN_MPOOL_MEASURETIME
	static StatTimer& timerReleasePin = StatTimer::Create("Memory", "Release processing time (pinned)", true);
	mncudaSafeCallNoSync(timerReleasePin.Start());
#endif

	size_t freed = m_pHostChunk->Release(h_buffer);
	if(freed)
		err = cudaSuccess;

#ifdef MN_MPOOL_MEASURETIME
	mncudaSafeCallNoSync(timerReleasePin.Stop());
#endif
	return err;
}

void MNCudaMemPool::UpdatePool()
{
	KillObsoleteChunks();
}

void MNCudaMemPool::KillObsoleteChunks()
{
	time_t tCurrent = time(NULL);
	if(tCurrent <= m_tLastObsoleteCheck)
		return;

	list<MemoryChunk*>::iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ) 
	{
		MemoryChunk* pChunk = *iter;
		if(pChunk->IsObsolete(tCurrent))
		{
			// Kill chunk. Post-increment to get the value to kill. This
			// also ensures iter is still valid.
			list<MemoryChunk*>::iterator iter2erase = iter++;
			m_DevChunks.erase(iter2erase);
			SAFE_DELETE(pChunk);
		}
		else
			++iter;
	}

	m_tLastObsoleteCheck = tCurrent;
}

void MNCudaMemPool::PrintState(FILE *f) const
{
	double dTotalMB = (double)GetAllocatedSize() / (1024.0*1024.0);
	double dAssignMB = (double)GetAssignedSize() / (1024.0*1024.0);

	fprintf(f, "MNCudaMemPool - Statistics\n");
	fprintf(f, "              - Device chunks:     %d (%.3lf MBytes total).\n", 
		GetDeviceChunkCount(), dTotalMB);
	fprintf(f, "              - Assigned segments: %d (%.3lf MBytes total).\n",
		GetAssignedSegmentCount(), dAssignMB);
	if(dTotalMB <= 0.f)
		fprintf(f, "              - Usage rate:        N/A.\n");
	else
		fprintf(f, "              - Usage rate:        %.2lf%%.\n", 100.0 * dAssignMB / dTotalMB);

	// Print pinned host memory information.
	double dPinnedHost = 0., dPinnedAssigned = 0.;
	if(m_pHostChunk)
	{
		dPinnedHost = (double)m_pHostChunk->sizeChunk / (1024.0*1024.0);
		dPinnedAssigned = (double)m_pHostChunk->GetAssignedSize() / (1024.0*1024.0);
	}
	fprintf(f, "              - Pinned memory:     %.3lf MBytes total.\n", dPinnedHost);
	fprintf(f, "              - Assigned pinned:   %.3lf MBytes total.\n", dPinnedAssigned);

	// Also print global memory data as there are other things using GPU memory.
	size_t free, total;
	cuMemGetInfo(&free, &total);
	fprintf(f, "GPU memory    - Used:              %.3lf MBytes.\n", (total-free) / (1024.0*1024.0));
	fprintf(f, "GPU memory    - Free:              %.3lf MBytes.\n", free / (1024.0*1024.0));
}

void MNCudaMemPool::TestPool(FILE* stream/* = stdout*/) const
{
	fprintf(stream, "MNCudaMemPool - Testing...\n");	

	list<MemoryChunk*>::const_iterator iter;
	uint idxChunk = 0;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter, idxChunk++)
	{
		if(!(*iter)->Test(stream))
			fprintf(stream, "              - Errors in device chunk %d.\n", idxChunk);
	}

	if(m_pHostChunk)
	{
		if(!m_pHostChunk->Test(stream))
			fprintf(stream, "              - Errors in pinned host chunk.\n");
	}

	fprintf(stream, "MNCudaMemPool - Testing done.\n");	
}

void MNCudaMemPool::Free()
{
	if(!m_bInited)
		return;

	list<MemoryChunk*>::iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
		SAFE_DELETE(*iter);
	m_DevChunks.clear();

	SAFE_DELETE(m_pHostChunk);

	m_bInited = false;
}

cudaError_t MNCudaMemPool::AllocChunk(size_t size_bytes)
{
	MNAssert(size_bytes > 0);
	if(size_bytes == 0)
		MNFatal("MNCudaMemPool - Allocating empty chunk...");
	cudaError_t err = cudaSuccess;

	MemoryChunk* pChunk = new MemoryChunk(size_bytes, false, err);
	pChunk->SetRemoveable(m_DevChunks.size() > 0); // First chunk is non-removeable.
	m_DevChunks.push_back(pChunk);

	return cudaSuccess;
}

size_t MNCudaMemPool::GetAllocatedSize() const
{
	if(!m_bInited)
		return 0;
	size_t size = 0;

	list<MemoryChunk*>::const_iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
		size += (*iter)->sizeChunk;

	return size;
}

size_t MNCudaMemPool::GetAssignedSegmentCount() const
{
	if(!m_bInited)
		return 0;
	size_t count = 0;

	list<MemoryChunk*>::const_iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
		count += (*iter)->lstAssigned.size();

	return count;
}

size_t MNCudaMemPool::GetAssignedSize() const
{
	
	if(!m_bInited)
		return 0;
	size_t size = 0;

	list<MemoryChunk*>::const_iterator iter;
	for(iter=m_DevChunks.begin(); iter!=m_DevChunks.end(); ++iter) 
		size += (*iter)->GetAssignedSize();

	if(m_nBytesAssigned != size)
		MNFatal("MNCudaMemPool - Assigned size error.");

	return size;
}