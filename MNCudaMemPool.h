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
/// \file	MNRT\MNCudaMemPool.h
///
/// \brief	Declares the MNCudaMemPool and MNCudaMemory classes. 
/// \author	Mathias Neumann
/// \date	19.03.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_CUDA_MEMPOOL_H__
#define __MN_CUDA_MEMPOOL_H__

#pragma once

#include <list>
#include "MNCudaUtil.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNCudaMemPool
///
/// \brief	Manages device memory by preallocating a large amount of device memory and handling
///			out chunks to requesters. 
///
///			This avoids multiple calls to \c cudaMalloc and therefore reduces CUDA API overhead. Was
///			suggested by \ref lit_wang "[Wang et al. 2009]" and \ref lit_zhou "[Zhou et al. 2008]".
///
///			Class is designed as singleton and might need optimizations for when used from
///			multiple CPU-threads.
///
/// \author	Mathias Neumann
/// \date	19.03.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNCudaMemPool
{
private:
	/// Keeps track of assigned device memory segments.
	class AssignedSegment
	{
	public:
		AssignedSegment(size_t _offset, size_t _size, void* _buffer, const std::string& _strCategory)
		{
			offset = _offset;
			size_bytes = _size;
			d_buffer = _buffer;
			strCategory = _strCategory;
		}

		// Byte offset. Relative to owning chunk base offset.
		// Has to be aligned.
		size_t offset;
		// Number of bytes.
		size_t size_bytes;
		// Pointer to the device memory segment.
		void* d_buffer;
		// Memory category.
		std::string strCategory;
	};

	/// Keeps track of allocated device or pinned host memory.
	class MemoryChunk
	{
	public:
		MemoryChunk(size_t _size, bool pinnedHost, cudaError_t& outError);
		~MemoryChunk();

		// If true, the chunk can be killed when not used for some time. Default: true.
		bool isRemoveable;
		// If true, this is a pinned host memory chunk. Else it's a device memory chunk.
		bool isHostPinned;
		// Size in bytes.
		size_t sizeChunk;
		// Memory of *sizeChunk* bytes.
		void* d_buffer;
		// Assigned segments. Ordered by offsets.
		std::list<AssignedSegment> lstAssigned;
		// Time of last use (time(), seconds). Used to detect obsolete chunks.
		time_t tLastUse;

	public:
		// Returns a pointer to the requested memory space within this chunk, if any.
		// Else NULL is returned. In the first case the free range list is updated.
		void* Request(size_t size_bytes, size_t alignment, const std::string& strCategory);
		// Releases the given buffer if it is assigned within this chunk. Number of
		// free'd bytes is returned, else 0.
		size_t Release(void* d_buffer);
		// Returns the assigned size within this segment.
		size_t GetAssignedSize() const;
		// Sets whether the chunk is removeable or not.
		void SetRemoveable(bool b) { isRemoveable = b; }
		// Returns true when this chunk is obsolete and can be destroyed.
		bool IsObsolete(time_t tCurrent) const;
		// Tests this chunk for errors, e.g. non disjoint segments.
		bool Test(FILE* stream) const;
	};

	// Singleton. Hide constructors.
private:
	MNCudaMemPool(void);
	MNCudaMemPool(const MNCudaMemPool& other);

public:
	~MNCudaMemPool(void);

private:
	/// If true, we are initialized.
	bool m_bInited;
	/// Number of bytes currently assigned.
	size_t m_nBytesAssigned;
	/// Texture alignment requirement for current device.
	size_t m_texAlignment;
	/// Last obsolete check time.
	time_t m_tLastObsoleteCheck;
	/// Initial segment size in bytes.
	size_t m_sizeInitial_bytes;

	/// The only pinned host memory chunk.
	MemoryChunk* m_pHostChunk;
	/// Device memory chunks.
	std::list<MemoryChunk*> m_DevChunks;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNCudaMemPool& GetInstance()
	///
	/// \brief	Returns the only memory pool instance.
	///
	/// \warning Not thread-safe!
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	///
	/// \return	The instance. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNCudaMemPool& GetInstance();

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Initialize(size_t sizeInitial_bytes, size_t sizePinnedHost_bytes)
	///
	/// \brief	Initializes the memory pool. 
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	///
	/// \param	sizeInitial_bytes		The size of the initial device chunk in bytes. 
	/// \param	sizePinnedHost_bytes	The size of the only pinned host chunk in bytes. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Initialize(size_t sizeInitial_bytes, size_t sizePinnedHost_bytes);


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Request(void** d_buffer, size_t size_bytes,
	/// 	const std::string& strCategory = "General", size_t alignment = 64)
	///
	/// \brief	Requests a device buffer of a given size.
	/// 		
	/// 		You can specify an alignment for the memory segment. This allows using the segment
	/// 		for coalesced access or for linear memory to texture mappings. For example, coalesced
	/// 		access to 64 bit words on 1.1 computing capability devices require an alignment of
	/// 		128 bytes (16 * 8). 
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	/// \see	RequestTexture(), RequestHost()
	///
	/// \param [out]	d_buffer	The allocated device memory buffer. 
	/// \param	size_bytes			The requested size in bytes. 
	/// \param	strCategory			Category of the request. Used for bookkeeping only.
	/// \param	alignment			The alignment in bytes. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Request(void** d_buffer, size_t size_bytes, 
		const std::string& strCategory = "General", size_t alignment = 64);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t RequestTexture(void** d_buffer, size_t size_bytes,
	/// 	const std::string& strCategory = "General")
	///
	/// \brief	Requests a buffer of a given size to use as linear memory to map to textures. 
	///
	///			It is aligned according to the CUDA device properties to avoid using offsets
	/// 		returned by \c cudaBindTexture(). This method equals the Request() method with
	///			a special alignment parameter.
	///
	/// \author	Mathias Neumann
	/// \date	26.03.2010
	/// \see	Request(), RequestHost()
	///
	/// \param [out]	d_buffer	The allocated buffer. 
	/// \param	size_bytes			The size in bytes. 
	/// \param	strCategory			Category the string belongs to. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t RequestTexture(void** d_buffer, size_t size_bytes, 
		const std::string& strCategory = "General");

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Release(void* d_buffer)
	///
	/// \brief	Releases the given device buffer.
	/// 		
	/// 		This will only work if the buffer has been allocated in this pool. After this call
	/// 		the buffer is no more valid. 
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010 
	///	\see	ReleaseHost()
	///
	/// \param [in]		d_buffer	The buffer to release. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Release(void* d_buffer);


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t RequestHost(void** h_buffer, size_t size_bytes)
	///
	/// \brief	Requests pinned host memory of given size.
	/// 		
	/// 		Note that pinned host memory is limited as we currently only provide a fixed chunk of
	/// 		chosen size. This was done due to the fact that lots of pinned host memory can reduce
	/// 		system performance significantly. Check the CUDA SDK programming guide for more
	///			information.
	///
	/// \author	Mathias Neumann
	/// \date	01.08.2010 
	///	\see	Request(), RequestTexture()
	///
	/// \param [out]	h_buffer	Pinned host memory of \a size_bytes bytes. 
	/// \param	size_bytes			The size in bytes. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t RequestHost(void** h_buffer, size_t size_bytes);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t ReleaseHost(void* h_buffer)
	///
	/// \brief	Releases the given assigned buffer of pinned host memory.
	/// 		
	/// 		This will only work if the buffer has been allocated in this pool. After this call
	/// 		the buffer is no more valid. 
	///
	/// \author	Mathias Neumann
	/// \date	01.08.2010 
	///	\see	Release()
	///
	/// \param [in]		h_buffer	The pinned host buffer to release. 
	///
	/// \return	\c cudaSuccess, else some error value. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t ReleaseHost(void* h_buffer);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void PrintState(FILE* stream = stdout) const
	///
	/// \brief	Prints the state of the memory pool to the given file. 
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \param [in]		stream	The file stream to print to. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void PrintState(FILE* stream = stdout) const;


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void UpdatePool()
	///
	/// \brief	Updates the pool by removing any chunks that are unused for a long time.
	///
	///			Call this on a regular base if you want to avoid stealing to much GPU memory for the
	///			lifetime of your application. In most cases, there are peaks of pool usage where
	///			big new chunks of device memory are added. After that, these chunks are completely
	///			unused. This method tries to eliminate those chunks after some time has passed.
	///
	/// \author	Mathias Neumann
	/// \date	05.10.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void UpdatePool();


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void TestPool(FILE* stream = stdout) const
	///
	/// \brief	Tests the memory pool by checking all memory chunks managed.
	///
	///			Checks memory chunks for errors, e.g. non-disjoint assigned segments.
	///
	/// \author	Mathias Neumann
	/// \date	05.10.2010
	///
	/// \param [in]	stream	File for result output.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void TestPool(FILE* stream = stdout) const;

// Accessors
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	size_t GetTextureAlignment() const
	///
	/// \brief	Gets the texture alignment for the current device. 
	///
	///			Linear device memory that is mapped to texture memory has to be aligned using this
	///			alignment. Else offsets have to be used when binding the texture using the CUDA API.
	///
	/// \author	Mathias Neumann
	/// \date	26.03.2010
	///
	/// \return	The texture alignment in bytes. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	size_t GetTextureAlignment() const { return m_texAlignment; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	size_t GetDeviceChunkCount() const
	///
	/// \brief	Gets the device chunk count.
	///
	///			This is the number of device chunks currently managed by this pool.
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \return	The device chunk count. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	size_t GetDeviceChunkCount() const { return m_DevChunks.size(); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	size_t GetAllocatedSize() const
	///
	/// \brief	Gets the size of the allocated device memory in bytes. 
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \return	The allocated device memory size in bytes. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	size_t GetAllocatedSize() const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	size_t GetAssignedSegmentCount() const
	///
	/// \brief	Gets the assigned device memory segment count.
	/// 		
	/// 		This is the number of assigned segments within the device memory chunks. Each
	/// 		Request() or RequestTexture() creates a new assigned segment. 
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \return	The assigned segment count. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	size_t GetAssignedSegmentCount() const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	size_t GetAssignedSize() const
	///
	/// \brief	Gets the assigned device memory size in bytes. 
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \return	The assigned memory size in bytes. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	size_t GetAssignedSize() const;

private:
	// Frees all memory. Called on destruction.
	void Free();
	// Allocates a new chunk of device memory. Used for pool resizing.
	cudaError_t AllocChunk(size_t size_bytes);
	// Kills obsolete chunks.
	void KillObsoleteChunks();
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNCudaMemory
///
/// \brief	Device memory wrapper class that uses the memory pool to request device memory.
///
///			Releasing is performed within destructor, so this is an easy way to use the memory pool.
///			Automatic conversion operator avoids the use of GetPtr(). It however might create
///			compiler errors when used in conjunction with template function parameters. Then an
///			explicit cast to (T*) might be required. Use it the following way:
///
///	\code
///	{
///		MNCudaMemory<uint> d_temp(1000); // Creates temporary device memory.
///		...
///		// Use d_temp, if required with an explicit cast (uint*)d_temp.
///		...
///	} // Destructor releases memory automatically.
///	\endcode
///
/// \author	Mathias Neumann
/// \date	30.03.2010
///
/// \tparam	T	Element type of the memory requested. Allows requesting arrays of integers 
///				or floats.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class MNCudaMemory
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNCudaMemory(size_t numElements, const std::string& strCategory = "Temporary",
	/// 	size_t alignment = 64)
	///
	/// \brief	Constructor. Requests memory from MNCudaMemPool.
	///
	/// \author	Mathias Neumann
	/// \date	30.03.2010
	/// \see	MNCudaMemPool::Request()
	///
	/// \param	numElements	Number of elements. 
	/// \param	strCategory	Category of memory. For bookkeeping only.
	/// \param	alignment	The alignment in bytes. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNCudaMemory(size_t numElements, const std::string& strCategory = "Temporary", size_t alignment = 64)
	{
		MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
		mncudaSafeCallNoSync(pool.Request((void**)&d_buffer, numElements*sizeof(T), strCategory, alignment));
		numElems = numElements;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	~MNCudaMemory()
	///
	/// \brief	Destructor. Releases requested device memory.
	///
	/// \author	Mathias Neumann
	/// \date	30.03.2010
	/// \see	MNCudaMemPool::Release()
	////////////////////////////////////////////////////////////////////////////////////////////////////
	~MNCudaMemory()
	{
		MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
		mncudaSafeCallNoSync(pool.Release(d_buffer));
	}

private:
	/// Number of elements.
	size_t numElems;
	/// The device memory.
	T* d_buffer;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	T* GetPtr() const
	///
	/// \brief	To retrieve the device memory pointer. 
	///
	/// \author	Mathias Neumann
	/// \date	30.03.2010
	///
	/// \return	Device memory pointer of type \a T. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	T* GetPtr() const { return d_buffer; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	operator T* ()
	///
	/// \brief	Automatic conversion operator. Avoids GetPtr().
	///
	///			Might not suffice in some cases, e.g. when this object is used as parameter for a
	///			template function. In this case an explicit cast with (T*) would be required.
	///
	/// \author	Mathias Neumann
	/// \date	30.03.2010
	///
	/// \return	Device memory pointer of type \a T. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	operator T* () { return d_buffer; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	T Read(size_t idx)
	///
	/// \brief	Reads an entry of the device memory. 
	///
	///			Use sparely since it uses a \c cudaMemcpy to copy from device to host. This can be
	///			quite slow.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param	idx	Zero-based index of the entry.
	///
	/// \return	The entry. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	T Read(size_t idx)
	{
		if(idx >= numElems)
			MNFatal("MNCudaMemory - Illegal element index.");

		T res;
		mncudaSafeCallNoSync(cudaMemcpy(&res, d_buffer + idx, sizeof(T), cudaMemcpyDeviceToHost));

		return res;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void InitConstant(const T& constant)
	///
	/// \brief	Initialises the memory with the given constant.
	///
	///			\c cudaMemset might be faster than this method. But it does not allow to initialize the
	///			\em elements in most cases. Only trivial cases, e.g. with a constant of zero, might
	///			work.
	///
	/// \author	Mathias Neumann
	/// \date	08.07.2010
	/// \see	::mncudaInitConstant()
	///
	/// \param	constant	The constant (type of element) to initialize each element with.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void InitConstant(const T& constant)
	{
		mncudaInitConstant((T*)d_buffer, numElems, constant);
	}
};

#endif // __MN_CUDA_MEMPOOL_H__