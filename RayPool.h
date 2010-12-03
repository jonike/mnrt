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
/// \file	MNRT\RayPool.h
///
/// \brief	Declares the RayPool and RayChunk classes. 
/// \author	Mathias Neumann
/// \date	03.02.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_RAYPOOL_H__
#define __MN_RAYPOOL_H__

#pragma once

#include "KernelDefs.h"
#include <vector>

class CameraModel;
class Scene;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	RayChunk
///
/// \brief	Stores rays for GPU processing.
/// 		
/// 		Rays are generated on the GPU, so the buffers are SoA- style to increase performance
/// 		and are located in device memory. A RayChunk object is normally seen in conjunction
/// 		with a ShadingPoints object, where both objects have the same amount of elements
/// 		(rays and shading points respectively). The i-th ray of the RayChunk corresponds with
/// 		the i-th shading point in that case. This allows reducing the memory requirement for
/// 		ShadingPoints by avoiding to store too much redundant information in that structure. 
///
/// \note	A chunk should only store rays of the same recursion depth.
///
/// \author	Mathias Neumann
/// \date	13.02.2010
/// \see	RayPool
////////////////////////////////////////////////////////////////////////////////////////////////////
class RayChunk
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(uint _maxRays)
	///
	/// \brief	Initializes device memory. 
	///
	/// \author	Mathias Neumann
	/// \date	13.02.2010
	///
	/// \param	_maxRays	The maximum number of rays to store. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(uint _maxRays);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy(void)
	///
	/// \brief	Destroys chunk by releasing device memory. 
	///
	/// \author	Mathias Neumann
	/// \date	13.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy(void);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint Compact(uint* d_isValid)
	///
	/// \brief	Compacts this ray chunk. 
	///
	///			The compaction is controlled using the binary 0/1 array \a d_isValid.
	///
	/// \author	Mathias Neumann
	/// \date	21.10.2010
	///
	/// \param [in]	d_isValid	Binary 0/1 array. The i-th ray is retained when \a d_isValid[i] = 1.
	///							Else, when \a d_isValid[i] = 0, the i-th ray is removed.
	///
	/// \return	Returns new ray count.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint Compact(uint* d_isValid);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void CompactSrcAddr(uint* d_srcAddr, uint countNew)
	///
	/// \brief	Compacts this ray chunk using the given source address array.
	/// 		
	/// 		This operation assumes that the source addresses were generated before, e.g. using ::
	/// 		mncudaGenCompactAddresses(). The latter also returns the required new number of rays.
	/// 		Basically, this was done to allow compacting multiple structures using the same
	/// 		source addresses. 
	///
	/// \author	Mathias Neumann
	/// \date	April 2010 
	/// \see	::mncudaGenCompactAddresses(), PhotonData::Compact(), ShadingPoints::Compact()
	///
	/// \param [in]		d_srcAddr	The source addresses (device memory). \a d_srcAddr[i] defines at
	/// 							which original index the new value for the i-th ray can be found. 
	/// \param	countNew			The new number of rays. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompactSrcAddr(uint* d_srcAddr, uint countNew);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetFrom(const RayChunk& other, uint* d_srcAddr, uint numSrcAddr)
	///
	/// \brief	Sets this ray chunk based on another ray chunk.
	/// 		
	/// 		The process of copying the \a other ray chunk into this object is guided by the
	/// 		source addresses \a d_srcAddr in the following way:
	/// 		
	/// 		\code d_origins[i] = other.d_origins[d_srcAddr[i]]; // Same for other members \endcode
	///
	/// \author	Mathias Neumann
	/// \date	July 2010 
	///	\see	::mncudaSetFromAddress(), CompactSrcAddr()
	///
	/// \param	other				Ray chunk to set this object from. 
	/// \param [in,out]	d_srcAddr	The source addresses (device memory). \a d_srcAddr[i] defines at
	/// 							which original index the new value for the i-th ray can be found. 
	/// \param	numSrcAddr			Number of source address. This is also the new number of rays in
	/// 							this object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetFrom(const RayChunk& other, uint* d_srcAddr, uint numSrcAddr);

#endif // __cplusplus

	// GPU only data

	/// Ray origin for each ray (device memory).
	float4* d_origins;
	/// Ray direction for each ray (device memory).
	float4* d_dirs;
	/// Associated pixel index for each ray (device memory).
	uint* d_pixels;
	/// \brief	Ray influence for each ray (device memory). 
	///
	///			One value for each color component (r, g, b). It is used to record the influence
	///			of secondary rays or final gather rays on the radiance result.
	float4* d_influences;

	// Chunk information

	/// Recursion depth of the rays in this chunk.
	uint rekDepth;
	/// Number of rays.
	uint numRays;
	/// Maximum number of rays that can be stored.
	uint maxRays;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	RayPool
///
/// \brief	Performs ray generation and keeps track on which rays to process.
///
///			Manages a list of RayChunk objects that are not yet processed by the ray tracer.
///
/// \author	Mathias Neumann
/// \date	03.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class RayPool
{
public:
	/// Default constructur.
	RayPool(void);
	/// Destructor. Calls Destroy().
	~RayPool(void);

private:
	// If inited.
	bool m_bInited;
	// Maximum number of rays in each chunk.
	uint m_maxRaysPerChunk;
	// Number of samples per pixel in x and y direction (for stratified sampling).
	uint m_nSamplesPerPixelX;
	uint m_nSamplesPerPixelY;

	// Memory chunks.
	std::vector<RayChunk*> m_RayChunks;

	// Chunk that was used for the current task or NULL, if no current task.
	RayChunk* m_pChunkTask;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(uint maxRays, uint numSamplesPerPixelX = 1, uint numSamplesPerPixelY = 1)
	///
	/// \brief	Initializes the ray pool. 
	///
	/// \author	Mathias Neumann
	/// \date	03.02.2010
	///
	/// \param	maxRays				The maximum number of rays per RayChunk. 
	/// \param	numSamplesPerPixelX	Number of samples per pixel X (for stratification). Passing
	/// 							values larger one here was not considered for some time. 
	/// \param	numSamplesPerPixelY	Number of samples per pixel Y (for stratification).  Passing
	/// 							values larger one here was not considered for some time. 
	///
	/// \todo	Ensure that multiple samples per pixel still work, even when using all acceleration
	/// 		techniques. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(uint maxRays, uint numSamplesPerPixelX = 1, uint numSamplesPerPixelY = 1);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Destroys the pool. Releases and destroys all chunks.
	///
	/// \author	Mathias Neumann
	/// \date	03.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void AddPrimaryRays(CameraModel* pCamera, bool useMultiSample)
	///
	/// \brief	Adds primary rays and stores them in this pool. 
	///
	/// \author	Mathias Neumann
	/// \date	03.02.2010
	///
	/// \param [in]		pCamera	Current camera object. May not be \c NULL. 
	/// \param	useMultiSample	Whether to use multi-sampling. If \c true is passed, the number of
	/// 						samples per pixel is given by Initialize() parameters. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void AddPrimaryRays(CameraModel* pCamera, bool useMultiSample);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool GenerateChildRays(RayChunk* pChunk, ShadingPoints& shadingPts, TriangleData& triData,
	/// 	bool bGenReflect, bool bGenTransmit)
	///
	/// \brief	Generates child rays for given ray chunk and hit points.
	///
	///			The resulting ray chunks are added to the pool. Currently reflected and transmitted
	///			chunks are handled within separate ray chunks. This was done to avoid access conflicts
	///			when processing two rays for the same pixel in parallel. By adding a write priority to
	///			these rays, one could easily solve this problem.
	///
	///			This method handles specular reflection/transmission only. Furthermore the maximum
	///			recursion depth is limited. Currently this limit is fixed to 1 to avoid secondary
	///			ray generation completely.
	///
	///	\todo	Reenable and test secondary ray generation.
	///
	/// \warning	This will only work if \a pChunk is an \e active task that was returned from
	///				GetNextChunk() and for which FinalizeChunk() was not yet called.
	///
	/// \author	Mathias Neumann
	/// \date	03.02.2010
	///
	/// \param [in]	pChunk			The original ray chunk (parent rays).
	/// \param [in]	shadingPts		Hit points of original rays.
	/// \param [in]	triData			Scene geometry information.
	/// \param	bGenReflect			Pass \c true to generate reflected rays.
	/// \param	bGenTransmit		Pass \c true to generate transmitted rays.
	///
	/// \return	Returns \c true if some rays were generated.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool GenerateChildRays(RayChunk* pChunk, ShadingPoints& shadingPts, TriangleData& triData,
		bool bGenReflect, bool bGenTransmit);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Clear()
	///
	/// \brief	Removes all ray chunks from the pool.
	///
	///			Do not call this when there is an active chunk that wasn't finalized (see FinalizeChunk())
	///			yet.
	///
	/// \author	Mathias Neumann
	/// \date	03.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Clear();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	RayChunk* GetNextChunk()
	///
	/// \brief	Gets the next chunk to process.
	///
	///			This will only work when there is no other active ray chunk. When you are
	///			done processing the chunk, call FinalizeChunk().
	///
	/// \author	Mathias Neumann
	/// \date	February 2010
	///
	/// \return	Returns an unprocessed ray chunk or \c NULL, if no more chunks available.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	RayChunk* GetNextChunk();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void FinalizeChunk(RayChunk* pTask)
	///
	/// \brief	Finalizes given ray chunk.
	/// 		
	/// 		Call this for \e active ray chunks only, i.e. chunks returned from GetNextChunk()
	/// 		that are not yet finalized. 
	///
	/// \author	Mathias Neumann
	/// \date	February 2010
	///
	/// \param [in,out]	pTask	The ray chunk to finalize. The chunk may not be used anymore after
	/// 						calling this method. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void FinalizeChunk(RayChunk* pTask);

// Accessors
public:
	/// Returns \c true iff there is an active chunk that is currently processed by the caller.
	bool isChunkActive() { return m_pChunkTask != NULL; }
	/// Returns whether there are more ray chunks to process.
	bool hasMoreRays() { return FindProcessingChunk() != NULL; }
	/// Returns the number of samples per pixel.
	uint GetSamplesPerPixel() { return m_nSamplesPerPixelX*m_nSamplesPerPixelY; }

private:
	// Checks where to allocate new rays.
	RayChunk* FindAllocChunk();
	// Allocates a new chunk and returns it's address.
	RayChunk* AllocateNewChunk(uint maxRays);
	// Finds the best chunk to use for processing. Returns NULL if no more chunks.
	RayChunk* FindProcessingChunk();
};


#endif // __MN_RAYPOOL_H__