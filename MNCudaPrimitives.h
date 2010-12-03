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
/// \file	MNRT\MNCudaPrimitives.h
///
/// \brief	Declares the MNCudaPrimitives class. 
/// \author	Mathias Neumann
/// \date	29.07.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_CUDA_PRIMITIVES_H__
#define __MN_CUDA_PRIMITIVES_H__

#pragma once

#include <map>
#include <cudpp/cudpp.h>
#include "MNMath.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNCudaPrimitives
///
/// \brief	Wrapper class for CUDA parallel primitives implementation. 
///
///			Main concern is hiding CUDPP from user. Check
///			http://www.gpgpu.org/static/developer/cudpp/rel/cudpp_1.1/html/ for more information
///			about CUDPP. CUDPP requires plans to run operations. These plans include preallocated
///			auxiliary data structures of a given maximum size that is specified when creating the
///			plan. Since MNRT uses parallel primitives at countless places, I decided to add
///			this class and to centralize plan management.
///
///			For each primitive, a plan is created at construction of this class. Note that this
///			requires that the first call to GetInstance() occurrs \em after CUDA has been
///			initialized. These plans are used to execute all CUDPP operations. Once there is
///			a request that does not fit into the given plan due to exceeding element counts,
///			the given plan is resized by recreation.
///
///			Class is designed as singleton and might need optimizations for when used from
///			multiple CPU-threads.
///
///	\todo	Implement plan management in a more GPU memory friendly way.
/// \todo	Move reduction and segmented reduction to this class.
///
/// \author	Mathias Neumann
/// \date	29.07.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNCudaPrimitives
{
private:
	// Types of parallel primitives supported by MNCudaPrimitives class.
	enum MNPrimitiveType
	{
		// Scan (ADD, FWD, EXCLUSIVE).
		MNPrim_ScanAddE,
		// Scan (ADD, FWD, INCLUSIVE).
		MNPrim_ScanAddI,
		// Compact.
		MNPrim_Compact,
		// Segmented scan (ADD, FWD, INCLUSIVE).
		MNPrim_SegScanAddI,
		// Segmented scan (ADD, FWD, EXCLUSIVE).
		MNPrim_SegScanAddE,
		// Sort plan (KEY-VALUE-PAIR radix sort).
		MNPrim_SortKeyValue
	};

private:
	/// Holds primitive information including CDUPP plan handle.
	class CUDAPrimitive
	{
	public:
		CUDAPrimitive(const CUDPPConfiguration& _config, const CUDPPHandle& _plan, size_t _maxElemCount)
		{
			config = _config;
			plan = _plan;
			maxElemCount = _maxElemCount;
		}
	public:
		/// Configuration to create plan.
		CUDPPConfiguration config;
		/// Plan handle.
		CUDPPHandle plan;
		/// Maximum element count.
		size_t maxElemCount;
	};

	// Singleton. Hide constructors.
private:
	MNCudaPrimitives(void);
	MNCudaPrimitives(const MNCudaPrimitives& other);
public:
	~MNCudaPrimitives(void);

	// Attributes
private:
	// Primitives
	std::map<MNPrimitiveType, CUDAPrimitive*> m_mapPrims;

	
// Class
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNCudaPrimitives& GetInstance()
	///
	/// \brief	Returns the only MNCudaPrimitives instance.
	///
	///			Must be called \em after CUDA initialization as the first call initializes CUDPP
	///			plans.
	/// 		
	/// \warning Not thread-safe! 
	///
	/// \author	Mathias Neumann
	/// \date	29.07.2010
	///
	/// \return	The instance. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNCudaPrimitives& GetInstance();

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void DestoryPlans()
	///
	/// \brief	Destroys all CUDPP plans. 
	///
	///			Call this just before \c cudaThreadExit() to avoid errors.
	///
	/// \author	Mathias Neumann
	/// \date	06.10.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void DestoryPlans();

// Parallel primitives
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Scan(const void* d_in, size_t numElems, bool bInclusive, void* d_out)
	///
	/// \brief	Performs scan operation on given array \a d_in. Result is stored in \a d_out.
	/// 		
	/// 		The performed scan has the parameters ADD, FWD. Inplace scan is supported.
	/// 		
	/// 		\warning Use for 32-bit data types only, e.g. ::uint. 
	///
	/// \author	Mathias Neumann
	/// \date	29.07.2010
	///
	/// \param [in]		d_in	The data to scan (device memory). 
	/// \param	numElems		Number of \em elements in \a d_in. 
	/// \param	bInclusive		Whether to use inclusive or exclusive scan. 
	/// \param [out]	d_out	The result of scanning \a d_in (device memory). According to CUDPP
	/// 						developers, this can be equal to \a d_in (inplace operation). 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Scan(const void* d_in, size_t numElems, bool bInclusive, void* d_out);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Compact(const void* d_in, const unsigned* d_isValid, size_t numElems,
	/// 	void* d_outCompacted, size_t* d_outNewCount)
	///
	/// \brief	Compacts given array \a d_in using binary array \a d_isValid.
	///
	///			Note that the number of compacted elements is returned in form of device memory.
	///			This was done to try to avoid the read back to host memory when possible. This
	///			operation may not be used inplace.
	/// 		
	/// 		\warning Use for 32-bit data types only, e.g. ::uint. 
	///
	/// \author	Mathias Neumann
	/// \date	29.07.2010
	///
	/// \param	[in]	d_in			The array to compact (device memory). 
	/// \param	d_isValid				Binary 0/1 array (device memory). Same size as \a d_in. 
	///									If \a d_isValid[i] = 1, the i-th element is included. Else, if
	///									\a d_isValid[i] = 0, the i-th element is ignored.
	/// \param	numElems				Number of elements in \a d_in. 
	/// \param [out]	d_outCompacted	The compacted array (device memory). 
	/// \param [out]	d_outNewCount	Number of elements in compacted array (device memory). 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Compact(const void* d_in, const unsigned* d_isValid, size_t numElems, 
		void* d_outCompacted, size_t* d_outNewCount);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SegmentedScan(const void* d_in, const uint* d_flags, size_t numElems,
	/// 	bool bInclusive, void *d_out) private: void CreatePlans(size_t maxElements)
	///
	/// \brief	Performs a segmented scan on \a d_in. 
	///
	///			The flag array \a d_flags marks the start of segments
	/// 		as follows: 1 marks the start of a new segment. 0 says the element belongs to the
	/// 		previous segment. The performed scan has the parameters ADD, FWD. Note that
	///			SegmentedScan() can be called inplace with \a d_in = \a d_out.
	///
	/// \author	Mathias Neumann
	/// \date	29.07.2010
	///
	/// \param	d_in			The array to scan (device memory). 
	/// \param	d_flags			The flags array (device memory). 1 for segment start, else 0. 
	/// \param	numElems		Number of elements in \a d_in and \a d_flags. 
	/// \param	bInclusive		Whether to perform an inclusive or exclusive segmented scan. For
	/// 						inclusive scans, \a d_out[j] would contain the sum of all elements in
	/// 						the j-th segment including \a d_in[i0], when i0 is the first element
	///							of the segment. Else, for exclusive scans, the first element \a d_in[i0]
	///							will be ignored.
	/// \param [out]	d_out	The result of segmented scanning \a d_in (device memory). According to
	///							CUDPP developers, this can be equal to d_in (inplace operation).
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SegmentedScan(const void* d_in, const uint* d_flags, size_t numElems,
		bool bInclusive, void *d_out);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Sort(void* d_ioKeys, void* d_ioValues, uint keyValueMax, size_t numElems)
	///
	/// \brief	Sorts the given (key, value) pair set by the key values. Operation is performed
	/// 		inplace.
	/// 		
	/// 		\warning Large temporary memory required by \c cudppSort (8 byte per element to sort)
	/// 		and cudppSort element count limitations. Such a limitation seems to be 64M according
	/// 		to the CUDPP group. 
	///
	/// \author	Mathias Neumann
	/// \date	29.07.2010
	///
	/// \param [in,out]	d_ioKeys	The key components. Sorted after. 
	/// \param [in,out]	d_ioValues	The value components. 
	/// \param	keyValueMax			The key value maximum. Used to optimize sorting by restricting
	/// 							the number of significant key bits. 
	/// \param	numElems			Number of elements in both \a d_ioKeys and \a d_ioValues. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Sort(void* d_ioKeys, void* d_ioValues, uint keyValueMax, size_t numElems);

// Implementation
private:
	// Creates CUDPP plans for given maximum of elements.
	void CreatePlans(size_t maxElements);
	void CreatePlan(MNPrimitiveType type, const CUDPPConfiguration& config, size_t maxElemCount);
	// Ensures CUDPP plan is large enough for given element count. Returns true if plan was recreated.
	bool CheckPlanSize(MNPrimitiveType type, size_t requiredMaxElems);
};

#endif // __MN_CUDA_PRIMITIVES_H__