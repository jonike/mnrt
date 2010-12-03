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
/// \file	MNRT\MNCudaUtil.h
///
/// \brief	Provides useful macros and CUDA functions.
/// \author	Mathias Neumann
/// \date	16.02.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	cudautil	CUDA-related Utilities
/// 
/// \brief	General components used for GPU-based implementation. 
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_CUDAUTIL_H__
#define __MN_CUDAUTIL_H__

#pragma once

#include "MNUtilities.h"
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>


/// Performs a CUDA synchronization and checks for returned errors (optional in release mode).
#ifdef _DEBUG
#define MNCUDA_CHECKERROR mncudaSafeCallNoSync(mncudaCheckError(true))
#else
#define MNCUDA_CHECKERROR mncudaSafeCallNoSync(mncudaCheckError(false))
#endif

/// CUDA error check macro. Comparable to cudaSafeCallNoSync, however using ::MNFatal().
#define mncudaSafeCallNoSync(err)     __mncudaSafeCallNoSync(err, __FILE__, __LINE__)
/// CUDA CUtil error check macro. Uses ::MNFatal().
#define mncudaCheckErrorCUtil(err)     __mncudaCheckErrorCUtil(err, __FILE__, __LINE__)


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_DIVUP(count, chunkSize) (count / chunkSize + ((count % chunkSize)?1:0))
///
/// \brief	Divides \a count by \a chunkSize and adds 1 if there is some remainder.
///
/// \author	Mathias Neumann
/// \date	25.04.2010
///
/// \param	count		Count to divide in chunks, e.g. number of elements. 
/// \param	chunkSize	Size of each chunk. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_DIVUP(count, chunkSize) ((count) / (chunkSize) + (((count) % (chunkSize))?1:0))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_MAKEGRID2D(numBlocks, maxBlocks) dim3(min(numBlocks, maxBlocks), 1 + numBlocks
/// 		/ maxBlocks, 1)
///
/// \brief	Avoids the maximum CUDA grid size by using two grid dimensions for a one dimensional
/// 		grid. I added this due to problems with exceeding 1D grid sizes. 
///
/// \todo	Evaluate impact of spawning many more threads, e.g. when we got only maxBlocks+1
///			threads. In this case, the second slice of blocks would also get maxBlocks blocks.
///
/// \author	Mathias Neumann
/// \date	21.03.2010
///
/// \param	numBlocks	Number of thread blocks. 
/// \param	maxBlocks	Maximum size of the grid (number of thread blocks) in the first dimension. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_MAKEGRID2D(numBlocks, maxBlocks) dim3(min((numBlocks), (maxBlocks)), 1 + (numBlocks) / (maxBlocks), 1)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_GRID2DINDEX
///
/// \brief	Calculates the one dimensional block index for the given 2D grid that was created by
/// 		MNCUDA_MAKEGRID2D(). 
///
/// \author	Mathias Neumann
/// \date	21.03.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_GRID2DINDEX  (blockIdx.x + (blockIdx.y*gridDim.x))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_ALIGN_BYTES(size, alignment) ( ( ((size) % (alignment)) == 0 ) ? (size) :
/// 		((size) + (alignment) - ((size) % (alignment))) )
///
/// \brief	Computes the aligned byte size for a given alignment.
///
///			This is required to gain profit from coalesced access
/// 		or assign linear memory to textures without using offsets. 
///
/// \author	Mathias Neumann
/// \date	21.03.2010
///
/// \param	size		The size in bytes to align. 
/// \param	alignment	The alignment in bytes. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_ALIGN_BYTES(size, alignment) \
		( ( ((size) % (alignment)) == 0 ) ? \
					(size) : \
					((size) + (alignment) - ((size) % (alignment))) )

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_ALIGN(count) MNCUDA_ALIGN_EX(count, 16)
///
/// \brief	Computes the aligned \em element count for an alignment of 16 \em elements. 
///
///			This is required to gain profit from coalesced access. 
///
/// \author	Mathias Neumann
/// \date	14.03.2010
///
/// \param	count	Number of \em elements to align. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_ALIGN(count) MNCUDA_ALIGN_EX(count, 16)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_ALIGN_NZERO(count) MNCUDA_ALIGN_EX((((count) == 0) ? 1 : (count)), 16)
///
/// \brief	Computes the aligned \em element count with special treatment for zero counts. 
///
///			This version avoids zero counts by aligning them to a non-zero value.
///
/// \author	Mathias Neumann
/// \date	16.08.2010
///
/// \param	count	Number of \em elements to align. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_ALIGN_NZERO(count) MNCUDA_ALIGN_EX((((count) == 0) ? 1 : (count)), 16)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	MNCUDA_ALIGN_EX(count, alignment) ( ( ((count) % (alignment)) == 0 ) ? (count) :
/// 		((count) + (alignment) - ((count) % (alignment))) )
///
/// \brief	Computes the aligned element count, extended version.
/// 		
/// 		We allow here to pass an alignment (number of \em elements). This can be useful when
/// 		aligning counts for linear texture memory which requires a special device-dependent
/// 		alignment. 
///
/// \author	Mathias Neumann
/// \date	26.03.2010
///
/// \param	count		Number of \em elements to align. 
/// \param	alignment	The alignment, as a number of \em elements. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MNCUDA_ALIGN_EX(count, alignment) \
	( ( ((count) % (alignment)) == 0 ) ? \
					(count) : \
					((count) + (alignment) - ((count) % (alignment))) )

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \enum	MNCudaOP
///
/// \brief	Operator types used in utility algorithms. These operators are usually used as
///			binary operators, e.g. when working component-wise on two arrays.
///
/// \author	Mathias Neumann
/// \date	30.06.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
enum MNCudaOP
{
	/// Addition.
	MNCuda_ADD,
	/// Subtraction.
	MNCuda_SUB,
	/// Multiplication.
	MNCuda_MUL,
	/// Division.
	MNCuda_DIV,
	/// Minimum.
	MNCuda_MIN,
	/// Maximum.
	MNCuda_MAX,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	ReduceOperatorTraits
///
/// \brief	Reduction operator traits class.
///
///			This class can be passed to CUDA kernels as a functor class. So one reduce kernel
///			can be used to perform addition, minimum or maximum reduces.
///
/// \author	Mathias Neumann
/// \date	30.06.2010
///
/// \tparam T		Base type of the reduction, e.g. \c float.
/// \tparam oper	Reduction operator in form of an MNCudaOP.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, MNCudaOP oper>
class ReduceOperatorTraits
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \fn	static inline __device__ __host__ T op(T& a, T& b)
    ///
    /// \brief	Performs reduction on \a a and \a b.
	///
	///			In case the template parameter \a oper is neither MNCuda_ADD, MNCuda_MIN nor MNCuda_MAX,
	///			this operator just performs an addition.
    ///
    /// \author	Mathias Neumann
    /// \date	30.06.2010
    ///
    /// \param [in]	a	The first value. 
    /// \param [in]	b	The second value.
    ///
    /// \return	Reduction result. 
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    static inline __device__ __host__ T op(T& a, T& b)
    {
		switch(oper) // COMPILE TIME
		{
		case MNCuda_ADD:
			return a + b;
		case MNCuda_MIN:
			return myMin(a, b);
		case MNCuda_MAX:
			return myMax(a, b);
		default:
			return a + b;
		}
    }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static inline __device__ __host__ T op(volatile T& a, volatile T& b)
	///
	/// \brief	Performs reduction on \a a and \a b (\c volatile variant for shared memory).
	/// 		
	/// 		In case the template parameter \a oper is neither MNCuda_ADD, MNCuda_MIN nor
	/// 		MNCuda_MAX, this operator just performs an addition. This variant is required \c
	/// 		volatile has to be used for shared memory variables to avoid compiler optimizations. 
	///
	/// \author	Mathias Neumann
	/// \date	08.09.2010
	///
    /// \param [in]	a	The first value. 
    /// \param [in]	b	The second value. 
	///
	/// \return	Reduction result.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static inline __device__ __host__ T op(volatile T& a, volatile T& b)
    {
		switch(oper) // COMPILE TIME
		{
		case MNCuda_ADD:
			return a + b;
		case MNCuda_MIN:
			return myMin(a, b);
		case MNCuda_MAX:
			return myMax(a, b);
		default:
			return a + b;
		}
    }

private:
	// Maximum and minimum functions as min/max are macros in C, check
	// http://www.aristeia.com/Papers/C++ReportColumns/jan95.pdf.
	// Therefore we define new custom functions for min/max. I tried templates first, however
	// float4 uses a different, component-wise approach, that doesn't fit into the scheme for
	// default min/max implementation.
	static inline __device__ __host__ float myMax(float& a, float& b)
	{ 
		return ((a) > (b)) ? (a) : (b); 
	}
	static inline __device__ __host__ float myMin(float& a, float& b)
	{ 
		return ((a) > (b)) ? (b) : (a); 
	}
	static inline __device__ __host__ float myMax(volatile float& a, volatile float& b)
	{ 
		return ((a) > (b)) ? (a) : (b); 
	}
	static inline __device__ __host__ float myMin(volatile float& a, volatile float& b)
	{ 
		return ((a) > (b)) ? (b) : (a); 
	}

	static inline __device__ __host__ uint myMax(uint& a, uint& b)
	{ 
		return ((a) > (b)) ? (a) : (b); 
	}
	static inline __device__ __host__ uint myMin(uint& a, uint& b)
	{ 
		return ((a) > (b)) ? (b) : (a); 
	}
	static inline __device__ __host__ uint myMax(volatile uint& a, volatile uint& b)
	{ 
		return ((a) > (b)) ? (a) : (b); 
	}
	static inline __device__ __host__ uint myMin(volatile uint& a, volatile uint& b)
	{ 
		return ((a) > (b)) ? (b) : (a); 
	}

	static inline __device__ __host__ float4 myMax(float4& a, float4& b)
	{
		return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
	}

	static inline __device__ __host__ float4 myMin(float4& a, float4& b)
	{
		return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
	}
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline void __mncudaSafeCallNoSync(cudaError err, const char *file, const int line)
///
/// \brief	Checks for CUDA errors without synchronization. Uses ::MNFatal() to report the error.	
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	err		The error code. 
/// \param	file	The file where the error occurred. 
/// \param	line	The line where the error occurred. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline void __mncudaSafeCallNoSync(cudaError err, const char *file, const int line)
{
    if(cudaSuccess != err) 
	{
		// "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		// when the user double clicks on the error line in the output panel.
        MNFatal("%s(%i) : CUDA Runtime API error : %s.\n", file, line, cudaGetErrorString(err));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline void __mncudaCheckErrorCUtil(CUTBoolean err, const char *file, const int line)
///
/// \brief	Checks for CUDA CUtil errors without synchronization. Uses ::MNFatal() to report the
/// 		error. 
///
/// \author	Mathias Neumann
/// \date	05.10.2010
///
/// \param	err		The error code. 
/// \param	file	The file where the error occurred. 
/// \param	line	The line where the error occurred.  
////////////////////////////////////////////////////////////////////////////////////////////////////
inline void __mncudaCheckErrorCUtil(CUTBoolean err, const char *file, const int line)
{
    if(CUTTrue != err) 
	{
		// "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		// when the user double clicks on the error line in the output panel.
        MNFatal("%s(%i) : CUTIL CUDA error.\n", file, line);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	cudaError_t mncudaCheckError(bool bForce = true)
///
/// \brief	Checks for pending errors and returns them.
///
///			A synchronization is required to get all such errors. Hence this might be too
///			costly for release mode. Therefore I added a way to make these checks \e optional in
///			release mode, more precisely controllable using ::mncudaEnableErrorChecks().
///
/// \author	Mathias Neumann
/// \date	03.11.2010
///
/// \param	bForce	\c true to force error check. Is used in debug mode, see ::MNCUDA_CHECKERROR, to
///					enforce error checks, even if disabled. If \c false is passed, the error check
///					is performed only if checks are enabled.
///
/// \return	Return value from \c cudaThreadSynchronize(), if check is performed. Else \c
///			cudaSuccess is returned.
////////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t mncudaCheckError(bool bForce = true);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaEnableErrorChecks(bool bEnable = true)
///
/// \brief	Enables error checks using ::mncudaCheckError().
///
///			This function is useful in release mode only, as in debug mode, error checks are
///			forced.
///
/// \author	Mathias Neumann
/// \date	03.11.2010
///
/// \param	bEnable	\c true to enable, \c false to disable. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaEnableErrorChecks(bool bEnable = true);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	bool mncudaIsErrorChecking() const
///
/// \brief	Gets whether error checking is enabled.
///
/// \author	Mathias Neumann
/// \date	03.11.2010
///
/// \return	\c true if enabled, else \c false.
////////////////////////////////////////////////////////////////////////////////////////////////////
bool mncudaIsErrorChecking();

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	uint mncudaGetMaxBlockSize(uint reqSharedMemPerThread, uint maxRegPerThread,
/// 	bool useMultipleWarpSize = true)
///
/// \brief	Computes best thread block size for given shared memory requirement.
/// 		
/// 		Thread block size is limited due to maximum shared memory per block. This maximum
/// 		depends on the actual CUDA GPU.
/// 		
/// 		\warning This assumes 16 bytes (for blockIdx, ...) + 256 bytes (for parameters)
///					 of shared memory are reserved. That might not be correct for all GPUs.
///					 Value are taken from CUDA FAQ, see 
///					 http://forums.nvidia.com/index.php?showtopic=84440.
///
/// \author	Mathias Neumann
/// \date	29.10.2010
///
/// \param	reqSharedMemPerThread	Required shared memory per \em thread in bytes. 
/// \param	maxRegPerThread			Maximum number of registers per \em thread.
/// \param	useMultipleWarpSize		Whether to round the maximum to the next multiple of the
/// 								device's warp size. 
///
/// \return	Maximum thread block size for current CUDA device. Note that the returned value is \e
/// 		not rounded to a valid power of two. 
////////////////////////////////////////////////////////////////////////////////////////////////////
uint mncudaGetMaxBlockSize(uint reqSharedMemPerThread, uint maxRegPerThread, 
						   bool useMultipleWarpSize = true);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaInitIdentity(uint* d_buffer, uint count)
///
/// \brief	Initializes the given buffer with the identity relation, that is buffer[i] = i. 
///
///			Each component is handled by it's own CUDA thread.
///
/// \author	Mathias Neumann
/// \date	16.02.2010
///
/// \param [in,out]	d_buffer	Device buffer to initialize. 
/// \param	count				The number of elements in \a d_buffer. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaInitIdentity(uint* d_buffer, uint count);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> void mncudaInitConstant(T* d_buffer, uint count, T constant)
///
/// \brief	Initializes the given buffer with a constant value. 
///
///			Each spawned thread works on one array component.
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
/// \tparam T					Element type of the buffer.
/// \param [in,out]	d_buffer	Device buffer to initialize. 
/// \param	count				Number of elements. 
/// \param	constant			The constant. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void mncudaInitConstant(T* d_buffer, uint count, T constant);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaAddIdentity(uint* d_buffer, uint count)
///
/// \brief	Adds the index to all elements of the given buffer.
///
///			This corresponds to adding the identity relation to the elements. Here each thread
///			works on one buffer component.
///
/// \author	Mathias Neumann
/// \date	22.03.2010
///
/// \param [in,out]	d_buffer	The buffer to update.
/// \param	count				Number of elements.
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaAddIdentity(uint* d_buffer, uint count);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <MNCudaOP op, class T> void mncudaConstantOp(T* d_array, uint count, T constant)
///
/// \brief	Performs constant operation on all array elements:
///
///			\code d_array[i] = d_array[i] op constant \endcode
///
///			Each spawned thread works on one array component.
///
/// \author	Mathias Neumann
/// \date	22.04.2010
///
/// \tparam op				Operator type.
///	\tparam	T				Element type.
/// \param [in,out]	d_array	The array to manipulate.
/// \param	count			Number of elements.
/// \param	constant		The constant.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <MNCudaOP op, class T>
void mncudaConstantOp(T* d_array, uint count, T constant);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class V, class S> void mncudaScaleVectorArray(V* d_vecArray, uint count,
/// 	S scalar)
///
/// \brief	Scales given vector array by given scalar (component wise).
///
///			Each spawned thread works on one array component.
///
/// \author	Mathias Neumann
/// \date	27.06.2010
///
///	\tparam	V				Vector type.
///	\tparam	S				Scalar type.
/// \param [in,out]	d_vecArray	The array of vectors to scale. 
/// \param	count				Number of vectors. 
/// \param	scalar				The scalar. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class V, class S>
void mncudaScaleVectorArray(V* d_vecArray, uint count, S scalar);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class V, class S> void mncudaAverageArray(V* d_array, uint count, S* d_counts)
///
/// \brief	Averages given vector array (type of V) by dividing each element by the corresponding
/// 		count.
/// 		
/// 		\code d_array[i] = d_array[i] / d_counts[i] \endcode
/// 		
/// 		If \c d_counts[i] is zero, \c d_array[i] stays unchanged. Each spawned thread works
/// 		on one component of the buffers. 
///
/// \author	Mathias Neumann
/// \date	12.07.2010
///
///	\tparam	V				Vector type.
///	\tparam	S				Scalar type.
/// \param [in,out]	d_array		The array of vectors to scale. 
/// \param	count				Number of elements in both \a d_array and \a d_counts. 
/// \param [in]		d_counts	Counts used for averaging. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class V, class S>
void mncudaAverageArray(V* d_array, uint count, S* d_counts);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaInverseBinary(uint* d_buffer, uint count)
///
/// \brief	Inverses the given "binary" 0-1-buffer.
/// 		
/// 		This is done by setting all one (1) components to zero (0) and all zero components to
/// 		one respectively. Each thread handles one component of the buffer.
///
/// \author	Mathias Neumann
/// \date	19.02.2010
///
/// \param [in,out]	d_buffer	The binary buffer to inverse. 
/// \param	count				Number of elements in d_buffer. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaInverseBinary(uint* d_buffer, uint count);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, MNCudaOP op> void mncudaArrayOp(T* d_target, T* d_other, uint count)
///
/// \brief	Performs an operation on the given two arrays.
///
///			\code d_target[i] = d_target[i] op d_other[i] \endcode
///
///			Here each thread handles one component \c i.
///
/// \author	Mathias Neumann
/// \date	27.02.2010
///
/// \tparam op				Operator type.
///	\tparam	T				Element type.
/// \param [in,out]	d_target	Target array. 
/// \param [in]		d_other		Other array. 
/// \param	count				Number of elements in both arrays. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <MNCudaOP op, class T>
void mncudaArrayOp(T* d_target, T* d_other, uint count);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> uint mncudaResizeMNCudaMem(T** d_buffer, uint numOld, uint numRequest,
/// 	uint slices = 1)
///
/// \brief	Resizes the given buffer from \a numOld to at least \a numRequest elements.
/// 		
/// 		Copies the contents of the old buffer to the beginning of the new buffer. Works only
/// 		for MNCudaMemPool buffers. 
///
///			Additionally, the buffer can be organized into slices of contiguously placed
///			elements. When having \a slices > 1, it is assumed that \a d_buffer has \a numOld
///			times \a slices elements before the call. The buffer is resized so that each
///			slice is resized to at least \a numRequest elements.
///
/// \author	Mathias Neumann
/// \date	16.02.2010
///
/// \tparam T			Element type of buffer.
/// \param [in,out]	d_buffer	The old buffer. After execution it is free'd and replaced by the
/// 							new buffer. Has to be valid MNCudaMemPool memory. 
/// \param	numOld				Old number of \em elements in a single slice. 
/// \param	numRequest			Requested new number of \em elements in a single slice. 
/// \param	slices				Number of slices. Usually 1, however you might have a buffer
/// 							consiting of multiple slices of \a numOld elements and want to
/// 							resize it to multiple slices of at least \a numNew elements. 
///
/// \return	New element count within single slice. Might be slightly greater than \a numRequest
/// 		in case \a numRequest wasn't correcly aligned. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
uint mncudaResizeMNCudaMem(T** d_buffer, uint numOld, uint numRequest, uint slices = 1);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> cudaError_t mncudaReduce(T& result, T* d_data, uint count, MNCudaOP op,
/// 	T identity)
///
/// \brief	Performs reduction on \a d_data.
///
///			\a d_data remains unchanged. The reduction depends on the passed operator. The
///			reduction algorithm is implemented with the help of the CUDA SDK sample.
///
/// \author	Mathias Neumann
/// \date	16.03.2010
///
/// \tparam T		Element type of input and output arrays.
/// \param [out]	result	Reduction result. Single element of type \a T. 
/// \param [in]		d_data	Data to reduce, remains unchanged. 
/// \param	count			Number of elements in \a d_data. 
/// \param	op				The reduction operator. One of ::MNCuda_ADD, ::MNCuda_MIN,
/// 						::MNCuda_MAX. 
/// \param	identity		Identity value associated with \a op. 
///
/// \return	\a cudaSuccess if successful, else some error value. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
cudaError_t mncudaReduce(T& result, T* d_data, uint count, MNCudaOP op, T identity);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> cudaError_t mncudaSegmentedReduce(T* d_data, uint* d_owner, uint count,
/// 	MNCudaOP op, T identity, T* d_result, uint numSegments)
///
/// \brief	Performs segmented reduction on \a d_data.
/// 		
/// 		Segments are defined by \a d_owner, where \a d_owner[i] contains the segment of \a
/// 		d_data[i]. The result is put into \a d_result. This array has to be preallocated and
/// 		should have space for all segment results.
/// 		
/// 		Algorithmic idea: http://www.nvidia.com/object/nvidia_research_pub_013.html. 
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
///	\tparam T		Element type of \a d_data.
/// \param [in]		d_data		The segmented data to perform reduction on. Count A. It is
/// 							assumed that the data of the same segment is stored contiguously. 
/// \param [in]		d_owner		The data-segment association list. Count A. 
/// \param	count				Defines A, that is the count in \a d_data and \a d_owner. 
/// \param	op					The reduction operator. One of ::MNCuda_ADD, ::MNCuda_MIN,
/// 							::MNCuda_MAX. 
/// \param	identity			Identity value associated with \a op.
/// \param [out]	d_result	Takes the result, that is the reduction result of each segment.
/// 							Size of B <= A. 
/// \param	numSegments			Defines B, the number of segments. 
///
/// \return	\a cudaSuccess if OK, else some error. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
cudaError_t mncudaSegmentedReduce(T* d_data, uint* d_owner, uint count, MNCudaOP op, T identity, 
								  T* d_result, uint numSegments);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> void mncudaSetAtAddress(T* d_array, uint* d_address, T* d_vals,
/// 	uint countVals)
///
/// \brief	Moves data from device memory \a d_vals to device memory \a d_array using target
/// 		addresses specified in \a d_address.
///
///			\code d_array[d_address[i]] = d_vals[i] \endcode
/// 		
/// 		\warning	Heavy uncoalesced access possible. Depends on addresses. 
///
/// \author	Mathias Neumann
/// \date	22.03.2010
/// \see	::mncudaSetConstAtAddress()
///
/// \tparam T		Element type of array and values.
/// \param [in,out]	d_array		The data array to manipulate. 
/// \param [in]		d_address	Addresses of the values to manipulate. It is assumed that
///								all addresses \a d_address[i] are valid with respect to \a
///								d_array.
/// \param [in]		d_vals		The values to fill in. 
/// \param	countVals			Number of values in \a d_address and \a d_vals. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void mncudaSetAtAddress(T* d_array, uint* d_address, T* d_vals, uint countVals);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> void mncudaSetConstAtAddress(T* d_array, uint* d_address, T constant,
/// 	uint countVals)
///
/// \brief	Moves a given constant to device memory \a d_array at the adresses specified in \a
/// 		d_address.
///
///			\code d_array[d_address[i]] = constant \endcode
/// 		
/// 		\warning	Heavy uncoalesced access possible. Depends on addresses. 
///
/// \author	Mathias Neumann
/// \date	18.02.2010
/// \see	::mncudaSetAtAddress()
///
/// \tparam T		Element type of array and constant.
/// \param [in,out]	d_array		The data array to manipulate. 
/// \param [in]		d_address	Addresses of the values to manipulate. 
/// \param	constant			The constant to move to \a d_array. 
/// \param	countVals			Number of values in \a d_address. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void mncudaSetConstAtAddress(T* d_array, uint* d_address, T constant, uint countVals);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> void mncudaSetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals,
/// 	uint countTarget)
///
/// \brief	Moves data from device memory \a d_vals to device memory \a d_array using \em source
/// 		addresses specified in \a d_srcAddr.
/// 		
/// 		\code d_array[i] = d_vals[d_srcAddr[i]] \endcode
/// 		
/// 		When the source address is \c 0xffffffff, the corresponding target entry will get zero'd.
///			This can be helpful for some algorithms.
/// 		
/// 		\warning	Heavy uncoalesced access possible. Depends on addresses. 
///
/// \author	Mathias Neumann
/// \date	24.03.2010
///
/// \tparam T		Element type of array and values.
/// \param [in,out]	d_array		The data array to manipulate. 
/// \param [in]		d_srcAddr	Addresses of the source values in \a d_vals. Use \c 0xffffffff to zero
/// 							the corresponding entry in \a d_array. 
/// \param [in]		d_vals		The values to fill in. 
/// \param	countTarget			Number of values in \a d_array and \a d_srcAddr. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void mncudaSetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals, uint countTarget);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaAlignCounts(uint* d_outAligned, uint* d_counts, uint count)
///
/// \brief	Aligns the given count array by aligning all element counts.
/// 		
/// 		\code d_outAligned[i] = MNCUDA_ALIGN_NZERO(d_counts[i]) \endcode
/// 		
/// 		This should be useful to get coalesced access when accessing offsets calculated by
/// 		scanning the count array. These offsets are aligned, too. Note that th
/// 		::MNCUDA_ALIGN_NZERO macro is used to provide special handling of zero counts. This
/// 		ensures that even zero counts would get aligned to a non-zero count and helps
/// 		avoiding problems with the corresponding offsets. If this would be left out, two
/// 		adjacent elements would get the same offsets. Parallel access at
/// 		these offsets could create a race condition. 
///
/// \author	Mathias Neumann
/// \date	14.03.2010
///
/// \param [out]	d_outAligned	The aligned counts array. Has to be preallocated. 
/// \param [in]		d_counts		The counts array. 
/// \param	count					Number of elements (counts). 
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaAlignCounts(uint* d_outAligned, uint* d_counts, uint count);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> cudaError_t mncudaPrintArray(T* d_array, uint count, bool isFloat,
/// 	const char* strArray = NULL)
///
/// \brief	Prints the given array using \c printf. 
///
///			For debugging purposes only. Too slow for release code. 
///
///	\todo	Find way to print out data of different types.
///
/// \author	Mathias Neumann
/// \date	15.03.2010
///
/// \tparam	T		Element type.
/// \param [in]		d_array	The device array to print out. 
/// \param	count			Number of elements of d_array to print. 
/// \param	isFloat			Whether \a T is floating point type. 
/// \param	strArray		The name of the array. Used for printing purposes. Might be \c NULL.
///
/// \return	\c cudaSuccess if OK, else some error. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
cudaError_t mncudaPrintArray(T* d_array, uint count, bool isFloat, const char* strArray = NULL);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> void mncudaCompactInplace(T* d_data, uint* d_srcAddr, uint countOld,
/// 	uint countNew)
///
/// \brief	Compacts the data array "inplace" using a temporary buffer, the given source 
///			adresses and count. 
///
///			Enables to compact a structure of arrays using only one real compact 
///			and multiple set from addresses. 
///
/// \author	Mathias Neumann
/// \date	17.06.2010
/// \see	::mncudaSetFromAddress(), ::mncudaGenCompactAddresses()
///
/// \tparam T		Element type.
/// \param [in,out]	d_data		The data array to compact. Operation is performed "inplace".
/// \param [in]		d_srcAddr	The source addresses. 
/// \param	countOld			The original count (elements in \a d_data). 
/// \param	countNew			The new count (elements in \a d_srcAddr). Note that this should
///								normally be available, as it will be retrieved from the source
///								address generation.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void mncudaCompactInplace(T* d_data, uint* d_srcAddr, uint countOld, uint countNew);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	uint mncudaGenCompactAddresses(uint* d_isValid, uint countOld, uint* d_outSrcAddr)
///
/// \brief	Generates compact addresses using CUDPP's compact.
/// 		
/// 		These can be used to compact data corresponding to \a d_isValid without using CUDPP's
/// 		compact, but using ::mncudaCompactInplace(). To compact a structure of arrays, you'd
/// 		have to call this once and ::mncudaCompactInplace() for each array. In my tests I
/// 		observed that this is much more efficient than multiple \c cudppCompact calls.
///
/// \note	This corresponds to compacting an identity array.
///
/// \author	Mathias Neumann
/// \date	01.07.2010 
///	\see	::mncudaCompactInplace(), ::mncudaSetFromAddress()
///
/// \param [in]		d_isValid		Contains 1 if entry is valid, 0 if it should be dropped. 
/// \param	countOld				Old count before compacting. 
/// \param [out]	d_outSrcAddr	The source address array. Device memory provided by caller. 
///
/// \return	Number of compacted elements, i.e. the number of source addresses. 
////////////////////////////////////////////////////////////////////////////////////////////////////
uint mncudaGenCompactAddresses(uint* d_isValid, uint countOld, uint* d_outSrcAddr);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	void mncudaNormalize(float4* d_vectors, uint numVectors)
///
/// \brief	Normalizes each element of a given vector array.
///
/// \author	Mathias Neumann
/// \date	23.10.2010
///
/// \param [in,out]	d_vectors	The vector array to normalize. Array is given as \c float4 array
///								of three-dimensional vectors to ensure alignment, where the actual
///								vector is in the xyz components.
/// \param	numVectors			Number of vectors. 
////////////////////////////////////////////////////////////////////////////////////////////////////
void mncudaNormalize(float4* d_vectors, uint numVectors);



#endif // __MN_CUDAUTIL_H__