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
/// \file	GPU\mncudautil.cu
///
/// \brief	CUDA implementations of the methods defined in MNCudaUtil.h.
/// \author	Mathias Neumann
/// \date	16.02.2010
/// \ingroup	cudautil
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MNMath.h"
#include "MNCudaUtil.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"

//#define TEST_SEG_REDUCE

/// \c float4 inequality operator (component-wise).
bool operator!=(const float4& a, const float4& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, class traits, uint blockSize> __device__ void dev_SegReduceBlock(
///			const uint* owner, T* data, T identity)
///
/// \brief	Performs segmented reduction within a single block in shared memory.
///
/// \author	Mathias Neumann
/// \date	23.04.2010
/// \see	kernel_SegmentedReducePrepare(), kernel_SegmentedReduceFinal()
///
/// \tparam	T			Type of elements.
/// \tparam blockSize	Thread block size to use. Maximum is 512. Should be power of two.
/// \tparam	opTraits	Operator to use. See ReduceOperatorTraits.
///
/// \param	owner			Owner array in shared memory.
/// \param [in,out]	data	Data array in shared memory.
/// \param	identity		Identity value for the given operation. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class traits, uint blockSize>
__device__ void dev_SegReduceBlock(const uint* owner, T* data, T identity)
{
    T left = identity;
	if( threadIdx.x >=   1 && owner[threadIdx.x] == owner[threadIdx.x -   1] ) { left = data[threadIdx.x -   1]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();  
    if( threadIdx.x >=   2 && owner[threadIdx.x] == owner[threadIdx.x -   2] ) { left = data[threadIdx.x -   2]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
    if( threadIdx.x >=   4 && owner[threadIdx.x] == owner[threadIdx.x -   4] ) { left = data[threadIdx.x -   4]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
    if( threadIdx.x >=   8 && owner[threadIdx.x] == owner[threadIdx.x -   8] ) { left = data[threadIdx.x -   8]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
    if( threadIdx.x >=  16 && owner[threadIdx.x] == owner[threadIdx.x -  16] ) { left = data[threadIdx.x -  16]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
    if( threadIdx.x >=  32 && owner[threadIdx.x] == owner[threadIdx.x -  32] ) { left = data[threadIdx.x -  32]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();  
    if( threadIdx.x >=  64 && owner[threadIdx.x] == owner[threadIdx.x -  64] ) { left = data[threadIdx.x -  64]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
    if( threadIdx.x >= 128 && owner[threadIdx.x] == owner[threadIdx.x - 128] ) { left = data[threadIdx.x - 128]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
	if( blockSize == 512 ) // COMPILE TIME
	{
		if( threadIdx.x >= 256 && owner[threadIdx.x] == owner[threadIdx.x - 256] ) { left = data[threadIdx.x - 256]; } __syncthreads(); data[threadIdx.x] = traits::op(data[threadIdx.x], left); left = identity; __syncthreads();
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_InitIdentity(uint* d_buffer, uint count)
///
/// \brief	Initializes the given buffer using identity relation.
///
//			\code d_buffer[idx] = idx; \endcode 
///
/// \author	Mathias Neumann
/// \date	16.02.2010
///
/// \param [in,out]	d_buffer	The buffer to initialize. 
/// \param	count				The element count of the buffer. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_InitIdentity(uint* d_buffer, uint count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
		d_buffer[idx] = idx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __global__ void kernel_InitConstant(T* d_buffer, uint count,
/// 	T constant)
///
/// \brief	Initializes the given buffer with a constant.
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
/// \tparam	T		Element type of buffer. 
///
/// \param [in,out]	d_buffer	The buffer to initialize.
/// \param	count				Number of elements in buffer. 
/// \param	constant			The constant.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void kernel_InitConstant(T* d_buffer, uint count, T constant)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
		d_buffer[idx] = constant;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_AddIdentity(uint* d_buffer, uint count)
///
/// \brief	Adds a constant to all elements of the buffer. 
///
/// \author	Mathias Neumann
/// \date	22.03.2010
///
/// \param [in,out]	d_buffer	The buffer to manipulate. 
/// \param	count				Number of elements in buffer. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AddIdentity(uint* d_buffer, uint count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
		d_buffer[idx] += idx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <MNCudaOP op, class T> __global__ void kernel_ConstantOp(T* d_array, uint count,
/// 	T constant)
///
/// \brief	Performs constant operation on all array elements.
///
///			\code d_array[idx] = d_array[idx] op constant; \endcode
///
/// \author	Mathias Neumann
/// \date	22.04.2010
///
/// \tparam op		The operator. Has to be one of ::MNCuda_ADD, ::MNCuda_SUB, ::MNCuda_MUL and
///					::MNCuda_DIV.
/// \tparam	T		Element type of buffer. 
///
/// \param [in,out]	d_array	The array to manipulate. 
/// \param	count			Number of elements in buffer. 
/// \param	constant		The constant. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <MNCudaOP op, class T>
__global__ void kernel_ConstantOp(T* d_array, uint count, T constant)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
	{
		switch(op) // COMPILE TIME
		{
		case MNCuda_ADD:
			d_array[idx] += constant;
			break;
		case MNCuda_SUB:
			d_array[idx] -= constant;
			break;
		case MNCuda_MUL:
			d_array[idx] *= constant;
			break;
		case MNCuda_DIV:
			d_array[idx] /= constant;
			break;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class V, class S> __global__ void kernel_ScaleVectorArray(V* d_vecArray,
/// 	uint count, S scalar)
///
/// \brief	Scales given vector array (V vector type, e.g. float4) by given scalar (type S). 
///
/// \author	Mathias Neumann
/// \date	27.06.2010
///
/// \tparam	V	Vector type, i.e. element type of \a d_vecArray. 
/// \tparam	S	Scalar type, i.e. type of \a scalar. Has to be compatible with vector type. 
///
/// \param [in,out]	d_vecArray	The array of vectors to scale. 
/// \param	count				Number of vectors in array. 
/// \param	scalar				The scalar. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class V, class S>
__global__ void kernel_ScaleVectorArray(V* d_vecArray, uint count, S scalar)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < count)
		d_vecArray[tid] *= scalar;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class V, class S> __global__ void kernel_AverageArray(V* d_array, uint count,
/// 	S* d_counts)
///
/// \brief	Computes averages for given value and count array.
/// 		
/// 		\code d_array[idx] /= d_counts[idx]; \endcode
/// 		
/// 		When the count is zero, the corresponding value is left unchanged. 
///
/// \author	Mathias Neumann
/// \date	12.07.2010
///
/// \tparam	V	Vector type, i.e. element type of \a d_array. 
/// \tparam	S	Scalar type, i.e. element type of \a d_counts. Has to be compatible with vector type. 
///
/// \param [in,out]	d_array		Array of values to average. 
/// \param	count				Number of values and counts. 
/// \param [in]		d_counts	Array of counts. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class V, class S>
__global__ void kernel_AverageArray(V* d_array, uint count, S* d_counts)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < count)
	{
		V temp = d_array[tid];
		if(d_counts[tid] != 0)
			temp /= d_counts[tid];
		d_array[tid] = temp;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_InverseBinary(uint* d_buffer, uint count)
///
/// \brief	Inverses the given binary 0/1 array.
///
///			All one (1) components are set to zero (0) and all zero components to one respectively. 
///
/// \author	Mathias Neumann
/// \date	19.02.2010
///
/// \param [in,out]	d_buffer	The binary 0/1 buffer to inverse. 
/// \param	count				Number of elements in buffer. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_InverseBinary(uint* d_buffer, uint count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
		d_buffer[idx] = 1 - d_buffer[idx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <MNCudaOP op, class T> __global__ void kernel_ArrayOp(T* d_target, T* d_other,
/// 	uint count)
///
/// \brief	Performs an operation on the given two arrays.
/// 		
/// 		\code d_target[idx] = d_target[idx] op d_other[idx]; \endcode
///
/// \author	Mathias Neumann
/// \date	27.02.2010
///
/// \tparam	op	The operator. Has to be one of ::MNCuda_ADD, ::MNCuda_SUB, ::MNCuda_MUL and
/// 			::MNCuda_DIV. 
/// \tparam	T	Element type of buffers. 
///
/// \param [in,out]	d_target	Target array. 
/// \param [in]		d_other		Other array. 
/// \param	count				Number of elements in both arrays. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <MNCudaOP op, class T>
__global__ void kernel_ArrayOp(T* d_target, T* d_other, uint count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
	{
		switch(op) // COMPILE TIME
		{
		case MNCuda_ADD:
			d_target[idx] = d_target[idx] + d_other[idx];
			break;
		case MNCuda_SUB:
			d_target[idx] = d_target[idx] - d_other[idx];
			break;
		case MNCuda_MUL:
			d_target[idx] = d_target[idx] * d_other[idx];
			break;
		case MNCuda_DIV:
			d_target[idx] = d_target[idx] / d_other[idx];
			break;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, uint blockSize, MNCudaOP op> __global__ void kernel_Reduce( 
/// 	T* d_data, uint count, T identity, T* d_results)
///
/// \brief	Performs per block reduction on given data array of arbitrary size.
///
/// 		Per block results are returned in \a d_results.
///
/// \note	Required shared memory per thread block of size N: \c sizeof(T) * N bytes.
///
/// \author	Mathias Neumann
/// \date	16.03.2010
/// \see	::dev_ReduceFast()
///
/// \tparam	T			Type of elements. \c float4 or other structs are not supported, see
///						::dev_ReduceFast().
/// \tparam blockSize	Thread block size to use. Maximum is 512. Should be power of two.
/// \tparam	opTraits	Operator to use. See ReduceOperatorTraits.
///
/// \param [in]		d_data		The data to reduce. 
/// \param	count				Number of elements in \a d_data. 
/// \param	identity			The identity value for op. 
/// \param [out]	d_results	Computed per block results.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, uint blockSize, class opTraits>
__global__ void kernel_Reduce(T* d_data, uint count, T identity, T* d_results)
{
	uint blk = blockIdx.x;
	uint idxBlockStart = blockIdx.x * (2*blockDim.x);
	uint idx = idxBlockStart + threadIdx.x;

	// Reduce within block.
	__shared__ T s_blockData[blockSize];

	// Padding with identity value.
	T v1 = identity, v2 = identity;
	if(idx < count)
		v1 = d_data[idx];
	if(idx+blockDim.x < count)
		v2 = d_data[idx+blockDim.x];
	// Perform first step of reduction.
	s_blockData[threadIdx.x] = opTraits::op(v1, v2);
	__syncthreads();

	T result = dev_ReduceFast<T, blockSize, opTraits>(s_blockData);

	// Write result to
	if(threadIdx.x == 0)
		d_results[blk] = result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, class traits> __global__ void kernel_SegmentedReducePrepare(T* d_data,
/// 	uint* d_owner, uint count, T identity, T* d_outCarryData, uint* d_outCarryOwner,
/// 	T* d_outData)
///
/// \brief	First phase of segmented reduction algorithm.
/// 		
/// 		Does per block segmented reduction in parallel on all thread blocks of size 256.
/// 		Since we cannot write the result of the last element to device memory (access
/// 		conflicts!), we pass it out as carry data. It is handled in the second phase. 
///
/// 		Algorithmic idea taken from: "Sparse Matrix-Vector Multiplication on CUDA",
/// 		http://www.nvidia.com/object/nvidia_research_pub_013.html. 
///
/// \note	Required shared memory per thread block: 4 * 257 + \c sizeof(T) * 257 bytes.
///
/// \author	Mathias Neumann
/// \date	23.04.2010
/// \see	kernel_SegmentedReduceFinal(), dev_SegReduceBlock()
///
/// \tparam	T			Type of elements.
/// \tparam	traits		Operator to use. See ReduceOperatorTraits.
///
/// \param [in]		d_data			The segmented data. It is assumed that the data of the same
/// 								segment is stored contiguosly. 
/// \param [in]		d_owner			Owner array that contains contiguous owner indices for the
/// 								data. d_owner[i] is the segment index of d_data[i].  
/// \param	count					Number of data elements and owner indices. 
/// \param	identity				Identity value for the operator. 
/// \param [out]	d_outCarryData	Carry out data for each block. 
/// \param [out]	d_outCarryOwner	Carry out owners for each block. 
/// \param [out]	d_outData		Reduction result after first phase. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class traits>
__global__ void kernel_SegmentedReducePrepare(T* d_data, uint* d_owner, uint count, T identity, 
											  T* d_outCarryData, uint* d_outCarryOwner, T* d_outData)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ uint s_owner[256 + 1];    
    __shared__ T s_data[256 + 1];

	// Use a dummy entry to allow writing the last result of each block to device memory,
	// even if the block is not yet done.
    if (threadIdx.x == 0)
	{
        s_owner[256] = (uint)-1;
        s_data[256] = identity;
    }
    __syncthreads();

    // Reduce for our 256 elements within our thread block. So read in data into shared
	// memory first.
	uint owner = (uint)-1;
	T data = identity;
	if(tid < count)
	{
		owner = d_owner[tid];
		data = d_data[tid];
	}
    s_owner[threadIdx.x] = owner;
    s_data[threadIdx.x] = data;
    __syncthreads();

    dev_SegReduceBlock<T, traits, 256>(s_owner, s_data, identity);

	if(threadIdx.x != blockDim.x - 1)
		if(s_owner[threadIdx.x] != s_owner[threadIdx.x + 1])
			d_outData[s_owner[threadIdx.x]] = traits::op(d_outData[s_owner[threadIdx.x]], s_data[threadIdx.x]);
    
	// Now write carry out values to device memory for next pass.
	if(tid < count && threadIdx.x == blockDim.x - 1)
    {
        // write the carry out values
        d_outCarryOwner[blockIdx.x] = s_owner[threadIdx.x];
        d_outCarryData[blockIdx.x] = s_data[threadIdx.x];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T, class traits> __global__ void kernel_SegmentedReduceFinal(T* d_data,
/// 	uint* d_owner, uint count, T identity, T* d_result)
///
/// \brief	Second and final phase of segmented reduction.
/// 		
/// 		Uses a single thread block to avoid access conflicts. To avoid large amounts of work
/// 		for a single thread block, use the first phase to reduce the data.
///
/// 		Algorithmic idea taken from: "Sparse Matrix-Vector Multiplication on CUDA",
/// 		http://www.nvidia.com/object/nvidia_research_pub_013.html. 
///
/// \note	Required shared memory per thread block: 4 * 257 + \c sizeof(T) * 257 bytes.
///
/// \author	Mathias Neumann
/// \date	23.04.2010
/// \see	kernel_SegmentedReducePrepare(), dev_SegReduceBlock()
///
/// \tparam	T			Type of elements.
/// \tparam	traits		Operator to use. See ReduceOperatorTraits.
///
/// \param [in]		d_data		The segmented data. It is assumed that the data of the same
/// 							segment is stored contiguosly. 
/// \param [in]		d_owner		Owner array that contains contiguous owner indices for the
/// 							data. d_owner[i] is the segment index of d_data[i]. 
/// \param	count				Number of data elements and owner indices.
/// \param	identity			Identity value for the operator.
/// \param [out]	d_result	Final reduction result. Contains one value for each segment.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class traits>
__global__ void kernel_SegmentedReduceFinal(T* d_data, uint* d_owner, uint count, T identity, 
											T* d_result)
{
	// WARNING: When T is float4, T is 16 bytes! Then we would get more than 8 kB shared memory,
	//          too much for my GTS 250...
	__shared__ uint s_owner[256 + 1];    
    __shared__ T s_data[256 + 1];    

    const uint end = count - (count & (256 - 1));

	// Use a dummy entry to allow writing the last result of each block to device memory,
	// even if the block is not yet done.
    if (threadIdx.x == 0)
	{
        s_owner[256] = (uint)-1;
        s_data[256] = identity;
    }
    __syncthreads();

    uint i = threadIdx.x;

	// Work on full blocks first.
    while(i < end)
	{
        s_owner[threadIdx.x] = d_owner[i];
        s_data[threadIdx.x] = d_data[i];
        __syncthreads();

        dev_SegReduceBlock<T, traits, 256>(s_owner, s_data, identity);

        if(s_owner[threadIdx.x] != s_owner[threadIdx.x + 1])
            d_result[s_owner[threadIdx.x]] = traits::op(d_result[s_owner[threadIdx.x]], s_data[threadIdx.x]);
        __syncthreads();

        i += 256; 
    }

	// Work on final elements (less than 256).
    if (end < count)
	{
        if (i < count)
		{
            s_owner[threadIdx.x] = d_owner[i];
            s_data[threadIdx.x] = d_data[i];
        } 
		else 
		{
            s_owner[threadIdx.x] = (uint)-1;
            s_data[threadIdx.x] = identity;
        }
        __syncthreads();
   
        dev_SegReduceBlock<T, traits, 256>(s_owner, s_data, identity);

        if (i < count)
            if(s_owner[threadIdx.x] != s_owner[threadIdx.x + 1])
				d_result[s_owner[threadIdx.x]] = traits::op(d_result[s_owner[threadIdx.x]], s_data[threadIdx.x]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __global__ void kernel_SetAtAddress(T* d_array, uint* d_address,
/// 	T* d_vals, uint countVals)
///
/// \brief	Moves data from device memory \a d_vals to device memory \a d_array using target
/// 		addresses specified in \a d_address.
///
///			\code d_array[d_address[i]] = d_vals[i] \endcode
/// 		
/// 		\warning	Heavy uncoalesced access possible. Depends on addresses. 
///
/// \author	Mathias Neumann
/// \date	18.02.2010
///
/// \tparam T		Element type of array and values.
///
/// \param [in,out]	d_array		The data array to manipulate.
/// \param [in]		d_address	Addresses of the values to manipulate. It is assumed that
///								all addresses \a d_address[i] are valid with respect to \a
///								d_array.
/// \param [in]		d_vals		The values to fill in.
/// \param	countVals			Number of values in \a d_address and \a d_vals. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void kernel_SetAtAddress(T* d_array, uint* d_address, T* d_vals, uint countVals)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < countVals)
	{
		uint addr = d_address[idx];
		d_array[addr] = d_vals[idx];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __global__ void kernel_SetFromAddress(T* d_array, uint* d_srcAddr,
/// 	T* d_vals, uint countTarget)
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
///
/// \param [in,out]	d_array		The data array to manipulate.
/// \param [in]		d_srcAddr	Addresses of the source values in \a d_vals. Use \c 0xffffffff to zero
/// 							the corresponding entry in \a d_array.
/// \param [in]		d_vals		The values to fill in.
/// \param	countTarget			Number of values in \a d_array and \a d_srcAddr.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void kernel_SetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals, uint countTarget)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < countTarget)
	{
		uint addr = d_srcAddr[idx];
		T value = {0};
		if(addr != 0xffffffff)
			value = d_vals[addr];
		d_array[idx] = value;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __global__ void kernel_SetConstAtAddress(T* d_array, uint* d_address,
/// 	T constant, uint countVals)
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
///
/// \tparam T		Element type of array and constant.
///
/// \param [in,out]	d_array		The data array to manipulate. 
/// \param [in]		d_address	Addresses of the values to manipulate. 
/// \param	constant			The constant to move to \a d_array. 
/// \param	countVals			Number of values in \a d_address. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void kernel_SetConstAtAddress(T* d_array, uint* d_address, T constant, uint countVals)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	// WARNING: Heavy uncoalesced access...
	if(idx < countVals)
	{
		uint addr = d_address[idx];
		d_array[addr] = constant;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <class T> __global__ void kernel_AlignCounts(uint* d_outAligned, uint* d_counts,
/// 	uint count)
///
/// \brief	Aligns the given count array by aligning all element counts.
/// 		
/// 		\code d_outAligned[i] = MNCUDA_ALIGN_NZERO(d_counts[i]) \endcode
///
/// \author	Mathias Neumann
/// \date	14.03.2010
/// \see	::mncudaAlignCounts()
///
/// \param [out]	d_outAligned	The aligned counts array. Has to be preallocated. 
/// \param [in]		d_counts		The count array. 
/// \param	count					Number of elements (counts). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AlignCounts(uint* d_outAligned, uint* d_counts, uint count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count)
	{
		uint old = d_counts[idx];
		d_outAligned[idx] = MNCUDA_ALIGN_NZERO(old);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_Normalize(float4* d_vectors, uint numVectors)
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
__global__ void kernel_Normalize(float4* d_vectors, uint numVectors)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numVectors)
	{
		float3 n = make_float3(d_vectors[tid]);
		float3 nNew; 
		if(n.x != 0.f || n.y != 0.f || n.z != 0.f)
			nNew = normalize(n); // No need to divide by count.
		d_vectors[tid] = make_float4(nNew);
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////



/// \brief	Tests segmented reduction result.
///
///			Compares the GPU result with a result computed on the CPU.
///
/// \see	kernel_SegmentedReducePrepare(), kernel_SegmentedReduceFinal()
template <class T, class traits>
void TestSegmentedReduce(T* d_data, uint* d_owner, uint count, MNCudaOP op, 
						 T identity, T* d_result, uint numSegments)
{
	T* h_data = new T[count];
	uint* h_owner = new uint[count];
	T* h_cudaResult = new T[numSegments];
	T* h_hostResult = new T[numSegments];

	// Get data from device memory.
	mncudaSafeCallNoSync(cudaMemcpy(h_data, d_data, count*sizeof(T), cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(h_owner, d_owner, count*sizeof(uint), cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(h_cudaResult, d_result, numSegments*sizeof(T), cudaMemcpyDeviceToHost));

	// Init host result with identity.
	for(uint i=0; i<numSegments; i++)
		h_hostResult[i] = identity;

	// Do (slow) segmented reduction on host.
	T currentRes = identity;
	for(uint i=0; i<count; i++)
	{
		currentRes = traits::op(currentRes, h_data[i]);

		if(i+1 == count || h_owner[i] != h_owner[i+1])
		{
			h_hostResult[h_owner[i]] = currentRes;
			currentRes = identity;
		}
	}

	// Compare results. 
	bool equal = true;
	for(uint i=0; i<numSegments; i++)
	{
		if(h_cudaResult[i] != h_hostResult[i])
		{
			printf("FAILED: %f vs. %f.\n", h_cudaResult[i], h_hostResult[i]);
			equal = false;
			break;
		}
	}

	if(equal)
		printf("SEGMENTED REDUCTION OK     (op: %d, count: %d, segments: %d)\n", op, count, numSegments);
	else
		printf("SEGMENTED REDUCTION FAILED (op: %d, count: %d, segments: %d)\n", op, count, numSegments);

	// Test if owner array was correct.
	uint curMax = h_owner[0];
	for(uint i=1; i<count; i++)
	{
		if(curMax > h_owner[i])
		{
			printf("SEGMENTED REDUCTION Illegal owner array!\n");
			break;
		}
		curMax = max(curMax, h_owner[i]);
	}

	delete [] h_data;
	delete [] h_owner;
	delete [] h_cudaResult;
	delete [] h_hostResult;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_InitIdentity() kernel call.
extern "C"
void KernelInitIdentity(uint* d_buffer, uint count)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_InitIdentity<<<gridSize, blockSize>>>(d_buffer, count);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_InitConstant() kernel call.
extern "C++"
template <class T>
void KernelInitConstant(T* d_buffer, uint count, T constant)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_InitConstant<<<gridSize, blockSize>>>(d_buffer, count, constant);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_AddIdentity() kernel call.
extern "C"
void KernelAddIdentity(uint* d_buffer, uint count)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_AddIdentity<<<gridSize, blockSize>>>(d_buffer, count);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ConstantOp() kernel call.
extern "C++"
template <MNCudaOP op, class T>
void KernelConstantOp(T* d_array, uint count, T constant)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_ConstantOp<op, T><<<gridSize, blockSize>>>(d_array, count, constant);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ScaleVectorArray() kernel call.
extern "C++"
template <class V, class S>
void KernelScaleVectorArray(V* d_vecArray, uint count, S scalar)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_ScaleVectorArray<V, S><<<gridSize, blockSize>>>(d_vecArray, count, scalar);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_AverageArray() kernel call.
extern "C++"
template <class V, class S>
void KernelAverageArray(V* d_array, uint count, S* d_counts)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_AverageArray<V, S><<<gridSize, blockSize>>>(d_array, count, d_counts);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_InverseBinary() kernel call.
extern "C"
void KernelInverseBinary(uint* d_buffer, uint count)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_InverseBinary<<<gridSize, blockSize>>>(d_buffer, count);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ArrayOp() kernel call.
extern "C++"
template <MNCudaOP op, class T>
void KernelArrayOp(T* d_target, T* d_other, uint count)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_ArrayOp<op, T><<<gridSize, blockSize>>>(d_target, d_other, count);
	MNCUDA_CHECKERROR;
}


/// \brief	Performs reduction for a given array of arbitrary length.
///
///			First, the number of elements is reduced using kernel_Reduce(). After that,
///			the per block results are read back to CPU memory. Finally the CPU performs the
///			reduction of the block results.
///
/// \see	kernel_Reduce()
extern "C++"
template <class T, MNCudaOP op>
cudaError_t KernelReduce(T& result, T* d_data, uint count, T identity)
{
	uint numBlocks = MNCUDA_DIVUP(count, 256);
	cudaError_t err;
	T* d_blockRes;

	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	err = pool.Request((void**)&d_blockRes, numBlocks*sizeof(T), "Temporary", 64 * sizeof(T)/4);
	if(err != cudaSuccess)
		return err;

	// Note that we use half the block size since we preload and -add two elements first.
	dim3 blockSize = dim3(256/2, 1, 1);
	dim3 gridSize = dim3(numBlocks, 1, 1);
	kernel_Reduce<T, 128, ReduceOperatorTraits<T, op>><<<gridSize, blockSize>>>(
		d_data, count, identity, d_blockRes);
	MNCUDA_CHECKERROR;

	// Read out block results to host...
	T* h_blockRes;
	err = pool.RequestHost((void**)&h_blockRes, numBlocks*sizeof(T));
	if(err != cudaSuccess)
		return err;

	err = cudaMemcpy(h_blockRes, d_blockRes, numBlocks*sizeof(T), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
		return err;

	// Reduce block results on CPU.
	result = identity;
	for(uint i=0; i<numBlocks; i++) 
		result = ReduceOperatorTraits<T, op>::op(result, h_blockRes[i]);

	err = pool.ReleaseHost(h_blockRes);
	if(err != cudaSuccess)
		return err;

	err = pool.Release(d_blockRes);
	if(err != cudaSuccess)
		return err;

	return cudaSuccess;
}

/// \brief	Performs segmented reduction for given data and owner array.
///
///			When the number of data elements is smaller than a chosen threshold, only the single thread block
///			kernel kernel_SegmentedReduceFinal() is called. Else the number of elements is reduced first by
///			calling kernel_SegmentedReducePrepare().
///
/// \see	kernel_SegmentedReducePrepare(), kernel_SegmentedReduceFinal()
extern "C++"
template <class T, MNCudaOP op>
void KernelSegmentedReduce(T* d_data, uint* d_owner, uint count, T identity, T* d_result, uint numSegments)
{
	if(count < 16*256)
	{
		kernel_SegmentedReduceFinal<T, ReduceOperatorTraits<T, op>><<<1, 256>>>(
			d_data, d_owner, count, identity, d_result);
		MNCUDA_CHECKERROR;
	}
	else
	{
		const uint blockSizePrepare = 256;
		const uint numBlocks = MNCUDA_DIVUP(count, blockSizePrepare);

		// Memory for carry out data from prepare phase.
		MNCudaMemory<uint> d_carryOwner(numBlocks);
		MNCudaMemory<T> d_carryData(numBlocks);

		// Phase 1: Per block computation to reduce work for single thread in last phase.
		kernel_SegmentedReducePrepare<T, ReduceOperatorTraits<T, op>><<<numBlocks, blockSizePrepare>>>(
			d_data, d_owner, count, identity, d_carryData, d_carryOwner, d_result);
		MNCUDA_CHECKERROR;

		// Phase 2: Finalize result.
		kernel_SegmentedReduceFinal<T, ReduceOperatorTraits<T, op>><<<1, 256>>>(
			d_carryData, d_carryOwner, numBlocks-1, identity, d_result);
		MNCUDA_CHECKERROR;
	}

	// Test the result... Currently I use this for standard types only.
#ifdef TEST_SEG_REDUCE
	if(sizeof(T) == 4)
		TestSegmentedReduce<T, ReduceOperatorTraits<T, op>>(d_data, d_owner, count, op, identity, d_result, numSegments);
#endif
}

/// Wraps kernel_SetAtAddress() kernel call.
extern "C++"
template <class T>
void KernelSetAtAddress(T* d_array, uint* d_address, T* d_vals, uint countVals)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(countVals, blockSize.x), 1, 1);

	kernel_SetAtAddress<<<gridSize, blockSize>>>(d_array, d_address, d_vals, countVals);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_SetFromAddress() kernel call.
extern "C++"
template <class T>
void KernelSetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals, uint countTarget)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(countTarget, blockSize.x), 1, 1);

	kernel_SetFromAddress<<<gridSize, blockSize>>>(d_array, d_srcAddr, d_vals, countTarget);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_SetConstAtAddress() kernel call.
extern "C++"
template <class T>
void KernelSetConstAtAddress(T* d_array, uint* d_address, T constant, uint countVals)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(countVals, blockSize.x), 1, 1);

	kernel_SetConstAtAddress<<<gridSize, blockSize>>>(d_array, d_address, constant, countVals);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_AlignCounts() kernel call.
extern "C"
void KernelAlignCounts(uint* d_outAligned, uint* d_counts, uint count)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_AlignCounts<<<gridSize, blockSize>>>(d_outAligned, d_counts, count);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_Normalize() kernel call.
extern "C"
void KernelNormalize(float4* d_vectors, uint numVectors)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numVectors, blockSize.x), 1, 1);

	kernel_Normalize<<<gridSize, blockSize>>>(d_vectors, numVectors);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef DOXYGEN_IGNORE

// Avoid linker errors by explicitly defining the used templates here. See
// http://www.parashift.com/c++-faq-lite/templates.html#faq-35.15
template void KernelInitConstant<float>(float* d_buffer, uint count, float constant);
template void KernelInitConstant<float4>(float4* d_buffer, uint count, float4 constant);
template void KernelInitConstant<uint>(uint* d_buffer, uint count, uint constant);

template void KernelConstantOp<MNCuda_ADD, float>(float* d_array, uint count, float constant);
template void KernelConstantOp<MNCuda_SUB, float>(float* d_array, uint count, float constant);
template void KernelConstantOp<MNCuda_MUL, float>(float* d_array, uint count, float constant);
template void KernelConstantOp<MNCuda_ADD, uint>(uint* d_array, uint count, uint constant);
template void KernelConstantOp<MNCuda_SUB, uint>(uint* d_array, uint count, uint constant);
template void KernelConstantOp<MNCuda_MUL, uint>(uint* d_array, uint count, uint constant);

template void KernelScaleVectorArray<float4, float>(float4* d_vecArray, uint count, float scalar);

template void KernelAverageArray<float4, float>(float4* d_array, uint count, float* d_counts);

template void KernelArrayOp<MNCuda_ADD, float>(float* d_target, float* d_other, uint count);
template void KernelArrayOp<MNCuda_SUB, float>(float* d_target, float* d_other, uint count);
template void KernelArrayOp<MNCuda_MUL, float>(float* d_target, float* d_other, uint count);
template void KernelArrayOp<MNCuda_DIV, float>(float* d_target, float* d_other, uint count);
template void KernelArrayOp<MNCuda_ADD, uint>(uint* d_target, uint* d_other, uint count);
template void KernelArrayOp<MNCuda_SUB, uint>(uint* d_target, uint* d_other, uint count);
template void KernelArrayOp<MNCuda_MUL, uint>(uint* d_target, uint* d_other, uint count);
template void KernelArrayOp<MNCuda_DIV, uint>(uint* d_target, uint* d_other, uint count);

template void KernelSegmentedReduce<float, MNCuda_ADD>(float* d_data, uint* d_owner, uint count, 
								  float identity, float* d_result, uint numSegments);
template void KernelSegmentedReduce<float, MNCuda_MIN>(float* d_data, uint* d_owner, uint count, 
								  float identity, float* d_result, uint numSegments);
template void KernelSegmentedReduce<float, MNCuda_MAX>(float* d_data, uint* d_owner, uint count, 
								  float identity, float* d_result, uint numSegments);
template void KernelSegmentedReduce<float4, MNCuda_ADD>(float4* d_data, uint* d_owner, uint count, 
								  float4 identity, float4* d_result, uint numSegments);
template void KernelSegmentedReduce<float4, MNCuda_MIN>(float4* d_data, uint* d_owner, uint count, 
								  float4 identity, float4* d_result, uint numSegments);
template void KernelSegmentedReduce<float4, MNCuda_MAX>(float4* d_data, uint* d_owner, uint count, 
								  float4 identity, float4* d_result, uint numSegments);
template void KernelSegmentedReduce<uint, MNCuda_ADD>(uint* d_data, uint* d_owner, uint count, 
								  uint identity, uint* d_result, uint numSegments);
template void KernelSegmentedReduce<uint, MNCuda_MIN>(uint* d_data, uint* d_owner, uint count, 
								  uint identity, uint* d_result, uint numSegments);
template void KernelSegmentedReduce<uint, MNCuda_MAX>(uint* d_data, uint* d_owner, uint count, 
								  uint identity, uint* d_result, uint numSegments);

// WARNING: reduce uses volatile T and therefore won't work with structs!
template cudaError_t KernelReduce<uint, MNCuda_ADD>(uint& result, uint* d_data, uint count, uint identity);
template cudaError_t KernelReduce<float, MNCuda_ADD>(float& result, float* d_data, uint count, float identity);
template cudaError_t KernelReduce<uint, MNCuda_MIN>(uint& result, uint* d_data, uint count, uint identity);
template cudaError_t KernelReduce<float, MNCuda_MIN>(float& result, float* d_data, uint count, float identity);
template cudaError_t KernelReduce<uint, MNCuda_MAX>(uint& result, uint* d_data, uint count, uint identity);
template cudaError_t KernelReduce<float, MNCuda_MAX>(float& result, float* d_data, uint count, float identity);

template void KernelSetAtAddress<uint>(uint* d_array, uint* d_address, uint* d_vals, uint countVals);
template void KernelSetAtAddress<float>(float* d_array, uint* d_address, float* d_vals, uint countVals);

template void KernelSetFromAddress<uint>(uint* d_array, uint* d_srcAddr, uint* d_vals, uint countTarget);
template void KernelSetFromAddress<unsigned long long>(unsigned long long* d_array, uint* d_srcAddr, unsigned long long* d_vals, uint countTarget);
template void KernelSetFromAddress<uint2>(uint2* d_array, uint* d_srcAddr, uint2* d_vals, uint countTarget);
template void KernelSetFromAddress<int>(int* d_array, uint* d_srcAddr, int* d_vals, uint countTarget);
template void KernelSetFromAddress<float>(float* d_array, uint* d_srcAddr, float* d_vals, uint countTarget);
template void KernelSetFromAddress<float2>(float2* d_array, uint* d_srcAddr, float2* d_vals, uint countTarget);
template void KernelSetFromAddress<float4>(float4* d_array, uint* d_srcAddr, float4* d_vals, uint countTarget);
template void KernelSetFromAddress<short2>(short2* d_array, uint* d_srcAddr, short2* d_vals, uint countTarget);

template void KernelSetConstAtAddress<uint>(uint* d_array, uint* d_address, uint constant, uint countVals);

#endif // DOXYGEN_IGNORE