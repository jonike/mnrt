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

#include "MNCudaUtil.h"
#include "MNCudaPrimitives.h"
#include "MNCudaMemPool.h"

extern "C"
void KernelInitIdentity(uint* d_buffer, uint count);
extern "C++"
template <class T>
void KernelInitConstant(T* d_buffer, uint count, T constant);
extern "C"
void KernelAddIdentity(uint* d_buffer, uint count);
extern "C++"
template <MNCudaOP op, class T>
void KernelConstantOp(T* d_array, uint count, T constant);
extern "C++"
template <class V, class S>
void KernelScaleVectorArray(V* d_vecArray, uint count, S scalar);
extern "C++"
template <class V, class S>
void KernelAverageArray(V* d_array, uint count, S* d_counts);
extern "C"
void KernelInverseBinary(uint* d_buffer, uint count);
extern "C++"
template <MNCudaOP op, class T>
void KernelArrayOp(T* d_target, T* d_other, uint count);
extern "C++"
template <class T, MNCudaOP op>
cudaError_t KernelReduce(T& result, T* d_data, uint count, T identity);
extern "C++"
template <class T, MNCudaOP op>
void KernelSegmentedReduce(T* d_data, uint* d_owner, uint count, T identity, T* d_result, uint numSegments);
extern "C++"
template <class T>
void KernelSetAtAddress(T* d_array, uint* d_address, T* d_vals, uint countVals);
extern "C++"
template <class T>
void KernelSetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals, uint countTarget);
extern "C++"
template <class T>
void KernelSetConstAtAddress(T* d_array, uint* d_address, T constant, uint countVals);
extern "C"
void KernelAlignCounts(uint* d_outAligned, uint* d_counts, uint count);
extern "C"
void KernelNormalize(float4* d_vectors, uint numVectors);

bool f_bEnableErrorChecks = false;

cudaError_t mncudaCheckError(bool bForce /*= true*/)
{
	if(bForce || f_bEnableErrorChecks)
		return cudaThreadSynchronize();
	else
		return cudaSuccess;
}

void mncudaEnableErrorChecks(bool bEnable /*= true*/)
{
	f_bEnableErrorChecks = bEnable;
}

bool mncudaIsErrorChecking()
{
	return f_bEnableErrorChecks;
}

uint mncudaGetMaxBlockSize(uint reqSharedMemPerThread, uint maxRegPerThread, 
						   bool useMultipleWarpSize/*= true*/)
{
	// Get device properties.
	int curDevice;
	cudaDeviceProp props;
	mncudaSafeCallNoSync(cudaGetDevice(&curDevice));
	mncudaSafeCallNoSync(cudaGetDeviceProperties(&props, curDevice));

	// Get maximum number of threads per block due to register requirement.
	uint maxForReg = props.maxThreadsPerBlock;
	if(maxRegPerThread > 0)
		maxForReg = props.regsPerBlock / maxRegPerThread;

	if(maxForReg == 0)
	{
		MNFatal("Current CUDA device cannot provide required register count.\n\n"
				"(required per thread: %d, maximum per block: %d)",
				maxRegPerThread, props.regsPerBlock);
		return 0;
	}
	else if(reqSharedMemPerThread > props.sharedMemPerBlock)
	{
		MNFatal("Current CUDA device cannot provide required shared memory.\n\n"
				"(required per thread: %d, maximum per block: %d)",
				reqSharedMemPerThread, props.sharedMemPerBlock);
		return 0;
	}
	else if(reqSharedMemPerThread <= 0)
	{
		return min(maxForReg, props.maxThreadsPerBlock);
	}
	else
	{
		// 16 bytes for blockIdx and stuff.
		// 256 bytes as maximum space for parameters.
		// See CUDA FAQ: http://forums.nvidia.com/index.php?showtopic=84440.
		uint reserved_bytes = 16 + 256;
		uint maxForShared = (props.sharedMemPerBlock - reserved_bytes) / reqSharedMemPerThread;
		uint maxSize = min(maxForShared, maxForReg);
		if(!useMultipleWarpSize)
			return maxSize;
		else
			return maxSize - (maxSize % props.warpSize);
	}
}

void mncudaInitIdentity(uint* d_buffer, uint count)
{
	MNAssert(d_buffer && count > 0);
	KernelInitIdentity(d_buffer, count);
}

template <class T>
void mncudaInitConstant(T* d_buffer, uint count, T constant)
{
	MNAssert(d_buffer && count > 0);
	KernelInitConstant(d_buffer, count, constant);
}

void mncudaAddIdentity(uint* d_buffer, uint count)
{
	MNAssert(d_buffer && count > 0);
	KernelAddIdentity(d_buffer, count);
}

template <MNCudaOP op, class T>
void mncudaConstantOp(T* d_array, uint count, T constant)
{
	MNAssert(d_array && count > 0);
	KernelConstantOp<op, T>(d_array, count, constant);
}

template <class V, class S>
void mncudaScaleVectorArray(V* d_vecArray, uint count, S scalar)
{
	MNAssert(d_vecArray && count > 0);
	KernelScaleVectorArray<V, S>(d_vecArray, count, scalar);
}

template <class V, class S>
void mncudaAverageArray(V* d_array, uint count, S* d_counts)
{
	MNAssert(d_array && count > 0 && d_counts);
	KernelAverageArray<V, S>(d_array, count, d_counts);
}

void mncudaInverseBinary(uint* d_buffer, uint count)
{
	MNAssert(d_buffer && count > 0);
	KernelInverseBinary(d_buffer, count);
}

template <MNCudaOP op, class T>
void mncudaArrayOp(T* d_target, T* d_other, uint count)
{
	MNAssert(d_target && d_other && count > 0);
	KernelArrayOp<op, T>(d_target, d_other, count);
}

template <class T>
uint mncudaResizeMNCudaMem(T** d_buffer, uint numOld, uint numRequest, uint slices/* = 1*/)
{
	MNAssert(d_buffer && numOld < numRequest && slices > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Ensure aligned access to get coalesced access in kernels.
	uint numNew = MNCUDA_ALIGN(numRequest);

	// First request the new buffer.
	T* d_newBuffer;
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_newBuffer, slices*numNew*sizeof(T), "Resize"));

	// Copy content of old buffers into new buffers. Slice by slice as there might be holes
	// at the end of a slice.
	for(uint s=0; s<slices; s++)
		mncudaSafeCallNoSync(cudaMemcpy(d_newBuffer + s*numNew, *d_buffer + s*numOld, 
			numOld*sizeof(T), cudaMemcpyDeviceToDevice));

	// Free old device memory.
	mncudaSafeCallNoSync(pool.Release(*d_buffer));

	// Assign new buffer.
	*d_buffer = d_newBuffer;

	return numNew;
}

template <class T>
cudaError_t mncudaReduce(T& result, T* d_data, uint count, MNCudaOP op, T identity)
{
	if(op == MNCuda_ADD)
		return KernelReduce<T, MNCuda_ADD>(result, d_data, count, identity);
	else if(op == MNCuda_MIN)
		return KernelReduce<T, MNCuda_MIN>(result, d_data, count, identity);
	else if(op == MNCuda_MAX)
		return KernelReduce<T, MNCuda_MAX>(result, d_data, count, identity);
	else
	{
		MNFatal("Illegal reduce operation.");
		return cudaErrorUnknown;
	}
}

template <class T>
cudaError_t mncudaSegmentedReduce(T* d_data, uint* d_owner, uint count, MNCudaOP op, T identity, 
								  T* d_result, uint numSegments)
{
	MNAssert(d_result && d_owner && d_result && count > 0 && numSegments > 0);
	cudaError_t err = cudaSuccess;

	// Catch pathetic case count=1.
	if(count == 1)
	{
		return cudaMemcpy(d_result, d_data, sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else
	{
		mncudaInitConstant(d_result, numSegments, identity);

		if(op == MNCuda_ADD)
			KernelSegmentedReduce<T, MNCuda_ADD>(d_data, d_owner, count, identity, d_result, numSegments);
		else if(op == MNCuda_MIN)
			KernelSegmentedReduce<T, MNCuda_MIN>(d_data, d_owner, count, identity, d_result, numSegments);
		else if(op == MNCuda_MAX)
			KernelSegmentedReduce<T, MNCuda_MAX>(d_data, d_owner, count, identity, d_result, numSegments);

		return err;
	}
}

template <class T>
void mncudaSetAtAddress(T* d_array, uint* d_address, T* d_vals, uint countVals)
{
	MNAssert(d_array && d_address && d_vals && countVals > 0);
	KernelSetAtAddress(d_array, d_address, d_vals, countVals);
}

template <class T>
void mncudaSetFromAddress(T* d_array, uint* d_srcAddr, T* d_vals, uint countTarget)
{
	MNAssert(d_array && d_srcAddr && d_vals && countTarget > 0);
	KernelSetFromAddress(d_array, d_srcAddr, d_vals, countTarget);
}

template <class T>
void mncudaSetConstAtAddress(T* d_array, uint* d_address, T constant, uint countVals)
{
	MNAssert(d_array && d_address && countVals > 0);
	KernelSetConstAtAddress(d_array, d_address, constant, countVals);
}

void mncudaAlignCounts(uint* d_outAligned, uint* d_counts, uint count)
{
	MNAssert(d_outAligned && d_counts && count > 0);
	KernelAlignCounts(d_outAligned, d_counts, count);
}

template <class T>
cudaError_t mncudaPrintArray(T* d_array, uint count, bool isFloat, const char* strArray/* = NULL*/)
{
	MNAssert(d_array && count > 0);
	cudaError_t err = cudaSuccess;

	// Move array to host.
	T* h_array = new T[count];
	err = cudaMemcpy(h_array, d_array, count*sizeof(T), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		SAFE_DELETE_ARRAY(h_array);
		return err;
	}

	if(strArray)
		printf("Device array %s:\n", strArray);
	else
		printf("Device array:\n");

	for(uint i=0; i<count; i++)
	{
		if(isFloat)
			printf("%.8f ", h_array[i]);
		else
			printf("%d ", h_array[i]);
	}
	printf("\n");

	SAFE_DELETE_ARRAY(h_array);

	return err;
}

template <class T>
void mncudaCompactInplace(T* d_data, uint* d_srcAddr, uint countOld, uint countNew)
{
	if(countNew == 0)
		return; // Nothing to do. Just leave d_data unchanged.

	// Move data into temp buffer.
	MNCudaMemory<T> d_tempBuf(countOld);
	mncudaSafeCallNoSync(cudaMemcpy(d_tempBuf, d_data, countOld*sizeof(T), cudaMemcpyDeviceToDevice));

	mncudaSetFromAddress(d_data, d_srcAddr, (T*)d_tempBuf, countNew);
}

uint mncudaGenCompactAddresses(uint* d_isValid, uint countOld, uint* d_outSrcAddr)
{
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	MNCudaMemory<uint> d_countNew(1);
	MNCudaMemory<uint> d_identity(countOld);

	// Compact indices array.
	mncudaInitIdentity(d_identity, countOld);
	cp.Compact(d_identity, d_isValid, countOld, 
		d_outSrcAddr, d_countNew);

	uint countNew = d_countNew.Read(0);

	return countNew;
}

void mncudaNormalize(float4* d_vectors, uint numVectors)
{
	MNAssert(d_vectors && numVectors > 0);
	KernelNormalize(d_vectors, numVectors);
}


// Avoid linker errors by explicitly defining the used templates here. See
// http://www.parashift.com/c++-faq-lite/templates.html#faq-35.15
template void mncudaInitConstant<uint>(uint* d_buffer, uint count, uint constant);
template void mncudaInitConstant<float>(float* d_buffer, uint count, float constant);

template void mncudaConstantOp<MNCuda_ADD, float>(float* d_array, uint count, float constant);
template void mncudaConstantOp<MNCuda_SUB, float>(float* d_array, uint count, float constant);
template void mncudaConstantOp<MNCuda_MUL, float>(float* d_array, uint count, float constant);
template void mncudaConstantOp<MNCuda_ADD, uint>(uint* d_array, uint count, uint constant);
template void mncudaConstantOp<MNCuda_SUB, uint>(uint* d_array, uint count, uint constant);
template void mncudaConstantOp<MNCuda_MUL, uint>(uint* d_array, uint count, uint constant);

template void mncudaScaleVectorArray<float4, float>(float4* d_vecArray, uint count, float scalar);

template void mncudaAverageArray<float4, float>(float4* d_array, uint count, float* d_counts);

template void mncudaArrayOp<MNCuda_ADD, float>(float* d_target, float* d_other, uint count);
template void mncudaArrayOp<MNCuda_SUB, float>(float* d_target, float* d_other, uint count);
template void mncudaArrayOp<MNCuda_MUL, float>(float* d_target, float* d_other, uint count);
template void mncudaArrayOp<MNCuda_DIV, float>(float* d_target, float* d_other, uint count);
template void mncudaArrayOp<MNCuda_ADD, uint>(uint* d_target, uint* d_other, uint count);
template void mncudaArrayOp<MNCuda_SUB, uint>(uint* d_target, uint* d_other, uint count);
template void mncudaArrayOp<MNCuda_MUL, uint>(uint* d_target, uint* d_other, uint count);
template void mncudaArrayOp<MNCuda_DIV, uint>(uint* d_target, uint* d_other, uint count);

template uint mncudaResizeMNCudaMem<uint>(uint** d_buffer, uint numOld, uint numNew, uint slices);
template uint mncudaResizeMNCudaMem<unsigned long long>(unsigned long long** d_buffer, uint numOld, uint numNew, uint slices);
template uint mncudaResizeMNCudaMem<uint2>(uint2** d_buffer, uint numOld, uint numNew, uint slices);
template uint mncudaResizeMNCudaMem<float>(float** d_buffer, uint numOld, uint numNew, uint slices);
template uint mncudaResizeMNCudaMem<float2>(float2** d_buffer, uint numOld, uint numNew, uint slices);
template uint mncudaResizeMNCudaMem<float4>(float4** d_buffer, uint numOld, uint numNew, uint slices);

template
cudaError_t mncudaSegmentedReduce<float>(float* d_data, uint* d_owner, uint count, MNCudaOP op, float identity, 
								  float* d_result, uint numSegments);
template
cudaError_t mncudaSegmentedReduce<float4>(float4* d_data, uint* d_owner, uint count, MNCudaOP op, float4 identity, 
								  float4* d_result, uint numSegments);
template
cudaError_t mncudaSegmentedReduce<uint>(uint* d_data, uint* d_owner, uint count, MNCudaOP op, uint identity, 
								  uint* d_result, uint numSegments);


template cudaError_t mncudaReduce<uint>(uint& result, uint* d_data, uint count, MNCudaOP op, uint identity);
template cudaError_t mncudaReduce<float>(float& result, float* d_data, uint count, MNCudaOP op, float identity);

template void mncudaSetAtAddress<uint>(uint* d_array, uint* d_address, uint* d_vals, uint countVals);
template void mncudaSetAtAddress<float>(float* d_array, uint* d_address, float* d_vals, uint countVals);

template void mncudaSetFromAddress<uint>(uint* d_array, uint* d_srcAddr, uint* d_vals, uint countTarget);
template void mncudaSetFromAddress<unsigned long long>(unsigned long long* d_array, uint* d_srcAddr, unsigned long long* d_vals, uint countTarget);
template void mncudaSetFromAddress<uint2>(uint2* d_array, uint* d_srcAddr, uint2* d_vals, uint countTarget);
template void mncudaSetFromAddress<int>(int* d_array, uint* d_srcAddr, int* d_vals, uint countTarget);
template void mncudaSetFromAddress<float>(float* d_array, uint* d_srcAddr, float* d_vals, uint countTarget);
template void mncudaSetFromAddress<float2>(float2* d_array, uint* d_srcAddr, float2* d_vals, uint countTarget);
template void mncudaSetFromAddress<float4>(float4* d_array, uint* d_srcAddr, float4* d_vals, uint countTarget);

template void mncudaSetConstAtAddress<uint>(uint* d_array, uint* d_address, uint constant, uint countVals);

template cudaError_t mncudaPrintArray<uint>(uint* d_array, uint count, bool isFloat, const char* strArray/* = NULL*/);
template cudaError_t mncudaPrintArray<float>(float* d_array, uint count, bool isFloat, const char* strArray/* = NULL*/);

template void mncudaCompactInplace<int>(int* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<uint>(uint* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<uint2>(uint2* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<float>(float* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<float2>(float2* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<float4>(float4* d_data, uint* d_srcAddr, uint countOld, uint countNew);
template void mncudaCompactInplace<short2>(short2* d_data, uint* d_srcAddr, uint countOld, uint countNew);