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
/// \file	GPU\kdtree.cu
///
/// \brief	Kernels for kd-tree construction.
/// \author	Mathias Neumann
/// \date	15.02.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "kd-tree/KDKernelDefs.h"
#include "MNCudaMT.h"
#include "MNCudaMemPool.h"
#include "MNCudaPrimitives.h"

#include "mncudautil_dev.h"

/// \brief	kd-tree traverse cost. Used to stop the splitting at some point where traversal would
///			lead to a higher cost than splitting a given node.
#define KD_COST_TRAVERSE	3.0f

/// CUDA device properties used to get maximum grid size.
cudaDeviceProp f_DevProps;

/// \brief	Empty space ratio.
///
///	\see	KDTreeGPU::SetEmptySpaceRatio()
__constant__ float c_emptySpaceRatio;
/// Maximum kNN query radius. Used for point kd-trees to evaluate VVH cost heuristic.
__constant__ float c_maxQueryRadius;
/// \brief	Maximum number of elements for small nodes.
///
/// \see	::KD_SMALLNODEMAX, KDTreeGPU::SetSmallNodeMax()
__constant__ uint c_smallNodeMax;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDNodeListAABB
///
/// \brief	Slim version of KDNodeList for node AABB related tasks.
///
///			To avoid parameter space overflows.
///
/// \author	Mathias Neumann
/// \date	23.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDNodeListAABB
{
	#ifdef __cplusplus
public:
	/// Initializes helper struct from given node list.
	void Initialize(const KDNodeList& src)
	{
		numNodes = src.numNodes;
		d_idxFirstElem = src.d_idxFirstElem;
		d_numElems = src.d_numElems;
		d_nodeLevel = src.d_nodeLevel;
		d_aabbMinTight = src.d_aabbMinTight;
		d_aabbMinInherit = src.d_aabbMinInherit;
		d_aabbMaxTight = src.d_aabbMaxTight;
		d_aabbMaxInherit = src.d_aabbMaxInherit;
		d_elemNodeAssoc = src.d_elemNodeAssoc;
	}
#endif // __cplusplus

	/// See KDNodeList::numNodes.
	uint numNodes;

	/// See KDNodeList::d_idxFirstElem.
	uint* d_idxFirstElem;
	/// See KDNodeList::d_idxFirstElem.
	uint* d_numElems;
	/// See KDNodeList::d_idxFirstElem.
	uint* d_nodeLevel;
	/// See KDNodeList::d_aabbMinTight.
	float4* d_aabbMinTight;
	/// See KDNodeList::d_aabbMaxTight.
	float4* d_aabbMaxTight;
	/// See KDNodeList::d_aabbMinInherit.
	float4* d_aabbMinInherit;
	/// See KDNodeList::d_aabbMaxInherit.
	float4* d_aabbMaxInherit;

	/// See KDNodeList::d_elemNodeAssoc.
	uint* d_elemNodeAssoc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDNodeListSmall
///
/// \brief	Slim version of KDNodeList for small node stage kernels.
///
///			To avoid parameter space overflows.
///
/// \author	Mathias Neumann
/// \date	March 2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDNodeListSmall
{
	/// See KDNodeList::numNodes.
	uint numNodes;

	/// See KDNodeList::d_numElems.
	uint* d_numElems;
	/// See KDNodeList::d_nodeLevel.
	uint* d_nodeLevel;
	/// See KDNodeList::d_aabbMinTight.
	float4 *d_aabbMinTight;
	/// See KDNodeList::d_aabbMaxTight.
	float4 *d_aabbMaxTight;

	/// See KDNodeList::d_splitAxis.
	uint* d_splitAxis;
	/// See KDNodeList::d_splitPos.
	float* d_splitPos;

	/// See KDNodeList::d_idxSmallRoot.
	uint* d_idxSmallRoot; 
	/// See KDNodeList::d_elemMask.
	ElementMask* d_elemMask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDNodeListENA
///
/// \brief	Slim version of KDNodeList for ENA generation in small node stage.
///
///			To avoid parameter space overflows.
///
/// \author	Mathias Neumann
/// \date	March 2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDNodeListENA
{
	/// See KDNodeList::numNodes.
	uint numNodes;

	/// See KDNodeList::d_numElems.
	uint* d_numElems;
	/// See KDNodeList::d_idxFirstElem.
	uint* d_idxFirstElem;

	/// See KDNodeList::d_idxSmallRoot.
	uint* d_idxSmallRoot; 
	/// See KDNodeList::d_elemMask.
	ElementMask* d_elemMask;

	/// See KDNodeList::d_elemNodeAssoc.
	uint* d_elemNodeAssoc;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_CreateChild(KDNodeListAABB& lstNext, uint idxNew,
/// 	float aabbMin[3], float aabbMax[3], uint nodeLevel)
///
/// \brief	Simple helper that creates a new node in the given node list.
/// 		
/// 		Sets inherited node bounds and node level, but nothing else. 
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
/// \param [in,out]	lstNext	Target node list. 
/// \param	idxNew			Node index to use in target node list. 
/// \param	aabbMin			Inherited AABB minimum to set. 
/// \param	aabbMax			Inherited AABB maximum to set. 
/// \param	nodeLevel		The node level for the new node. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_CreateChild(KDNodeListAABB& lstNext, uint idxNew, 
									   float aabbMin[3], float aabbMax[3], uint nodeLevel)
{
	lstNext.d_nodeLevel[idxNew] = nodeLevel;
	lstNext.d_aabbMinInherit[idxNew] = make_float4(aabbMin[0], aabbMin[1], aabbMin[2], 0.f);
	lstNext.d_aabbMaxInherit[idxNew] = make_float4(aabbMax[0], aabbMax[1], aabbMax[2], 0.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_CreateEmptyLeaf(KDFinalNodeList& lstFinal, uint idxNew,
/// 	float aabbMin[3], float aabbMax[3], uint nodeLevel)
///
/// \brief	Creates an empty leaf node in the final node list.
/// 		
/// 		Sets all required information for the empty node, that is node level, number of
/// 		elements, child addresses (both 0) and node bounding box. 
///
/// \author	Mathias Neumann
/// \date	23.08.2010
/// \see	kernel_EmptySpaceCutting()
///
/// \param [in,out]	lstFinal	The final node list to update. 
/// \param	idxNew				Node index to use for empty node. 
/// \param	aabbMin				Node AABB minimum to write. 
/// \param	aabbMax				Node AABB maximum to write. 
/// \param	nodeLevel			The node level to write. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_CreateEmptyLeaf(KDFinalNodeList& lstFinal, uint idxNew, 
										   float aabbMin[3], float aabbMax[3], uint nodeLevel)
{
	// d_idxFirstElem can stay undefined as we have no elements.
	lstFinal.d_numElems[idxNew] = 0;
	lstFinal.d_nodeLevel[idxNew] = nodeLevel;
	// d_splitAxis, d_splitPos are undefined for leafs.
	lstFinal.d_childLeft[idxNew] = 0;
	lstFinal.d_childRight[idxNew] = 0;
	lstFinal.d_aabbMin[idxNew] = make_float4(aabbMin[0], aabbMin[1], aabbMin[2], 0.f);
	lstFinal.d_aabbMax[idxNew] = make_float4(aabbMax[0], aabbMax[1], aabbMax[2], 0.f);
	// d_elemNodeAssoc: no changes required.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_CreateFinalNodeCopy(KDFinalNodeList& lstFinal, uint idxOld,
/// 	uint idxNew, float aabbMin[3], float aabbMax[3], uint nodeLevel)
///
/// \brief	Copies a final node list node.
/// 		
/// 		Used for empty space cutting to generate a copy of the actual node. This copy
/// 		represents the new, non-empty node. The empty node is generated using
/// 		dev_CreateEmptyLeaf(). 
///
/// \author	Mathias Neumann
/// \date	23.08.2010 
///	\see	kernel_EmptySpaceCutting()
///
/// \param [in,out]	lstFinal	The final node list to update. 
/// \param	idxOld				Source node index. 
/// \param	idxNew				Target node index for copied node. 
/// \param	aabbMin				Node AABB minimum to write. 
/// \param	aabbMax				Node AABB maximum to write. 
/// \param	nodeLevel			The node level to write. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_CreateFinalNodeCopy(KDFinalNodeList& lstFinal, uint idxOld, uint idxNew,
											   float aabbMin[3], float aabbMax[3], uint nodeLevel)
{
	// Both nodes use the same elements.
	lstFinal.d_idxFirstElem[idxNew] = lstFinal.d_idxFirstElem[idxOld];
	lstFinal.d_numElems[idxNew] = lstFinal.d_numElems[idxOld];
	lstFinal.d_nodeLevel[idxNew] = nodeLevel;
	// d_splitAxis, d_splitPos, d_childLeft, d_childRight are not yet known.
	lstFinal.d_aabbMin[idxNew] = make_float4(aabbMin[0], aabbMin[1], aabbMin[2], 0.f);
	lstFinal.d_aabbMax[idxNew] = make_float4(aabbMax[0], aabbMax[1], aabbMax[2], 0.f);
	// d_elemNodeAssoc: no changes required.
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <uint numElementPoints> __global__ void kernel_GenChunkAABB(KDNodeList lst,
/// 	KDChunkList lstChunks)
///
/// \brief	Generates AABBs for chunks using parallel reduction.
/// 		
/// 		Chunk AABBs are generated by performing parallel reductions on the element AABBs
///			given in the node list.
///
/// \note	Required shared memory per thread block of size N: 8 * N bytes.
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lst			The node list. 
/// \param	lstChunks	Chunk list for given node list. Will be enriched with AABBs. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <uint numElementPoints>
__global__ void kernel_GenChunkAABB(KDNodeList lst, KDChunkList lstChunks)
{
	uint chk = MNCUDA_GRID2DINDEX;

	__shared__ uint s_numElems;
	__shared__ uint s_idxFirstElem;
	if(threadIdx.x == 0)
	{
		s_numElems = lstChunks.d_numElems[chk];
		s_idxFirstElem = lstChunks.d_idxFirstElem[chk];
	}
	__syncthreads();

	// Copy values into shared memory.
	__shared__ float smem[KD_CHUNKSIZE];
	float3 aabbMin, aabbMax;

	// Manual unrolling since automatic did not work.
	float v1, v2;

	if(numElementPoints == 1)
	{
		// Use second shared buffer to avoid rereading.
		__shared__ float smem2[KD_CHUNKSIZE];

		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].x;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].x;
		smem[threadIdx.x] = fminf(v1, v2);
		if(threadIdx.x >= s_numElems)
			v1 = -MN_INFINITY;
		if(threadIdx.x+blockDim.x >= s_numElems)
			v2 = -MN_INFINITY;
		smem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMin.x = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);
		aabbMax.x = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem2);

		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].y;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].y;
		smem[threadIdx.x] = fminf(v1, v2);
		if(threadIdx.x >= s_numElems)
			v1 = -MN_INFINITY;
		if(threadIdx.x+blockDim.x >= s_numElems)
			v2 = -MN_INFINITY;
		smem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMin.y = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);
		aabbMax.y = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem2);

		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].z;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].z;
		smem[threadIdx.x] = fminf(v1, v2);
		if(threadIdx.x >= s_numElems)
			v1 = -MN_INFINITY;
		if(threadIdx.x+blockDim.x >= s_numElems)
			v2 = -MN_INFINITY;
		smem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMin.z = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);
		aabbMax.z = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem2);
	}
	else // numElementPoints = 2
	{
		// First element point, the minimum.
		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].x;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].x;
		smem[threadIdx.x] = fminf(v1, v2);
		__syncthreads();
		aabbMin.x = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);

		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].y;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].y;
		smem[threadIdx.x] = fminf(v1, v2);
		__syncthreads();
		aabbMin.y = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);

		v1 = MN_INFINITY; v2 = MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x].z;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint1[s_idxFirstElem + threadIdx.x+blockDim.x].z;
		smem[threadIdx.x] = fminf(v1, v2);
		__syncthreads();
		aabbMin.z = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MIN>>(smem);

		// Second element point, the maximum.
		v1 = -MN_INFINITY; v2 = -MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x].x;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x+blockDim.x].x;
		smem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMax.x = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem);

		v1 = -MN_INFINITY; v2 = -MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x].y;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x+blockDim.x].y;
		smem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMax.y = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem);

		v1 = -MN_INFINITY; v2 = -MN_INFINITY;
		if(threadIdx.x < s_numElems)
			v1 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x].z;
		if(threadIdx.x+blockDim.x < s_numElems)
			v2 = lst.d_elemPoint2[s_idxFirstElem + threadIdx.x+blockDim.x].z;
		smem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabbMax.z = dev_ReduceFast<float, KD_CHUNKSIZE/2, ReduceOperatorTraits<float, MNCuda_MAX>>(smem);
	}

	if(threadIdx.x == 0)
	{
		lstChunks.d_aabbMin[chk] = make_float4(aabbMin);
		lstChunks.d_aabbMax[chk] = make_float4(aabbMax);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_CanCutOffEmptySpace(KDNodeList lstActive, uint axis, bool bMax,
/// 	uint* d_outCanCutOff)
///
/// \brief	Checks if we can cut off empty space on a given fixed node side (axis, min/max).
/// 		
/// 		Empty space can be cut off when there is a given ratio ::c_emptySpaceRatio of empty
/// 		space. This kernel checks this for each node. 
///
/// \author	Mathias Neumann
/// \date	22.08.2010 
///	\see	kernel_EmptySpaceCutting()
///
/// \param	lstActive				The active node list. 
/// \param	axis					The axis to check. 
/// \param	bMax					Whether to check maximum or minimum sides. 
/// \param [out]	d_outCanCutOff	Binary 0/1 array. Contains 1 for nodes where empty space can
/// 								be cut off. This can be used to generate cut result (empty
/// 								node and remaining node). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_CanCutOffEmptySpace(KDNodeList lstActive, uint axis, bool bMax,
										   uint* d_outCanCutOff)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < lstActive.numNodes)
	{
		// Check for empty space on given side.
		float4 aabbMinInherit = lstActive.d_aabbMinInherit[idx];
		float4 aabbMaxInherit = lstActive.d_aabbMaxInherit[idx];
		float4 aabbMinTight =  lstActive.d_aabbMinTight[idx];
		float4 aabbMaxTight =  lstActive.d_aabbMaxTight[idx];
		float inheritMin = ((float*)&aabbMinInherit)[axis];
		float inheritMax = ((float*)&aabbMaxInherit)[axis];
		float tightMin = ((float*)&aabbMinTight)[axis];
		float tightMax = ((float*)&aabbMaxTight)[axis];
			
		// Get total.
		float total = inheritMax - inheritMin;
		float emptySpaceRatio = c_emptySpaceRatio;

		uint canCutOff = 0;
		if(!bMax)
		{
			// Minimum check.
			float empty = tightMin - inheritMin;
			if(empty > total*emptySpaceRatio) // Avoid division!
				canCutOff = 1;
		}
		else
		{
			// Maximum check.
			float empty = inheritMax - tightMax;
			if(empty > total*emptySpaceRatio) // Avoid division!
				canCutOff = 1;
		}

		d_outCanCutOff[idx] = canCutOff;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_EmptySpaceCutting(KDNodeList lstActive, KDFinalNodeList lstFinal,
/// 	uint axis, bool bMax, uint* d_canCutOff, uint* d_cutOffsets, uint numCuts,
/// 	uint* d_ioFinalListIndex)
///
/// \brief	Performs empty space cutting. 
///
///			For all nodes of the active list, for which empty space cutting on the given side
///			of their AABBs can be performed, the empty space is cut off. This is done by
///			generating new empty and non-empty nodes and inserting them into the final node list.
///			Furthermore the active list is updated to contain the new non-empty nodes.
///
/// \author	Mathias Neumann
/// \date	23.08.2010
///
/// \param	lstActive					The active node list. Contains the nodes that are to be
/// 									subdivided. When empty space is cut off for some node, its
///										AABB and node level are updated accordingly.
/// \param	lstFinal					The final node list. Will be updated with the generated
///										empty and non-empty nodes.
/// \param	axis						The axis to check. 
/// \param	bMax						Whether to check maximum or minimum sides. 
/// \param [in]		d_canCutOff			Binary 0/1 array. Contains 1 for nodes where empty space
/// 									can be cut off. Generated by
/// 									kernel_CanCutOffEmptySpace(). 
/// \param [in]		d_cutOffsets		Cut offsets. This should be the result of a scan of \a
/// 									d_canCutOff. It will indicate the offset to use for
/// 									writing the generated empty and non-empty child nodes
/// 									into the final node list. 
/// \param	numCuts						Number of cuts. Can be obtained by reduction on \a
/// 									d_canCutOff. 
/// \param [in,out]	d_ioFinalListIndex	Will contain updated final node list indices for the
///										current active list nodes. That is, the generated non-empty
///										node for the i-th active list node can be found at the index
///										\a d_ioFinalListIndex[i].
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_EmptySpaceCutting(KDNodeList lstActive, KDFinalNodeList lstFinal, uint axis, bool bMax,
										 uint* d_canCutOff, uint* d_cutOffsets, uint numCuts,
										 uint* d_ioFinalListIndex)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < lstActive.numNodes && d_canCutOff[idx])
	{
		float3 aabbMinInherit = make_float3(lstActive.d_aabbMinInherit[idx]);
		float3 aabbMaxInherit = make_float3(lstActive.d_aabbMaxInherit[idx]);
		float3 aabbMinTight = make_float3(lstActive.d_aabbMinTight[idx]);
		float3 aabbMaxTight = make_float3(lstActive.d_aabbMaxTight[idx]);

		float aabbMinChild[3] = {((float*)&aabbMinInherit)[0], ((float*)&aabbMinInherit)[1], ((float*)&aabbMinInherit)[2]};
		float aabbMaxChild[3] = {((float*)&aabbMaxInherit)[0], ((float*)&aabbMaxInherit)[1], ((float*)&aabbMaxInherit)[2]};
		
		uint nodeLevelParent = lstActive.d_nodeLevel[idx];

		// Compute indices for left and right node in final node list.
		uint cutOffset = d_cutOffsets[idx];
		uint idxParent = d_ioFinalListIndex[idx];
		uint idxLeft  = lstFinal.numNodes + cutOffset;
		uint idxRight = lstFinal.numNodes + numCuts + cutOffset;
		float splitPos = (bMax ? ((float*)&aabbMaxTight)[axis] : ((float*)&aabbMinTight)[axis]);

		if(!bMax)
		{
			// Below (left) is the empty node.
			aabbMaxChild[axis] = splitPos;
			dev_CreateEmptyLeaf(lstFinal, idxLeft, aabbMinChild, aabbMaxChild, nodeLevelParent+1);

			// Above (right) is the tighter node.
			aabbMinChild[axis] = aabbMaxChild[axis];
			aabbMaxChild[axis] = ((float*)&aabbMaxInherit)[axis];
			dev_CreateFinalNodeCopy(lstFinal, idxParent, idxRight, aabbMinChild, aabbMaxChild, nodeLevelParent+1);

			// Update active list node to describe the above node. Change inherited to be tighter.
			lstActive.d_aabbMinInherit[idx] = make_float4(aabbMinChild[0], aabbMinChild[1], aabbMinChild[2], 0.f);
			lstActive.d_aabbMaxInherit[idx] = make_float4(aabbMaxChild[0], aabbMaxChild[1], aabbMaxChild[2], 0.f);
			lstActive.d_nodeLevel[idx] = nodeLevelParent+1;
		}
		else
		{
			// Below (left) is the tighter node.
			aabbMaxChild[axis] = splitPos;
			dev_CreateFinalNodeCopy(lstFinal, idxParent, idxLeft, aabbMinChild, aabbMaxChild, nodeLevelParent+1);

			// Update active list node to describe the above node. Change inherited to be tighter.
			lstActive.d_aabbMinInherit[idx] = make_float4(aabbMinChild[0], aabbMinChild[1], aabbMinChild[2], 0.f);
			lstActive.d_aabbMaxInherit[idx] = make_float4(aabbMaxChild[0], aabbMaxChild[1], aabbMaxChild[2], 0.f);
			lstActive.d_nodeLevel[idx] = nodeLevelParent+1;

			// Above (right) is the empty node.
			aabbMinChild[axis] = aabbMaxChild[axis];
			aabbMaxChild[axis] = ((float*)&aabbMaxInherit)[axis];
			dev_CreateEmptyLeaf(lstFinal, idxRight, aabbMinChild, aabbMaxChild, nodeLevelParent+1);
		}

		// Write split information to original node in final node list.
		lstFinal.d_splitAxis[idxParent] = axis;
		lstFinal.d_splitPos[idxParent] = splitPos;
		lstFinal.d_childLeft[idxParent] = idxLeft;
		lstFinal.d_childRight[idxParent] = idxRight;

		// Update final list index to point to the tighter node.
		d_ioFinalListIndex[idx] = (bMax ? idxLeft : idxRight);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SplitLargeNodes(KDNodeList lstActive, KDNodeListAABB lstNext)
///
/// \brief	Splits large nodes in active list into smaller nodes.
/// 		
/// 		The resulting nodes are put in the next list, starting from index 0. Left (below)
/// 		nodes are written at the same indices as in active list. Right (above) nodes are
/// 		offsetted by the number of active list nodes. 
///
/// \author	Mathias Neumann
/// \date	17.02.2010
///
/// \param	lstActive	The active list. 
/// \param	lstNext		The next list. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SplitLargeNodes(KDNodeList lstActive, KDNodeListAABB lstNext)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < lstActive.numNodes)
	{
		// Empty space cutting was performed. Therefore our inherited bounds can be
		// used directly as basis.
		float3 aabbMinInherit = make_float3(lstActive.d_aabbMinInherit[idx]);
		float3 aabbMaxInherit = make_float3(lstActive.d_aabbMaxInherit[idx]);

		// Find longest axis of new bounds.
		uint longest = 0;
		if( aabbMaxInherit.y-aabbMinInherit.y > aabbMaxInherit.x-aabbMinInherit.x && 
			aabbMaxInherit.y-aabbMinInherit.y > aabbMaxInherit.z-aabbMinInherit.z )
			longest = 1;
		else if( aabbMaxInherit.z-aabbMinInherit.z > aabbMaxInherit.x-aabbMinInherit.x && 
				 aabbMaxInherit.z-aabbMinInherit.z > aabbMaxInherit.y-aabbMinInherit.y )
			longest = 2;

		// Split position.
		float splitPos = ((float*)&aabbMinInherit)[longest] + 
			.5f * (((float*)&aabbMaxInherit)[longest]-((float*)&aabbMinInherit)[longest]);

		// Store split information.
		lstActive.d_splitAxis[idx] = longest;
		lstActive.d_splitPos[idx] = splitPos;

		uint oldLevel = lstActive.d_nodeLevel[idx];

		// Add the two children for spatial median split.
		float aabbMinChild[3] = {aabbMinInherit.x, aabbMinInherit.y, aabbMinInherit.z};
		float aabbMaxChild[3] = {aabbMaxInherit.x, aabbMaxInherit.y, aabbMaxInherit.z};

		// Below node.
		uint idxWrite = idx;
		aabbMaxChild[longest] = splitPos;
		dev_CreateChild(lstNext, lstNext.numNodes + idxWrite, aabbMinChild, aabbMaxChild, oldLevel + 1);
		aabbMaxChild[longest] = ((float*)&aabbMaxInherit)[longest];

		lstActive.d_childLeft[idx] = idxWrite;
		// Set first index to same as parent node.
		lstNext.d_idxFirstElem[idxWrite] = lstActive.d_idxFirstElem[idx];

		// Above node.
		idxWrite = lstActive.numNodes + idx;
		aabbMinChild[longest] = splitPos;
		dev_CreateChild(lstNext, lstNext.numNodes + idxWrite, aabbMinChild, aabbMaxChild, oldLevel + 1);

		lstActive.d_childRight[idx] = idxWrite;
		// Set first index to offsetted parent node index.
		lstNext.d_idxFirstElem[idxWrite] = lstActive.nextFreePos + lstActive.d_idxFirstElem[idx];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_UpdateFinalListChildInfo(KDNodeList lstActive,
/// 	KDFinalNodeList lstFinal, uint* d_finalListIndex)
///
/// \brief	Updates final node list child information for allready added active list nodes.
/// 		
/// 		As a side-effect of empty space cutting, the active list nodes are already added to
/// 		the final node list. However, their child addresses and split axis/position are still
/// 		invalid. This kernel updates this information by using the data kernels wrote to the
/// 		active list \em after adding the nodes to the final list, i.e. it synchronizes the
/// 		active list with the final node list. 
///
/// \author	Mathias Neumann
/// \date	23.08.2010
///
/// \param	lstActive					The active node list. Contains updated child information. 
/// \param	lstFinal					The final node list that should be synchronized with the
/// 									active list. 
/// \param [in]		d_finalListIndex	Indices of the active list nodes in the final node list.
/// 									See kernel_EmptySpaceCutting(). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_UpdateFinalListChildInfo(KDNodeList lstActive, KDFinalNodeList lstFinal,
												uint* d_finalListIndex)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < lstActive.numNodes)
	{
		// Read final node list index for active list node.
		uint idxFinal = d_finalListIndex[idx];

		// Split information.
		lstFinal.d_splitAxis[idxFinal] = lstActive.d_splitAxis[idx];
		lstFinal.d_splitPos[idxFinal] = lstActive.d_splitPos[idx];

		// Child information. Note that we have to offset the indices by the number of
		// final nodes as they are currently relative to offset zero.
		lstFinal.d_childLeft[idxFinal] = lstFinal.numNodes + lstActive.d_childLeft[idx];
		lstFinal.d_childRight[idxFinal] = lstFinal.numNodes + lstActive.d_childRight[idx];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <uint numElementPoints> __global__ void kernel_MarkLeftRightElements(KDNodeList lstActive,
/// 	KDChunkList lstChunks, float* d_randoms, uint* d_outValid)
///
/// \brief	Marks left and right elements in next list ENA.
/// 		
/// 		It is assumed that active list's elements were duplicated in the following way: The
/// 		ENA was copied to the first \c lstActive.nextFreePos elements and to the second \c
/// 		lstActive.nextFreePos elements. 
///
/// \author	Mathias Neumann
/// \date	18.02.2010
///
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstActive			The active list. 
/// \param	lstChunks			The chunk list constructed for the active list. 
/// \param [in]		d_randoms	Uniform random numbers used to avoid endless split loops if
/// 							several elements lie within a splitting plane. There should be a
/// 							random number for each thread, i.e. for each element processed. 
/// \param [out]	d_outValid	Binary 0/1 array ordered as the ENA of the next list, i.e. the
/// 							left child valid flags are in the first half and the right child
/// 							valid flags are in the right half respectively. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <uint numElementPoints>
__global__ void kernel_MarkLeftRightElements(KDNodeList lstActive, KDChunkList lstChunks, 
											 float* d_randoms, uint* d_outValid)
{
	uint chk = MNCUDA_GRID2DINDEX;
	uint idx = threadIdx.x;

	__shared__ uint s_numElemsChunk;
	__shared__ uint s_idxNode;
	__shared__ uint s_idxFirstElem;
	__shared__ uint s_splitAxis;
	__shared__ float s_splitPos;
	if(threadIdx.x == 0)
	{
		s_numElemsChunk = lstChunks.d_numElems[chk];
		s_idxNode = lstChunks.d_idxNode[chk];
		s_idxFirstElem = lstChunks.d_idxFirstElem[chk];

		s_splitAxis = lstActive.d_splitAxis[s_idxNode];
		s_splitPos = lstActive.d_splitPos[s_idxNode];
	}
	__syncthreads();

	if(idx < s_numElemsChunk)
	{
		uint idxTNA = s_idxFirstElem + idx;
		uint tid = blockDim.x * MNCUDA_GRID2DINDEX + threadIdx.x;

		bool isLeft = false, isRight = false;
		if(numElementPoints == 2) // COMPILE TIME
		{
			// Get bounds.
			float boundsMin = ((float*)&lstActive.d_elemPoint1[idxTNA])[s_splitAxis];
			float boundsMax = ((float*)&lstActive.d_elemPoint2[idxTNA])[s_splitAxis];

			// Check on which sides the triangle is. It might be on both sides!
			if(d_randoms[tid] < 0.5f)
			{
				isLeft = boundsMin < s_splitPos || (boundsMin == s_splitPos && boundsMin == boundsMax);
				isRight = s_splitPos < boundsMax;
			}
			else
			{
				isLeft = boundsMin < s_splitPos;
				isRight = s_splitPos < boundsMax || (boundsMin == s_splitPos && boundsMin == boundsMax);
			}
		}
		else // Must be 1 point.
		{
			float value = ((float*)&lstActive.d_elemPoint1[idxTNA])[s_splitAxis];
			// Cannot use the same criterion (i.e. < and <= or <= and <) for all points.
			// Else we would have the special case where all points lie in the splitting plane
			// and therefore all points land on a single side. This would result in an endless
			// loop in large node stage!
			if(d_randoms[tid] < 0.5f)
			{
				isLeft = value < s_splitPos;
				isRight = s_splitPos <= value;
			}
			else
			{
				isLeft = value <= s_splitPos;
				isRight = s_splitPos < value; 
			}
		}

		// Left
		d_outValid[s_idxFirstElem + idx] = ((isLeft) ? 1 : 0);

		// Right
		d_outValid[lstActive.nextFreePos + s_idxFirstElem + idx] = ((isRight) ? 1 : 0);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MarkSmallNodes(KDNodeList lstNext, uint* d_finalListIndex,
/// 	uint* d_isSmall, uint* d_smallRootParent)
///
/// \brief	Marks all small nodes in the next node list.
/// 		
/// 		Furthermore it prepares a small root parent array (see below). 
///
/// \author	Mathias Neumann
/// \date	18.02.2010
///
/// \param	lstNext						The next list. 
/// \param [in]		d_finalListIndex	The indices of the active list nodes in the final node
/// 									list. See kernel_EmptySpaceCutting(). 
/// \param [out]	d_isSmall			Binary 0/1 array. Will contain 1 for small nodes, 0 for
/// 									large nodes. 
/// \param [out]	d_smallRootParent	Small root parent array. Will contain the index of the
/// 									corresponding parent node in the final node list. This is
/// 									required to keep track of where the parent nodes for a
/// 									given small node resides. In the MSB there is a 0 for
/// 									left child, else 1 for right child. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MarkSmallNodes(KDNodeList lstNext, uint* d_finalListIndex, uint* d_isSmall, 
									  uint* d_smallRootParent)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < lstNext.numNodes)
	{
		uint numNodesParent = lstNext.numNodes >> 1;

		int isSmall = ((lstNext.d_numElems[idx] <= c_smallNodeMax) ? 1 : 0);
		d_isSmall[idx] = isSmall;

		uint idxActive = idx;
		if(idxActive >= numNodesParent)
			idxActive -= numNodesParent;

		uint smpValue = d_finalListIndex[idxActive];

		// Set MSB to 1 for right nodes.
		if(idx < numNodesParent)
			smpValue &= 0x7FFFFFFF;
		else
			smpValue |= 0x80000000;
		
		d_smallRootParent[idx] = smpValue;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MarkElemsByNodeSize(KDChunkList lstChunks, uint* d_numElemsNext,
/// 	uint* d_outIsSmallElem, uint* d_outIsLargeElem)
///
/// \brief	Marks all node elements in the next list's ENA according to node size.
///
///			Generated marks can be used to separate small and large node's elements.
///
/// \author	Mathias Neumann
/// \date	18.02.2010
///
/// \param	lstChunks					The chunk list. Used to consider one chunk per thread
/// 									block. 
/// \param [in]		d_numElemsNext		Number of elements for each next list node. 
/// \param [out]	d_outIsSmallElem	Binary 0/1 array. Will contain 1 for each "small" element,
///										i.e. an element that belongs to a small node.
/// \param [out]	d_outIsLargeElem	Binary 0/1 array. Will contain 1 for each "large" element,
///										i.e. an element that belongs to a large node.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MarkElemsByNodeSize(KDChunkList lstChunks, uint* d_numElemsNext, 
										   uint* d_outIsSmallElem, uint* d_outIsLargeElem)
{
	uint chk = MNCUDA_GRID2DINDEX;

	__shared__ uint s_numElemsChunk;
	__shared__ uint s_idxNode;
	__shared__ uint s_idxFirstElem;
	if(threadIdx.x == 0)
	{
		s_numElemsChunk = lstChunks.d_numElems[chk];
		s_idxNode = lstChunks.d_idxNode[chk];
		s_idxFirstElem = lstChunks.d_idxFirstElem[chk];
	}
	__syncthreads();

	if(threadIdx.x < s_numElemsChunk)
	{
		uint isSmall = ((d_numElemsNext[s_idxNode] <= c_smallNodeMax) ? 1 : 0);
		d_outIsSmallElem[s_idxFirstElem + threadIdx.x] = isSmall;
		d_outIsLargeElem[s_idxFirstElem + threadIdx.x] = 1 - isSmall;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MoveNodes(uint srcNum, float4* srcAABBMin, float4* srcAABBMax,
/// 	uint* srcLevel, uint tarNum, float4* tarAABBMin, float4* tarAABBMax, uint* tarLevel,
/// 	uint* d_move, uint* d_offsets)
///
/// \brief	Copies nodes from one node list to another.
///
///			Only node levels and AABBs are copied.
/// 		
/// \note	Node list data structures were split up to avoid parameter space overflow. Furthermore
///			this allows to use different source/target AABB types.
///
/// \author	Mathias Neumann
/// \date	19.02.2010
///
/// \param	srcNum				Source number of nodes. 
/// \param [in]		srcAABBMin	Source AABB minimum. 
/// \param [in]		srcAABBMax	Source AABB maximum. 
/// \param [in]		srcLevel	Source node levels. 
/// \param	tarNum				Target number of elements. 
/// \param [in,out]	tarAABBMin	Target AABB minimum. 
/// \param [in,out]	tarAABBMax	Target AABB maximum. 
/// \param [in,out]	tarLevel	Target node levels. 
/// \param [in]		d_move		Binary 0/1 array. Contains 1 if to be copied, else 0. 
/// \param [in]		d_offsets	Target offsets. That is, \a d_offsets[i] conains where to place
/// 							the i-th node of the source node list. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MoveNodes(uint srcNum, float4* srcAABBMin, float4* srcAABBMax, uint* srcLevel,
								 uint tarNum, float4* tarAABBMin, float4* tarAABBMax, uint* tarLevel,
								 uint* d_move, uint* d_offsets)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < srcNum)
	{
		// Get node offset.
		uint offset = tarNum + d_offsets[idx];

		if(d_move[idx] != 0)
		{
			// Copy only valid information. ENA is handled separately.
			tarLevel[offset] = srcLevel[idx];

			// Inherited bounds. Tight bounds are computed.
			tarAABBMin[offset] = srcAABBMin[idx];
			tarAABBMax[offset] = srcAABBMax[idx];
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_UpdateSmallRootParents(KDFinalNodeList lstNodes,
/// 	uint* d_smallRootParents, uint numSmallNodes)
///
/// \brief	Updates small root parents in final node list.
/// 		
/// 		Small roots are the initial small nodes, i.e. element of the small node list. They
/// 		are appended to the final node list at a much later point, so that their concrete
/// 		indices are not known in large node stage. This kernel fixes this problem by setting
/// 		the KDFinalNodeList::d_childLeft and KDFinalNodeList::d_childRight members for all
/// 		parents of such small roots. 
///
/// \author	Mathias Neumann
/// \date	02.03.2010
///
/// \param	lstNodes					The final node list. 
/// \param [in]		d_smallRootParents	The small root parent array as constructed by
/// 									kernel_MarkSmallNodes(). Contains the index of the parent
/// 									node in the final node list for each small list node. The
/// 									MSB is used to mark left (unset) and right (set) children,
/// 									so that the correct address can be set. 
/// \param	numSmallNodes				Number of small nodes in small node list.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_UpdateSmallRootParents(KDFinalNodeList lstNodes, uint* d_smallRootParents, uint numSmallNodes)
{
	uint idxSmall = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxSmall < numSmallNodes)
	{
		uint smp = d_smallRootParents[idxSmall];
		uint idxNode = smp & 0x7FFFFFFF; // Mask out MSB.
		uint isRight = smp >> 31; // MSB tells us if right child.

		if(isRight)
			lstNodes.d_childRight[idxNode] = lstNodes.numNodes + idxSmall;
		else
			lstNodes.d_childLeft[idxNode] = lstNodes.numNodes + idxSmall;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <uint numElementPoints> __global__ void kernel_CreateSplitCandidates(KDNodeList lstSmall,
/// 	KDSplitList lstSplit)
///
/// \brief	Creates split candidate list for small node stage.
/// 		
///			A split list is prepared by assigning split position and split axis. This is done for
///			each small list node. All element boundaries define possible splits, hence the number
///			of splits depends on the number of element points. If it is 1, there is just one
///			possible split position on a single axis. If it is 2, there are two (min, max) split
///			positions.
///
/// \warning Heavy uncoalesced access!
///
/// \author	Mathias Neumann
/// \date	22.02.2010
///
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstSmall	The small node list. 
/// \param	lstSplit	The split list generated. Masks are initialized by kernel_InitSplitMasks().
////////////////////////////////////////////////////////////////////////////////////////////////////
template <uint numElementPoints>
__global__ void kernel_CreateSplitCandidates(KDNodeList lstSmall, KDSplitList lstSplit)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	// Over all nodes in small list.
	if(idxNode < lstSmall.numNodes)
	{
		// First split index.
		uint idxFirstTri = lstSmall.d_idxFirstElem[idxNode];
		uint numElems = lstSmall.d_numElems[idxNode];
		uint idxFirstSplit = lstSplit.d_idxFirstSplit[idxNode];

		// NOTE: Do not try to simplify the loops since the order is significant for other kernels.
		uint idxSplit = idxFirstSplit;
		for(uint axis=0; axis<3; axis++)
		{
			// Minimums / only point.
			for(uint i=0; i<numElems; i++)
			{
				uint idxTNA = idxFirstTri + i;
				// Store both position and axis (lowest 2 bits).
				lstSplit.d_splitPos[idxSplit] = ((float*)&lstSmall.d_elemPoint1[idxTNA])[axis];
				uint info = axis;
				lstSplit.d_splitInfo[idxSplit] = info;
				idxSplit++;
			}
		}

		if(numElementPoints == 2) // COMPILE TIME
		{
			for(uint axis=0; axis<3; axis++)
			{
				// Maximums
				for(uint i=0; i<numElems; i++)
				{
					uint idxTNA = idxFirstTri + i;
					// Store both position and axis (lowest 2 bits).
					lstSplit.d_splitPos[idxSplit] = ((float*)&lstSmall.d_elemPoint2[idxTNA])[axis];
					uint info = 4; // Set max bit.
					info |= axis;
					lstSplit.d_splitInfo[idxSplit] = info;
					idxSplit++;
				}	
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template<bool isMax, uint numElementPoints> __global__ void kernel_InitSplitMasks(KDNodeList lstSmall,
/// 	float* d_randoms, KDSplitList lstSplit)
///
/// \brief	Initializes split masks.
/// 		
/// 		Generates KDSplitList::d_maskLeft and KDSplitList::d_maskRight for each split. This
/// 		is done by setting all bits of those node elements lying on the left and right side
/// 		respectively. 
///
/// \note	Required shared memory per thread block: 24 * ::KD_SMALLNODEMAX + 12 bytes.
///
/// \author	Mathias Neumann
/// \date	21.02.2010
/// \tparam	isMax				Whether to generate minimum or maximum splits. When the number of
/// 							element point is 1, you have to pass \c false. 
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstSmall			The small node list. 
/// \param [in]		d_randoms	Uniform random numbers used to avoid endless split loops if
/// 							several elements lie in a splitting plane. Should contain at
/// 							least lstSmall.nextFreePos numbers. 
/// \param	lstSplit			The split list. Masks are initialized by this kernel. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool isMax, uint numElementPoints>
__global__ void kernel_InitSplitMasks(KDNodeList lstSmall, float* d_randoms, KDSplitList lstSplit)
{
	uint idxNode = MNCUDA_GRID2DINDEX;

	__shared__ uint s_idxFirstElem;
	__shared__ uint s_numElems;
	__shared__ uint s_idxFirstSplit;
	if(threadIdx.x == 0)
	{
		s_idxFirstElem = lstSmall.d_idxFirstElem[idxNode];
		s_numElems = lstSmall.d_numElems[idxNode];
		s_idxFirstSplit = lstSplit.d_idxFirstSplit[idxNode];
	}
	__syncthreads();

	uint idxSplit = s_idxFirstSplit + threadIdx.x;
	if(isMax) // evaluated at compile time!
		idxSplit += 3*s_numElems;

	// First read triangle bounds into shared memory.
	__shared__ float3 s_point1[KD_SMALLNODEMAX];
	// Find way to eliminate s_point2 for numElementPoints == 1. Probably it's just left out
	// by the compiler.
	__shared__ float3 s_point2[KD_SMALLNODEMAX];
	if(threadIdx.x < s_numElems)
	{
		s_point1[threadIdx.x] = make_float3(lstSmall.d_elemPoint1[s_idxFirstElem + threadIdx.x]);
		if(numElementPoints == 2) // COMPILE TIME
			s_point2[threadIdx.x] = make_float3(lstSmall.d_elemPoint2[s_idxFirstElem + threadIdx.x]);
	}
	__syncthreads();

	if(threadIdx.x < 3*s_numElems)
	{
		// Get check element on both sides.
		ElementMask maskL = 0, maskR = 0;

		// Get split information.
		// This *will* lead to uncoalesced access when s_numElems is not aligned.
		float splitPos = lstSplit.d_splitPos[idxSplit];
		uint splitAxis = lstSplit.d_splitInfo[idxSplit];
		splitAxis &= 3;
		// TNA index of the element that defined the split.
		uint idxTNA = s_idxFirstElem + (threadIdx.x - splitAxis*s_numElems);

		// NOTE: We are working on the small root list here. Therefore all relevant bits
		//		 of the element mask are set and we do not have to check which bit to
		//		 set. Instead we can just iterate from 0 to triCount-1.
		//uint cntL = 0, cntR = 0;
		if(numElementPoints == 1) // COMPILE TIME
		{
			// Unrolling won't work when moving the compile time if into the loop.
			#pragma unroll
			for(uint i=0; i<s_numElems; i++)
			{
				// Get point on our axis.
				float elemPos = ((float*)&s_point1[i])[splitAxis];

				uint isLeft = 0;
				uint isRight = 0;
				if(d_randoms[s_idxFirstElem + i] < 0.5f)
				{
					if(elemPos < splitPos)
						isLeft = 1;
					if(splitPos <= elemPos)
						isRight = 1;
				}
				else
				{
					if(elemPos <= splitPos)
						isLeft = 1;
					if(splitPos < elemPos)
						isRight = 1;
				}

				maskL |= (((ElementMask)isLeft) << i);
				maskR |= (((ElementMask)isRight) << i);
				//cntL += isLeft;
				//cntR += isRight;
			}
		}
		else // numElementPoints == 2
		{
			#pragma unroll
			for(uint i=0; i<s_numElems; i++)
			{
				// Get triangle bounds on our axis.
				float fMin = 0, fMax = 0;
				fMin = ((float*)&s_point1[i])[splitAxis];
				fMax = ((float*)&s_point2[i])[splitAxis];

				uint isLeft = 0;
				uint isRight = 0;
				if(fMin < splitPos)
					isLeft = 1;
				if(splitPos < fMax)
					isRight = 1;

				// Check whether the triangle is the split triangle and classify it according to
				// the type of the split:
				//  - Maximum splits split off the right part of the volume. Therefore the triangle
				//	  has to lie on the left side only.
				//  - Minimum splits split off the left part of the volume. Here the triangle has
				//	  to lie on the right side only.
				// NOTE: According to the profiler, this won't generate too many warp serializes.
				if(isMax) // evaluated at compile time!
					if(s_idxFirstElem + i == idxTNA)
						isLeft = 1;
				if(!isMax) // evaluated at compile time!
					if(s_idxFirstElem + i == idxTNA)
						isRight = 1;

				// Additionally check whether the element lies directly in the splitting plane
				// and is *not* the split element.
				// NOTE: This generates many warp serializes.
				if(s_idxFirstElem + i != idxTNA && fMin == fMax && fMin == splitPos)
				{
					if(d_randoms[s_idxFirstElem + i] < 0.5f)
						isLeft = 1;
					else
						isRight = 1;
				}

				maskL |= (((ElementMask)isLeft) << i);
				maskR |= (((ElementMask)isRight) << i);
				//cntL += isLeft;
				//cntR += isRight;
			}
		}

		//if(dev_CountBits(maskL) <= 1 || dev_CountBits(maskR) <= 1)
		//	printf("SPLIT: %d L, %d R (total: %d).\n", cntL, cntR, s_numElems);

#ifndef __DEVICE_EMULATION__
		__syncthreads();
#endif

		lstSplit.d_maskLeft[idxSplit] = maskL;
		lstSplit.d_maskRight[idxSplit] = maskR;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template<uint numElementPoints> __global__ void kernel_FindBestSplits(KDNodeList lstActive,
/// 	KDSplitList lstSplit, uint* d_outBestSplit, float* d_outSplitCost)
///
/// \brief	Looks for best splits by checking all nodes in parallel.
/// 		
/// 		Depending on the value of \a numElementPoints either VVH cost model (for value 1)
///			or SAH cost model (for value 2) are employed. For each node, the cost for using
///			each of the splitting planes is evaluated and the minimum of the costs is computed.
///			To do this in parallel, ::dev_ReduceFast() is used for reduction.
///
///			The used thread block size for this kernel has to be at least 
///
///			\code KD_SMALLNODEMAX * 6 / 3 = KD_SMALLNODEMAX * 2 \endcode
///
///			to ensure all splits will be handled. Each thread block works on a node. Each thread
///			reduces the number of cost values internally by the factor of 3 to improve reduction
///			performance (hence the division by 3). The factor 6 used above results from the fact
///			that \a numElementPoints = 2 leads to at most 6 possible splitting planes per
///			element.
///
/// \note	Required shared memory per thread block: 4 * 128 + 60 bytes.
///
/// \author	Mathias Neumann
/// \date	22.03.2010
///
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstActive				The active node list. 
/// \param	lstSplit				The split list. 
/// \param [out]	d_outBestSplit	The best split positions for each node. 
/// \param [out]	d_outSplitCost	The best split costs for each node. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template<uint numElementPoints>
__global__ void kernel_FindBestSplits(KDNodeList lstActive, KDSplitList lstSplit,
									   uint* d_outBestSplit, float* d_outSplitCost)
{
	uint idxNode = MNCUDA_GRID2DINDEX;
	
	__shared__ uint s_idxFirstSplit;
	__shared__ uint s_numSplits;
	__shared__ float s_fCostDenom;
	__shared__ ElementMask s_maskNode;
	__shared__ float s_aabbNodeMin[3];
	__shared__ float s_aabbNodeMax[3];
	__shared__ float s_extendNode[3];
	if(threadIdx.x == 0)
	{
		uint idxSmallRoot = lstActive.d_idxSmallRoot[idxNode];
		s_idxFirstSplit = lstSplit.d_idxFirstSplit[idxSmallRoot];
		s_numSplits = lstSplit.d_numSplits[idxSmallRoot];

		// Get node's tight bounds for SAH calculation. This is in small stage, therefore no
		// inherited bounds available!
		float4 min4 = lstActive.d_aabbMinTight[idxNode];
		float4 max4 = lstActive.d_aabbMaxTight[idxNode];
		for(uint i=0; i<3; i++)
		{
			s_aabbNodeMin[i] = ((float*)&min4)[i];
			s_aabbNodeMax[i] = ((float*)&max4)[i];
			s_extendNode[i] = s_aabbNodeMax[i] - s_aabbNodeMin[i];
		}

		// Get node's element mask.
		s_maskNode = lstActive.d_elemMask[idxNode];

		// Get denominator for cost evaluation.
		if(numElementPoints == 1) // COMPILE TIME
		{
			// Get the inverse of the extended node volume.
			float exNodeVol = 1.f;
			float maxQR2 = 2*c_maxQueryRadius;
			for(uint i=0; i<3; i++)
				exNodeVol *= s_extendNode[i] + maxQR2;
			s_fCostDenom = 1.f / exNodeVol;
		}
		else // numElementPoints == 2
		{
			// Compute the inverse area of the node for SAH calculation.
			float fAreaNode = 2.f * (s_extendNode[0]*s_extendNode[1] + s_extendNode[0]*s_extendNode[2] + 
									 s_extendNode[1]*s_extendNode[2]);
			s_fCostDenom = 1.f / fAreaNode;
		}
	}
	__syncthreads();

	// Shared memory for best split SAH.
	__shared__ float s_SplitCosts[128];

	// Do first step of the reduction by reducing three values per thread.
	uint idxSplit[3];
	float fMyCost[3];
	ElementMask maskNode = s_maskNode;
	if(numElementPoints == 1) // COMPILE TIME
	{
		// Moving if inside loop makes auto unrolling impossible...
		#pragma unroll
		for(uint i=0; i<3; i++)
		{
			uint myTID = threadIdx.x + i*blockDim.x;
			idxSplit[i] = s_idxFirstSplit + myTID;
			fMyCost[i] = MN_INFINITY;
			if(myTID < s_numSplits)
			{
				float splitPos = lstSplit.d_splitPos[idxSplit[i]];
				uint splitAxis = lstSplit.d_splitInfo[idxSplit[i]] & 3;

				// Get AND'ed element masks to only recognize the contained elements.
				ElementMask maskL = lstSplit.d_maskLeft[idxSplit[i]] & maskNode;
				ElementMask maskR = lstSplit.d_maskRight[idxSplit[i]] & maskNode;

				// Count triangles using parallel bit counting.
				uint countL = dev_CountBits(maskL);
				uint countR = dev_CountBits(maskR);

				// Get extended child volumes to perform VVH cost calculation.
				uint otherAxis1 = splitAxis+1;
				uint otherAxis2 = splitAxis+2;
				if(otherAxis1 == 3)
					otherAxis1 = 0;
				if(otherAxis2 > 2)
					otherAxis2 -= 3;
				float minSplitAxis = s_aabbNodeMin[splitAxis];
				float maxQR2 = 2*c_maxQueryRadius;
				float fVolL = (s_extendNode[otherAxis1] + maxQR2) * 
							  (splitPos - minSplitAxis + maxQR2) *
							  (s_extendNode[otherAxis2] + maxQR2);
				float maxSplitAxis = s_aabbNodeMax[splitAxis];
				float fVolR = (s_extendNode[otherAxis1] + maxQR2) * 
							  (maxSplitAxis - splitPos + maxQR2) *
							  (s_extendNode[otherAxis2] + maxQR2);

				// Compute VVH cost for this split.
				// WARNING: Picking traversal cost too low can result in endless splitting.
				fMyCost[i] = (countL*fVolL + countR*fVolR)*s_fCostDenom + KD_COST_TRAVERSE;

				// Avoid useless splits.
				if(countL == 0 || countR == 0)
					fMyCost[i] = MN_INFINITY;
			}
		}
	}
	else // numElementPoints == 2
	{
		#pragma unroll
		for(uint i=0; i<3; i++)
		{
			uint myTID = threadIdx.x + i*blockDim.x;
			idxSplit[i] = s_idxFirstSplit + myTID;
			fMyCost[i] = MN_INFINITY;
			if(myTID < s_numSplits)
			{
				float splitPos = lstSplit.d_splitPos[idxSplit[i]];
				uint splitAxis = lstSplit.d_splitInfo[idxSplit[i]] & 3;

				// Get AND'ed element masks to only recognize the contained elements.
				ElementMask maskL = lstSplit.d_maskLeft[idxSplit[i]] & maskNode;
				ElementMask maskR = lstSplit.d_maskRight[idxSplit[i]] & maskNode;

				// Count triangles using parallel bit counting.
				uint countL = dev_CountBits(maskL);
				uint countR = dev_CountBits(maskR);

				// Get child areas to perform SAH cost calculation.
				uint otherAxis1 = splitAxis+1;
				uint otherAxis2 = splitAxis+2;
				if(otherAxis1 == 3)
					otherAxis1 = 0;
				if(otherAxis2 > 2)
					otherAxis2 -= 3;
				float minSplitAxis = s_aabbNodeMin[splitAxis];
				float fAreaL = 2.f * (s_extendNode[otherAxis1]*s_extendNode[otherAxis2] + 
										(splitPos - minSplitAxis) *
										(s_extendNode[otherAxis1] + s_extendNode[otherAxis2]));
				float maxSplitAxis = s_aabbNodeMax[splitAxis];
				float fAreaR = 2.f * (s_extendNode[otherAxis1]*s_extendNode[otherAxis2] + 
										(maxSplitAxis - splitPos) *
										(s_extendNode[otherAxis1] + s_extendNode[otherAxis2]));

				// Compute SAH cost for this split.
				// WARNING: Picking traversal cost too low can result in endless splitting.
				fMyCost[i] = (countL*fAreaL + countR*fAreaR)*s_fCostDenom + KD_COST_TRAVERSE;

				// Avoid useless splits.
				if(countL == 0 || countR == 0)
					fMyCost[i] = MN_INFINITY;
			}
		}
	}

	s_SplitCosts[threadIdx.x] = fminf(fminf(fMyCost[0], fMyCost[1]), fMyCost[2]);
	__syncthreads();

	// Now perform reduction on costs to find minimum.
	float fMinCost = dev_ReduceFast<float, 128, ReduceOperatorTraits<float, MNCuda_MIN>>(s_SplitCosts);

	// Get minimum index. Initialize index to 0xffffffff to identify cases where we could not
	// determine the correct split index for the minimum.
	__shared__ volatile uint s_idxMin;
	if(threadIdx.x == 0)
		s_idxMin = 0xffffffff;
	__syncthreads();

	if(threadIdx.x < s_numSplits && fMyCost[0] == fMinCost)
		s_idxMin = idxSplit[0];
	if(threadIdx.x+blockDim.x < s_numSplits && fMyCost[1] == fMinCost)
		s_idxMin = idxSplit[1];
	if(threadIdx.x+2*blockDim.x < s_numSplits && fMyCost[2] == fMinCost)
		s_idxMin = idxSplit[2];
	__syncthreads();

	if(threadIdx.x == 0)
	{
		d_outBestSplit[idxNode] = s_idxMin;
		d_outSplitCost[idxNode] = fMinCost;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SplitSmallNodes(KDNodeListSmall lstActive, KDSplitList lstSplit,
/// 	KDNodeListSmall lstNext, uint* d_inBestSplit, float* d_inSplitCost, uint* d_outIsSplit)
///
/// \brief	Performs splitting of small nodes according to best split information.
/// 		
/// 		A split is created when the best split cost of a given node is smaller than it's
/// 		element count. The latter equals the cost of leaving the node unsplitted.
/// 		
/// 		All created nodes are written into the next list. The element masks of new nodes can
/// 		be obtained using a Boolean AND of the corresponding split list mask and the original
/// 		node's mask. The actual ENA is not updated by this kernel. See
/// 		kernel_GenerateENAFromMasks(). 
///
/// \author	Mathias Neumann
/// \date	22.03.2010 
///	\see	kernel_FindBestSplits()
///
/// \param	lstActive				The active list (source). Contains nodes to split. If a split
/// 								was performed, split information in this list, i.e. split
/// 								position and axis are updated. 
/// \param	lstSplit				The split list. 
/// \param	lstNext					The next list (target). Will contain child nodes resulting
/// 								from splits. Note that there might be holes as not all nodes
/// 								will get splitted up. 
/// \param [in]		d_inBestSplit	The best split positions for each active list node. 
/// \param [in]		d_inSplitCost	The best split costs for each active list node. 
/// \param [out]	d_outIsSplit	Binary 0/1 array of length \c lstActive.numNodes. Will
/// 								contain 1 only for nodes that were splitted. Can be used to
/// 								compact the resulting next list. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SplitSmallNodes(KDNodeListSmall lstActive, KDSplitList lstSplit, KDNodeListSmall lstNext,
									   uint* d_inBestSplit, float* d_inSplitCost, uint* d_outIsSplit)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	// Over all nodes in (small) active list.
	if(idxNode < lstActive.numNodes)
	{
		ElementMask maskNode = lstActive.d_elemMask[idxNode];
		uint idxSmallRoot = lstActive.d_idxSmallRoot[idxNode];
		uint oldLevel = lstActive.d_nodeLevel[idxNode];

		// Get node's tight bounds.
		float4 tmp = lstActive.d_aabbMinTight[idxNode];
		float aabbNodeMin[3] = {tmp.x, tmp.y, tmp.z};
		tmp = lstActive.d_aabbMaxTight[idxNode];
		float aabbNodeMax[3] = {tmp.x, tmp.y, tmp.z};

		// Just read out element count for no-split-cost. No bit counting here.
		float fCost0 = lstActive.d_numElems[idxNode];
		uint idxSplitMin = d_inBestSplit[idxNode];
		float fCostMin = d_inSplitCost[idxNode];
		
		// Check whether leaf node or not.
		uint left, right;
		uint isSplit;
		uint splitAxis = 0;
		float splitPos = 0.f;
		if(fCostMin >= fCost0 || idxSplitMin == 0xffffffff)
		{
			// Leaf. Set left/right to 0.
			left = 0;
			right = 0;
			isSplit = 0;
		}
		else
		{
			//printf("Is splitting. SAH0: %.3f; SAHMin: %.3f.\n", fSAH0, fSAHMin);

			// To split.
			isSplit = 1;
			splitPos = lstSplit.d_splitPos[idxSplitMin];
			splitAxis = lstSplit.d_splitInfo[idxSplitMin] & 3;

			ElementMask maskL = lstSplit.d_maskLeft[idxSplitMin] & maskNode;
			ElementMask maskR = lstSplit.d_maskRight[idxSplitMin] & maskNode;
			uint countL = dev_CountBits(maskL);
			uint countR = dev_CountBits(maskR);
			//printf("Split tri counts: %d and %d, node: %d.\n", countL, countR, lstActive.d_numElems[idxNode]);

			// Left child.
			left = idxNode;
			//float tempMin = aabbNodeMin[splitAxis];
			float tempMax = aabbNodeMax[splitAxis];
			aabbNodeMax[splitAxis] = splitPos;
			lstNext.d_aabbMinTight[left] = make_float4(aabbNodeMin[0], aabbNodeMin[1], aabbNodeMin[2], 0.f);
			lstNext.d_aabbMaxTight[left] = make_float4(aabbNodeMax[0], aabbNodeMax[1], aabbNodeMax[2], 0.f);
			lstNext.d_idxSmallRoot[left] = idxSmallRoot;
			lstNext.d_numElems[left] = countL;
			lstNext.d_nodeLevel[left] = oldLevel + 1;
			lstNext.d_elemMask[left] = maskL;

			// Right child.
			right = lstActive.numNodes + idxNode;
			aabbNodeMin[splitAxis] = splitPos;
			aabbNodeMax[splitAxis] = tempMax;
			lstNext.d_aabbMinTight[right] = make_float4(aabbNodeMin[0], aabbNodeMin[1], aabbNodeMin[2], 0.f);
			lstNext.d_aabbMaxTight[right] = make_float4(aabbNodeMax[0], aabbNodeMax[1], aabbNodeMax[2], 0.f);
			lstNext.d_idxSmallRoot[right] = idxSmallRoot;
			lstNext.d_numElems[right] = countR;
			lstNext.d_nodeLevel[right] = oldLevel + 1;
			lstNext.d_elemMask[right] = maskR;

			//printf("split axis: %.3f - %.3f - %.3f\n", tempMin, splitPos, tempMax);
		}

		// Also store split information since we need it later, even for small nodes!
		lstActive.d_splitPos[idxNode] = splitPos;
		lstActive.d_splitAxis[idxNode] = splitAxis;
		d_outIsSplit[idxNode] = isSplit;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GenerateENAFromMasks(KDNodeListENA lstActive,
/// 	KDNodeListENA lstSmallRoots)
///
/// \brief	Generates ENA entries from element masks in active list by using the small roots ENA
/// 		data.
/// 		
/// 		Currently this is implemented in a node based way, which leads to some uncoalesced
/// 		access. But this is by far superior to an approach that uses ::KD_SMALLNODEMAX
/// 		threads per node to mark each node in parallel and than scans those marks This will
/// 		lead to a large amount of instructions and a much higher computation time. 
///
/// \author	Mathias Neumann
/// \date	25.03.2010
///
/// \param	lstActive		The active node list. 
/// \param	lstSmallRoots	The small roots node list.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GenerateENAFromMasks(KDNodeListENA lstActive, KDNodeListENA lstSmallRoots)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	// Over all nodes in (small) active list.
	if(idxNode < lstActive.numNodes)
	{
		ElementMask maskNode = lstActive.d_elemMask[idxNode];
		uint idxFirstElem = lstActive.d_idxFirstElem[idxNode];
		uint idxSmallRoot = lstActive.d_idxSmallRoot[idxNode];
		uint idxFirstTriSR = lstSmallRoots.d_idxFirstElem[idxSmallRoot];

		// NOTE: lstSmallRoots is correct here since we need to consider all elements
		//		 in the small root list.
		uint offset = idxFirstElem;
		for(uint i=0; i<lstSmallRoots.d_numElems[idxSmallRoot]; i++)
		{
			if(maskNode & 0x1)
			{
				uint idxSmallElem = lstSmallRoots.d_elemNodeAssoc[idxFirstTriSR+i];
				lstActive.d_elemNodeAssoc[offset] = idxSmallElem;
				offset++;
			}
			maskNode = maskNode >> 1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TraversalUpPath(KDFinalNodeList lstFinal, uint curLevel,
/// 	uint* d_sizes)
///
/// \brief	Generates node sizes for a given node level.
/// 		
/// 		Considers all nodes on a given node level and writes the node side to the size array.
///			The node size is composed the following way:
///
///			\li Inner node: left size + elem idx (1) + parent info (2) + right size
///			\li Leaf: node index (1) + element count (1) + element indices (n)
///
///			Here left size is the size of the left subtree, i.e. \c d_sizes[\c left], when \c left
///			is the index of the left child (right size respectively).
///
/// \warning Has to be called bottom-up.
///
/// \author	Mathias Neumann
/// \date	05.03.2010
/// \see	KDTreeData
///
/// \param	lstFinal		The final node list. 
/// \param	curLevel		The level to consider. 
/// \param [in,out]	d_sizes	Generated node sizes. Sizes for given level are updated. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TraversalUpPath(KDFinalNodeList lstFinal, uint curLevel, uint* d_sizes)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes && lstFinal.d_nodeLevel[idxNode] == curLevel)
	{
		uint left = lstFinal.d_childLeft[idxNode];
		uint right = lstFinal.d_childRight[idxNode];
		uint numElems = lstFinal.d_numElems[idxNode];

		uint size;
		if(left == right)
		{
			// Leaf node (node index (1) + element count (1) + element indices (n)).
			size = 1 + 1 + numElems;
		}
		else
		{
			// Internal node: left size + elem idx (1) + parent info (2) + right size.
			size = d_sizes[left] + 1 + 2 + d_sizes[right];
		}

		d_sizes[idxNode] = size;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TraversalDownPath(KDFinalNodeList lstFinal, uint curLevel,
/// 	uint* d_sizes, uint* d_addresses, KDTreeData kdData)
///
/// \brief	Initializes KDTreeData::d_preorderTree using precomputed node sizes.
/// 		
/// 		Assigns final node list data to entries of KDTreeData::d_preorderTree.
/// 		
/// 		\warning Has to be called top-down. 
///
/// \author	Mathias Neumann
/// \date	05.03.2010
///
/// \param	lstFinal			The final node list. 
/// \param	curLevel			The current level. 
/// \param [in]		d_sizes		Generated node sizes, see kernel_TraversalUpPath(). 
/// \param [in,out]	d_addresses	The addresses of the nodes. Updated for the children of the
/// 							current tree level. First entry (root entry) should be
/// 							initialized with 0 because it is assumed that the addresses for
/// 							the current node level are valid. Hence calling this method
/// 							top-down is required. 
/// \param	kdData				The final kd-tree layout data structure.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TraversalDownPath(KDFinalNodeList lstFinal, uint curLevel, uint* d_sizes,
										 uint* d_addresses, KDTreeData kdData)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes && lstFinal.d_nodeLevel[idxNode] == curLevel)
	{
		uint left = lstFinal.d_childLeft[idxNode];
		uint right = lstFinal.d_childRight[idxNode];
		uint myAddress = d_addresses[idxNode];
		uint idxFirstTri = lstFinal.d_idxFirstElem[idxNode];
		uint numElems = lstFinal.d_numElems[idxNode];

		// Add in leaf information.
		uint idxNodeLeaf = idxNode;
		idxNodeLeaf |= ((left == right) ? 0x80000000 : 0);

		// Now write node idxNode to myAddress
		kdData.d_preorderTree[myAddress] = idxNodeLeaf;

		float splitPos = lstFinal.d_splitPos[idxNode];
		uint splitAxis = lstFinal.d_splitAxis[idxNode];

		uint addrL, addrR;
		if(left != right)
		{
			// Internal node.
			addrL = myAddress + 2 + 1;
			addrR = myAddress + 2 + 1 + d_sizes[left];
			d_addresses[left] = addrL;
			d_addresses[right] = addrR;

			// Write parent info.
			uint2 parentInfo;
			parentInfo.x = addrR;
			parentInfo.x &= 0x0FFFFFF; // only 28 bits
			// Write split axis (2 bits) to most significant two bits. Leave custom bits 28, 29 alone.
			parentInfo.x |= (splitAxis << 30);
			parentInfo.y = *(uint*)&splitPos;
			kdData.d_preorderTree[myAddress+1] = parentInfo.x;
			kdData.d_preorderTree[myAddress+2] = parentInfo.y;
		}
		else
		{
			addrL = 0;
			addrR = 0;

			kdData.d_preorderTree[myAddress+1] = numElems;

			// Write element indices to preorder tree.
			for(uint i=0; i<numElems; i++)
				kdData.d_preorderTree[myAddress+2+i] = lstFinal.d_elemNodeAssoc[idxFirstTri + i];
		}

		// Compute and write node extent (center, radius).
		float3 aabbMin = make_float3(lstFinal.d_aabbMin[idxNode]);
		float3 aabbMax = make_float3(lstFinal.d_aabbMax[idxNode]);

		float3 diagonal = aabbMax - aabbMin;
		float radius = 0.5f*length(diagonal);
		float3 nodeCenter = aabbMin + 0.5f*diagonal;
		kdData.d_nodeExtent[idxNode] = make_float4(nodeCenter.x, nodeCenter.y, nodeCenter.z, radius);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <uint numElementPoints> __global__ void kernel_TestNodeList(KDNodeList lstNodes,
/// 	uint* d_valid, float3 rootMin, float3 rootMax, bool useTightBounds)
///
/// \brief	Tests a given node list.
/// 		
/// 		This kernel tests whether the element bounds are within the node bounds. No other
///			tests are performed. Use for debugging purposes only, as it is quite slow.
///
/// \warning Requires element points and node bounds!
///
/// \author	Mathias Neumann
/// \date	15.03.2010
///
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstNodes		The node list to test. 
/// \param [in,out]	d_valid	Provide one element array. Initialize the element with 1. The kernel
///							will assign a 0 when the test failed.
/// \param	rootMin			Root AABB minimum. 
/// \param	rootMax			Root AABB maximum. 
/// \param	useTightBounds	Whether to use tight or inherited bounds. Depends on node list to test.
///							E.g. in small stage, there are no more inherited bounds available.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <uint numElementPoints>
__global__ void kernel_TestNodeList(KDNodeList lstNodes, uint* d_valid, float3 rootMin, float3 rootMax,
									bool useTightBounds)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	// Over all nodes in list.
	if(idxNode < lstNodes.numNodes)
	{
		float3 aabbNodeMin, aabbNodeMax;
		if(useTightBounds)
		{
			aabbNodeMin = make_float3(lstNodes.d_aabbMinTight[idxNode]);
			aabbNodeMax = make_float3(lstNodes.d_aabbMaxTight[idxNode]);
		}
		else
		{
			aabbNodeMin = make_float3(lstNodes.d_aabbMinInherit[idxNode]);
			aabbNodeMax = make_float3(lstNodes.d_aabbMaxInherit[idxNode]);
		}
		float* paabbNodeMin = (float*)&aabbNodeMin;
		float* paabbNodeMax = (float*)&aabbNodeMax;
		float* prootMin = (float*)&rootMin;
		float* prootMax = (float*)&rootMax;

		uint idxFirstTri = lstNodes.d_idxFirstElem[idxNode];
		uint numElems = lstNodes.d_numElems[idxNode];

		const float relErrorAllowed = 1e-6f;

		for(uint t=idxFirstTri; t<idxFirstTri+numElems; t++)
		{
			// Test if element is really in bounds.
			float3 pt1, pt2;
			pt1 = make_float3(lstNodes.d_elemPoint1[t]);
			if(numElementPoints == 2)
				pt2 = make_float3(lstNodes.d_elemPoint2[t]);

			for(uint c=0; c<3; c++)
			{
				if(numElementPoints == 1) // COMPILE TIME
				{
					float value = ((float*)&pt1)[c];
					bool ok = true;
					ok &= fabsf(fmaxf(0.f, paabbNodeMin[c] - value)) < relErrorAllowed * (prootMax[c] - prootMin[c]);
					ok &= fabsf(fmaxf(0.f, value - paabbNodeMax[c])) < relErrorAllowed * (prootMax[c] - prootMin[c]);
					if(!ok)
						d_valid[0] = 0;
				}
				else // numElementPoints == 2
				{
					float fMin = ((float*)&pt1)[c];
					float fMax = ((float*)&pt2)[c];
					bool ok = true;
					ok &= fabsf(fmaxf(0.f, paabbNodeMin[c] - fMin)) < relErrorAllowed * (prootMax[c] - prootMin[c]);
					ok &= fabsf(fmaxf(0.f, fMax - paabbNodeMax[c])) < relErrorAllowed * (prootMax[c] - prootMin[c]);
					if(!ok)
						d_valid[0] = 0;
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SetCustomBit(KDTreeData kdData, uint bitNo, uint* d_values)
///
/// \brief	Sets chosen custom bit in the kd-tree data. 
///
///			Allows to mark some \e inner nodes of the kd-tree without adding new arrays that could
///			increase cache load during traversal.
///
/// \author	Mathias Neumann
/// \date	28.07.2010
/// \see	KDTreeData
///
/// \param	kdData				The final kd-tree data. 
/// \param	bitNo				The custom bit to set. Has to be 0 or 1. 
/// \param [in]		d_values	The new custom bit values (0 or 1), one for each node. Only inner
///								node bits are used, as leaf nodes have no custom bits.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SetCustomBit(KDTreeData kdData, uint bitNo, uint* d_values)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < kdData.numNodes)
	{
		uint addrNode = kdData.d_nodeAddresses[idxNode];
		uint isLeaf = kdData.d_preorderTree[addrNode] & 0x80000000;

		// Ignore leafs as they have no parent info.
		if(!isLeaf)
		{
			uint2 parentInfo = make_uint2(kdData.d_preorderTree[addrNode+1], kdData.d_preorderTree[addrNode+2]);
			
			// Get value and move it to the correct bit position.
			uint posOfBit = 28 + bitNo;
			uint value = d_values[idxNode] & 1; // Ensure 0 or 1.
			value <<= posOfBit;

			// Mask old value to ensure it gets zeroed when or'ing.
			uint masked = parentInfo.x;
			masked &= ~(1 << posOfBit);
			parentInfo.x = masked | value;
			
			// Update parent info.
			kdData.d_preorderTree[addrNode+1] = parentInfo.x;
			kdData.d_preorderTree[addrNode+2] = parentInfo.y;
		}
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////



/// Initializes local kernel data.
extern "C"
void KDInitializeKernels()
{
	// Find out how many thread blocks (grid dimension) we can use on the current device.
	int curDevice;
	mncudaSafeCallNoSync(cudaGetDevice(&curDevice));
	mncudaSafeCallNoSync(cudaGetDeviceProperties(&f_DevProps, curDevice));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KDSetParameters(uint smallNodeMax)
///
/// \brief	Moves kd-tree construction parameters to constant memory.
///
/// \author	Mathias Neumann
/// \date	October 2010
///
/// \param	smallNodeMax	The small node maximum. See ::c_smallNodeMax.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KDSetParameters(uint smallNodeMax)
{
	if(smallNodeMax > KD_SMALLNODEMAX)
		MNFatal("Illegal small node maximum.");
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_smallNodeMax", &smallNodeMax, sizeof(uint)));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_GenChunkAABB() kernel call.
extern "C++"
template <uint numElementPoints>
void KernelKDGenChunkAABB(const KDNodeList& lstActive, KDChunkList& lstChunks)
{
	// Note that we use half the chunk size here. This is a reduction optimization.
	dim3 blockSize = dim3(KD_CHUNKSIZE/2, 1, 1);
	// Avoid the maximum grid size by using two dimensions.
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstChunks.numChunks, f_DevProps.maxGridSize[0]);

	kernel_GenChunkAABB<numElementPoints><<<gridSize, blockSize>>>(lstActive, lstChunks);
	MNCUDA_CHECKERROR;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelKDEmptySpaceCutting(KDNodeList& lstActive, KDFinalNodeList& lstFinal,
/// 	float emptySpaceRatio, uint* d_ioFinalListIndex)
///
/// \brief	Wraps kernel_CanCutOffEmptySpace() and kernel_EmptySpaceCutting() kernel calls.
///
///			Takes care of checking all sides of the node AABBs for empty space.
///
/// \author	Mathias Neumann
/// \date	22.08.2010 
///
/// \param	[in,out]	lstActive		The active node list. Contains the nodes that are to be
/// 									subdivided. When empty space is cut off for some node, its
///										AABB and node level are updated accordingly.
/// \param	[in,out]	lstFinal		The final node list. Will be updated with the generated
///										empty and non-empty nodes. 
/// \param	emptySpaceRatio				The empty space ratio. See ::c_emptySpaceRatio.
/// \param [in,out]	d_ioFinalListIndex	Will contain updated final node list indices for the
///										current active list nodes. That is, the generated non-empty
///										node for the i-th active list node can be found at the index
///										\a d_ioFinalListIndex[i].
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelKDEmptySpaceCutting(KDNodeList& lstActive, KDFinalNodeList& lstFinal, float emptySpaceRatio,
							   uint* d_ioFinalListIndex)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstActive.numNodes, blockSize.x), 1, 1);

	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	MNCudaMemory<uint> d_canCutOff(lstActive.numNodes);
	MNCudaMemory<uint> d_cutOffsets(lstActive.numNodes);

	// Set empty space ratio
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_emptySpaceRatio", &emptySpaceRatio, sizeof(float)));

	for(uint isMax=0; isMax<2; isMax++)
	{
		for(uint axis=0; axis<3; axis++)
		{
			bool bMax = (isMax == 1);
			kernel_CanCutOffEmptySpace<<<gridSize, blockSize>>>(lstActive, axis, bMax, d_canCutOff);
			MNCUDA_CHECKERROR;

			// Scan marks to get offsets.
			cp.Scan(d_canCutOff, lstActive.numNodes, false, d_cutOffsets);

			// Get number of cuts by reduction.
			uint numCuts;
			mncudaReduce(numCuts, (uint*)d_canCutOff, lstActive.numNodes, MNCuda_ADD, (uint)0);

			if(numCuts > 0)
			{
				// Verify we have enough space.
				if(lstFinal.maxNodes < lstFinal.numNodes + 2*numCuts)
					lstFinal.ResizeNodeData(lstFinal.numNodes + 2*numCuts);

				// Perform cut and generate new final list nodes and update active list nodes.
				kernel_EmptySpaceCutting<<<gridSize, blockSize>>>(lstActive, lstFinal, axis, bMax, 
					d_canCutOff, d_cutOffsets, numCuts, d_ioFinalListIndex);
				MNCUDA_CHECKERROR;

				// Update final list node count. It increases by 2*numCuts since we both had to create
				// the empty cut-off node and the tighter node.
				lstFinal.numNodes += 2*numCuts;
			}
			//printf("NUM CUTS: %d.\n", numCuts);
		}
	}
}

/// Wraps kernel_SplitLargeNodes() kernel call.
extern "C"
void KernelKDSplitLargeNodes(const KDNodeList& lstActive, KDNodeList& lstNext)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstActive.numNodes, blockSize.x), 1, 1);

	// Convert next list to internal representation.
	KDNodeListAABB lstNextIn;
	lstNextIn.Initialize(lstNext);

	kernel_SplitLargeNodes<<<gridSize, blockSize>>>(lstActive, lstNextIn);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_UpdateFinalListChildInfo() kernel call.
extern "C"
void KernelKDUpdateFinalListChildInfo(const KDNodeList& lstActive, KDFinalNodeList& lstFinal,
									  uint* d_finalListIndex)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstActive.numNodes, blockSize.x), 1, 1);

	kernel_UpdateFinalListChildInfo<<<gridSize, blockSize>>>(lstActive, lstFinal, d_finalListIndex);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_MarkLeftRightElements() kernel call.
extern "C++"
template <uint numElementPoints>
void KernelKDMarkLeftRightElements(const KDNodeList& lstActive, const KDChunkList& lstChunks, 
						           uint* d_valid)
{
	dim3 blockSize = dim3(KD_CHUNKSIZE, 1, 1);
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstChunks.numChunks, f_DevProps.maxGridSize[0]);

	// Build random number array.
	MNCudaMT& mtw = MNCudaMT::GetInstance();
	uint numRnd = mtw.GetAlignedCount(lstChunks.numChunks*KD_CHUNKSIZE); 
	MNCudaMemory<float> d_randoms(numRnd);
	mtw.Seed(rand());
	mncudaSafeCallNoSync(mtw.Generate(d_randoms, numRnd));

	kernel_MarkLeftRightElements<numElementPoints><<<gridSize, blockSize>>>(lstActive, lstChunks, d_randoms, d_valid);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_MarkSmallNodes() kernel call.
extern "C"
void KernelKDMarkSmallNodes(const KDNodeList& lstNext, uint* d_finalListIndex, uint* d_isSmall, 
						    uint* d_smallRootParent)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstNext.numNodes, blockSize.x), 1, 1);
	kernel_MarkSmallNodes<<<gridSize, blockSize>>>(lstNext, d_finalListIndex, d_isSmall, d_smallRootParent);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_MarkElemsByNodeSize() kernel call.
extern "C"
void KernelKDMarkElemsByNodeSize(const KDChunkList& lstChunks, uint* d_numElemsNext, 
							     uint* d_outIsSmallElem, uint* d_outIsLargeElem)
{
	dim3 blockSize = dim3(KD_CHUNKSIZE, 1, 1);
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstChunks.numChunks, f_DevProps.maxGridSize[0]);
	kernel_MarkElemsByNodeSize<<<gridSize, blockSize>>>(lstChunks, d_numElemsNext, 
		d_outIsSmallElem, d_outIsLargeElem);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_MoveNodes() kernel call.
extern "C"
void KernelKDMoveNodes(const KDNodeList& lstSource, KDNodeList& lstTarget, uint* d_move, uint* d_offsets,
					   bool bTargetIsSmall)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstSource.numNodes, blockSize.x), 1, 1);
	if(bTargetIsSmall)
	{
		// For small nodes we have to move to the tight bounds since for them we make no
		// difference between inherited and tight bounds!
		kernel_MoveNodes<<<gridSize, blockSize>>>(lstSource.numNodes, 
			lstSource.d_aabbMinInherit, lstSource.d_aabbMaxInherit, lstSource.d_nodeLevel,
			lstTarget.numNodes, 
			lstTarget.d_aabbMinTight, lstTarget.d_aabbMaxTight, lstTarget.d_nodeLevel, d_move, d_offsets);
	}
	else
	{
		// For the remaining large nodes tight bounds are calculated, therefore move to
		// inherited bounds.
		kernel_MoveNodes<<<gridSize, blockSize>>>(lstSource.numNodes, 
			lstSource.d_aabbMinInherit, lstSource.d_aabbMaxInherit, lstSource.d_nodeLevel,
			lstTarget.numNodes, 
			lstTarget.d_aabbMinInherit, lstTarget.d_aabbMaxInherit, lstTarget.d_nodeLevel, d_move, d_offsets);
	}
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_UpdateSmallRootParents() kernel call.
extern "C"
void KernelKDUpdateSmallRootParents(const KDFinalNodeList& lstNodes, uint* d_smallRootParents, uint numSmallNodes)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numSmallNodes, blockSize.x), 1, 1);
	kernel_UpdateSmallRootParents<<<gridSize, blockSize>>>(lstNodes, d_smallRootParents, numSmallNodes);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_CreateSplitCandidates() kernel call.
extern "C++"
template <uint numElementPoints>
void KernelKDCreateSplitCandidates(const KDNodeList& lstSmall, KDSplitList& lstSplit)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstSmall.numNodes, blockSize.x), 1, 1);

	kernel_CreateSplitCandidates<numElementPoints><<<gridSize, blockSize>>>(lstSmall, lstSplit);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_InitSplitMasks() kernel calls.
extern "C++"
template<uint numElementPoints>
void KernelKDInitSplitMasks(const KDNodeList& lstSmall, uint smallNodeMax, KDSplitList& lstSplit)
{
	dim3 blockSize = dim3(3*smallNodeMax, 1, 1);
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstSmall.numNodes, f_DevProps.maxGridSize[0]);

	// Build random number array.
	MNCudaMT& mtw = MNCudaMT::GetInstance();
	uint numRnd = mtw.GetAlignedCount(lstSmall.nextFreePos); 
	MNCudaMemory<float> d_randoms(numRnd);
	mtw.Seed(rand());
	mncudaSafeCallNoSync(mtw.Generate(d_randoms, numRnd));

	// Minimums / Single point
	kernel_InitSplitMasks<0, numElementPoints><<<gridSize, blockSize>>>(lstSmall, d_randoms, lstSplit);
	MNCUDA_CHECKERROR;
	if(numElementPoints == 2)
	{
		// Maximums
		kernel_InitSplitMasks<1, numElementPoints><<<gridSize, blockSize>>>(lstSmall, d_randoms, lstSplit);
		MNCUDA_CHECKERROR;
	}
}

/// Wraps kernel_FindBestSplits() kernel call.
extern "C++"
template<uint numElementPoints>
void KernelKDFindBestSplits(const KDNodeList& lstActive, const KDSplitList& lstSplit, float maxQueryRadius,
						    uint* d_outBestSplit, float* d_outSplitCost)
{
	// We use KD_SMALLNODEMAX*6/3 = 384/3 = 128 here for accelerated reduction.
	dim3 blockSize = dim3(128, 1, 1);
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstActive.numNodes, f_DevProps.maxGridSize[0]);

	// Set maximum query radius.
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_maxQueryRadius", &maxQueryRadius, sizeof(float)));

	kernel_FindBestSplits<numElementPoints><<<gridSize, blockSize>>>(
		lstActive, lstSplit, d_outBestSplit, d_outSplitCost);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_SplitSmallNodes() kernel call.
extern "C"
void KernelKDSplitSmallNodes(const KDNodeList& lstActive, const KDSplitList& lstSplit, const KDNodeList& lstNext,
						     uint* d_inBestSplit, float* d_inSplitCost, uint* d_outIsSplit)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstActive.numNodes, blockSize.x), 1, 1);

	// Convert next list to internal representation.
	KDNodeListSmall lstNextIn;
	lstNextIn.numNodes = lstNext.numNodes;
	lstNextIn.d_numElems = lstNext.d_numElems;
	lstNextIn.d_nodeLevel = lstNext.d_nodeLevel;
	lstNextIn.d_aabbMinTight = lstNext.d_aabbMinTight;
	lstNextIn.d_aabbMaxTight = lstNext.d_aabbMaxTight;
	lstNextIn.d_splitAxis = lstNext.d_splitAxis;
	lstNextIn.d_splitPos = lstNext.d_splitPos;
	lstNextIn.d_idxSmallRoot = lstNext.d_idxSmallRoot;
	lstNextIn.d_elemMask = lstNext.d_elemMask;

	KDNodeListSmall lstActiveIn;
	lstActiveIn.numNodes = lstActive.numNodes;
	lstActiveIn.d_numElems = lstActive.d_numElems;
	lstActiveIn.d_nodeLevel = lstActive.d_nodeLevel;
	lstActiveIn.d_aabbMinTight = lstActive.d_aabbMinTight;
	lstActiveIn.d_aabbMaxTight = lstActive.d_aabbMaxTight;
	lstActiveIn.d_splitAxis = lstActive.d_splitAxis;
	lstActiveIn.d_splitPos = lstActive.d_splitPos;
	lstActiveIn.d_idxSmallRoot = lstActive.d_idxSmallRoot;
	lstActiveIn.d_elemMask = lstActive.d_elemMask;

	kernel_SplitSmallNodes<<<gridSize, blockSize>>>(lstActiveIn, lstSplit, lstNextIn, 
		d_inBestSplit, d_inSplitCost, d_outIsSplit);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_GenerateENAFromMasks() kernel call.
extern "C"
void KernelKDGenerateENAFromMasks(KDNodeList& lstActive, const KDNodeList& lstSmallRoots)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstActive.numNodes, blockSize.x), 1, 1);

	// Convert next list to internal representation.
	KDNodeListENA lstSmallRootsIn;
	lstSmallRootsIn.numNodes = lstSmallRoots.numNodes;
	lstSmallRootsIn.d_numElems = lstSmallRoots.d_numElems;
	lstSmallRootsIn.d_idxFirstElem = lstSmallRoots.d_idxFirstElem;
	lstSmallRootsIn.d_idxSmallRoot = lstSmallRoots.d_idxSmallRoot;
	lstSmallRootsIn.d_elemMask = lstSmallRoots.d_elemMask;
	lstSmallRootsIn.d_elemNodeAssoc = lstSmallRoots.d_elemNodeAssoc;

	KDNodeListENA lstActiveIn;
	lstActiveIn.numNodes = lstActive.numNodes;
	lstActiveIn.d_numElems = lstActive.d_numElems;
	lstActiveIn.d_idxFirstElem = lstActive.d_idxFirstElem;
	lstActiveIn.d_idxSmallRoot = lstActive.d_idxSmallRoot;
	lstActiveIn.d_elemMask = lstActive.d_elemMask;
	lstActiveIn.d_elemNodeAssoc = lstActive.d_elemNodeAssoc;

	kernel_GenerateENAFromMasks<<<gridSize, blockSize>>>(lstActiveIn, lstSmallRootsIn);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_TraversalUpPath() kernel call.
extern "C"
void KernelKDTraversalUpPath(const KDFinalNodeList& lstFinal, uint curLevel, uint* d_sizes)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);
	kernel_TraversalUpPath<<<gridSize, blockSize>>>(lstFinal, curLevel, d_sizes);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_TraversalDownPath() kernel call.
extern "C"
void KernelKDTraversalDownPath(const KDFinalNodeList& lstFinal, uint curLevel, uint* d_sizes,
							   uint* d_addresses, KDTreeData& kdData)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);
	kernel_TraversalDownPath<<<gridSize, blockSize>>>(lstFinal, curLevel, d_sizes, d_addresses, kdData);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_TestNodeList() kernel call.
extern "C++"
template <uint numElementPoints>
void KernelKDTestNodeList(const KDNodeList& lstNodes, uint* d_valid, float3 rootMin, float3 rootMax,
						  bool useTightBounds)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstNodes.numNodes, blockSize.x), 1, 1);

	uint temp = 1;
	mncudaSafeCallNoSync(cudaMemcpy(d_valid, &temp, sizeof(uint), cudaMemcpyHostToDevice));

	kernel_TestNodeList<numElementPoints><<<gridSize, blockSize>>>(lstNodes, d_valid, rootMin, rootMax,
		useTightBounds);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_SetCustomBit() kernel call.
extern "C"
void KernelKDSetCustomBit(const KDTreeData& kdData, uint bitNo, uint* d_values)
{
	MNAssert(bitNo < 2 && d_values);

	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(kdData.numNodes, blockSize.x), 1, 1);

	kernel_SetCustomBit<<<gridSize, blockSize>>>(kdData, bitNo, d_values);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////



#ifndef DOXYGEN_IGNORE

// Explicit template declarations to avoid linker errors. This is no problem here since we only have
// two possibilities for the template param: 1 for points and 2 for AABB min/max.
extern "C++" template void KernelKDGenChunkAABB<1>(const KDNodeList& lstActive, KDChunkList& lstChunks);
extern "C++" template void KernelKDGenChunkAABB<2>(const KDNodeList& lstActive, KDChunkList& lstChunks);
extern "C++" template void KernelKDMarkLeftRightElements<1>(const KDNodeList& lstActive, const KDChunkList& lstChunks, 
											 uint* d_valid);
extern "C++" template void KernelKDMarkLeftRightElements<2>(const KDNodeList& lstActive, const KDChunkList& lstChunks, 
											 uint* d_valid);
extern "C++" template void KernelKDTestNodeList<1>(const KDNodeList& lstNodes, uint* d_valid, float3 rootMin, float3 rootMax, bool useTightBounds);
extern "C++" template void KernelKDTestNodeList<2>(const KDNodeList& lstNodes, uint* d_valid, float3 rootMin, float3 rootMax, bool useTightBounds);
extern "C++" template void KernelKDCreateSplitCandidates<1>(const KDNodeList& lstSmall, KDSplitList& lstSplit);
extern "C++" template void KernelKDCreateSplitCandidates<2>(const KDNodeList& lstSmall, KDSplitList& lstSplit);
extern "C++" template void KernelKDInitSplitMasks<1>(const KDNodeList& lstSmall, uint smallNodeMax, KDSplitList& lstSplit);
extern "C++" template void KernelKDInitSplitMasks<2>(const KDNodeList& lstSmall, uint smallNodeMax, KDSplitList& lstSplit);
extern "C++" template void KernelKDFindBestSplits<1>(const KDNodeList& lstActive, const KDSplitList& lstSplit, 
												     float maxQueryRadius, uint* d_outBestSplit, float* d_outSplitCost);
extern "C++" template void KernelKDFindBestSplits<2>(const KDNodeList& lstActive, const KDSplitList& lstSplit, 
												     float maxQueryRadius, uint* d_outBestSplit, float* d_outSplitCost);

#endif // DOXYGEN_IGNORE