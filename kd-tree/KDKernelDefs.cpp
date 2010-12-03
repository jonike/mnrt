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

#include "KDKernelDefs.h"
#include "../MNCudaUtil.h"
#include "../MNCudaMemPool.h"
#include "../MNStatContainer.h"

void KDNodeList::Initialize(uint _maxNodes, uint _maxElems, uint _numElementPoints/* = 2*/)
{
	MNAssert(_maxNodes > 0 && _maxElems > 0 && _numElementPoints <= 2 && _numElementPoints > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Starting with zero nodes.
	numNodes = 0;
	maxNodes = MNCUDA_ALIGN(_maxNodes);
	nextFreePos = 0;
	// Ensure aligned access (16 * 4 byte = 64 byte alignment) to get coalesced access in kernels.
	maxElems = MNCUDA_ALIGN(_maxElems);
	numElementPoints = std::max((uint)1, std::min((uint)2, _numElementPoints));

	// Node sized.
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxFirstElem, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_numElems, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_nodeLevel, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_splitAxis, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_splitPos, maxNodes*sizeof(float), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_childLeft, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_childRight, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMinTight, maxNodes*sizeof(float4), "kd-tree node", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMaxTight, maxNodes*sizeof(float4), "kd-tree node", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMinInherit, maxNodes*sizeof(float4), "kd-tree node", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMaxInherit, maxNodes*sizeof(float4), "kd-tree node", 256));

	// Small node stuff.
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxSmallRoot, maxNodes*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_elemMask, maxNodes*sizeof(ElementMask), "kd-tree node", 128));

	// Element sized.
	mncudaSafeCallNoSync(pool.Request((void**)&d_elemNodeAssoc, maxElems*sizeof(uint), "kd-tree node"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_elemPoint1, maxElems*sizeof(float4), "kd-tree node"));
	if(numElementPoints > 1)
		mncudaSafeCallNoSync(pool.Request((void**)&d_elemPoint2, maxElems*sizeof(float4), "kd-tree node"));
	
}

void KDNodeList::AppendList(KDNodeList* pList, bool appendENA)
{
	MNAssert(pList);
	// Resize node data if required.
	if(numNodes + pList->numNodes > maxNodes)
		ResizeNodeData(numNodes + pList->numNodes);

	// Copy data.
	mncudaSafeCallNoSync(cudaMemcpy(d_idxFirstElem + numNodes, pList->d_idxFirstElem, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_numElems + numNodes, pList->d_numElems, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_nodeLevel + numNodes, pList->d_nodeLevel, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));

	// Thight
	mncudaSafeCallNoSync(cudaMemcpy(d_aabbMinTight + numNodes, pList->d_aabbMinTight, 
		pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_aabbMaxTight + numNodes, pList->d_aabbMaxTight, 
		pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
	// Inherit
	mncudaSafeCallNoSync(cudaMemcpy(d_aabbMinInherit + numNodes, pList->d_aabbMinInherit, 
		pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_aabbMaxInherit + numNodes, pList->d_aabbMaxInherit, 
		pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));

	// Copy split information.
	mncudaSafeCallNoSync(cudaMemcpy(d_splitAxis + numNodes, pList->d_splitAxis, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_splitPos + numNodes, pList->d_splitPos, 
		pList->numNodes*sizeof(float), cudaMemcpyDeviceToDevice));

	// Copy child relationship data. Need to update this after that cause the child relationship
	// indices are relative to the next node list's indices.
	mncudaSafeCallNoSync(cudaMemcpy(d_childLeft + numNodes, pList->d_childLeft, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaConstantOp<MNCuda_ADD, uint>(d_childLeft + numNodes, pList->numNodes, numNodes + pList->numNodes);
	mncudaSafeCallNoSync(cudaMemcpy(d_childRight + numNodes, pList->d_childRight, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaConstantOp<MNCuda_ADD, uint>(d_childRight + numNodes, pList->numNodes, numNodes + pList->numNodes);

	// Copy element masks and small root indices.
	mncudaSafeCallNoSync(cudaMemcpy(d_elemMask + numNodes, pList->d_elemMask, 
		pList->numNodes*sizeof(ElementMask), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_idxSmallRoot + numNodes, pList->d_idxSmallRoot, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));

	if(appendENA)
	{
		// Check if there's still enough space in the ENA.
		if(nextFreePos + pList->nextFreePos > maxElems)
			ResizeElementData(nextFreePos + pList->nextFreePos);

		// Copy in new other ENA.
		mncudaSafeCallNoSync(cudaMemcpy(d_elemNodeAssoc + nextFreePos, pList->d_elemNodeAssoc, 
			pList->nextFreePos*sizeof(uint), cudaMemcpyDeviceToDevice));

		// Copy in element points.
		mncudaSafeCallNoSync(cudaMemcpy(d_elemPoint1 + nextFreePos, pList->d_elemPoint1, 
			pList->nextFreePos*sizeof(float4), cudaMemcpyDeviceToDevice));
		if(numElementPoints == 2)
			mncudaSafeCallNoSync(cudaMemcpy(d_elemPoint2 + nextFreePos, pList->d_elemPoint2, 
				pList->nextFreePos*sizeof(float4), cudaMemcpyDeviceToDevice));
	
		// Shift first element indices in d_idxFirstElem for new nodes.
		if(nextFreePos != 0)
			mncudaConstantOp<MNCuda_ADD, uint>(d_idxFirstElem + numNodes, pList->numNodes, nextFreePos);
	}

	// Now update counts.
	if(appendENA)
		nextFreePos += pList->nextFreePos;
	numNodes += pList->numNodes;
}

void KDNodeList::ResizeNodeData(uint required)
{
	MNAssert(required > maxNodes);

	// Add some space to avoid multiple resizes.
	uint newMax = std::max(2*maxNodes, required);

	// newMax should stay unchanged after first operation.
	newMax = mncudaResizeMNCudaMem(&d_idxFirstElem, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_numElems, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_nodeLevel, maxNodes, newMax);

	mncudaResizeMNCudaMem(&d_aabbMinTight, maxElems, newMax);
	mncudaResizeMNCudaMem(&d_aabbMaxTight, maxElems, newMax);
	mncudaResizeMNCudaMem(&d_aabbMinInherit, maxElems, newMax);
	mncudaResizeMNCudaMem(&d_aabbMaxInherit, maxElems, newMax);

	mncudaResizeMNCudaMem(&d_splitAxis, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_splitPos, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_childLeft, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_childRight, maxNodes, newMax);

	mncudaResizeMNCudaMem(&d_idxSmallRoot, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_elemMask, maxNodes, newMax);

	maxNodes = newMax;
	static StatCounter& ctrNodeListResize = StatCounter::Create("KD-Tree Building", "Node list resizes");
	ctrNodeListResize += 1;
}

void KDNodeList::ResizeElementData(uint required)
{
	MNAssert(required > maxElems);

	// Add some space to avoid multiple resizes.
	uint newMax = std::max(2*maxElems, required);
	
	// Element count might change due to alignment.
	newMax = mncudaResizeMNCudaMem(&d_elemNodeAssoc, maxElems, newMax);
	mncudaResizeMNCudaMem(&d_elemPoint1, maxElems, newMax);
	if(numElementPoints > 1)
		mncudaResizeMNCudaMem(&d_elemPoint2, maxElems, newMax);

	maxElems = newMax;
	static StatCounter& ctrNodeListElResize = StatCounter::Create("KD-Tree Building", "Node list resizes (elements)");
	ctrNodeListElResize += 1;
}

void KDNodeList::Clear()
{
	numNodes = 0;
	nextFreePos = 0;
}

void KDNodeList::Free()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	mncudaSafeCallNoSync(pool.Release(d_idxFirstElem));
	mncudaSafeCallNoSync(pool.Release(d_numElems));
	mncudaSafeCallNoSync(pool.Release(d_nodeLevel));

	mncudaSafeCallNoSync(pool.Release(d_splitAxis));
	mncudaSafeCallNoSync(pool.Release(d_splitPos));
	mncudaSafeCallNoSync(pool.Release(d_childLeft));
	mncudaSafeCallNoSync(pool.Release(d_childRight));

	mncudaSafeCallNoSync(pool.Release(d_aabbMinTight));
	mncudaSafeCallNoSync(pool.Release(d_aabbMaxTight));
	mncudaSafeCallNoSync(pool.Release(d_aabbMinInherit));
	mncudaSafeCallNoSync(pool.Release(d_aabbMaxInherit));

	mncudaSafeCallNoSync(pool.Release(d_idxSmallRoot));
	mncudaSafeCallNoSync(pool.Release(d_elemMask));

	mncudaSafeCallNoSync(pool.Release(d_elemNodeAssoc));
	mncudaSafeCallNoSync(pool.Release(d_elemPoint1));
	if(numElementPoints > 1)
		mncudaSafeCallNoSync(pool.Release(d_elemPoint2));

	maxNodes = 0;
	maxElems = 0;
	numNodes = 0;
}

void KDFinalNodeList::Initialize(uint _maxNodes, uint _maxElems)
{
	MNAssert(_maxNodes > 0 && _maxElems > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Starting with zero nodes.
	numNodes = 0;
	maxNodes = MNCUDA_ALIGN(_maxNodes);
	nextFreePos = 0;
	// Ensure alignment to get coalesced access in kernels.
	maxElems = MNCUDA_ALIGN(_maxElems);

	// Node sized.
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxFirstElem, maxNodes*sizeof(uint), "kd-tree final"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_numElems, maxNodes*sizeof(uint), "kd-tree final"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_nodeLevel, maxNodes*sizeof(uint), "kd-tree final"));

	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMin, maxNodes*sizeof(float4), "kd-tree final", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMax, maxNodes*sizeof(float4), "kd-tree final", 256));

	mncudaSafeCallNoSync(pool.Request((void**)&d_splitAxis, maxNodes*sizeof(uint), "kd-tree final"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_splitPos, maxNodes*sizeof(float), "kd-tree final"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_childLeft, maxNodes*sizeof(uint), "kd-tree final"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_childRight, maxNodes*sizeof(uint), "kd-tree final"));

	// Element sized.
	mncudaSafeCallNoSync(pool.Request((void**)&d_elemNodeAssoc, maxElems*sizeof(uint), "kd-tree final"));	
}

void KDFinalNodeList::AppendList(KDNodeList* pList, bool appendENA, bool hasInheritedBounds)
{
	MNAssert(pList);
	// Resize node data if required.
	if(numNodes + pList->numNodes > maxNodes)
		ResizeNodeData(numNodes + pList->numNodes);

	// Copy data. Ignore child relationship data.
	mncudaSafeCallNoSync(cudaMemcpy(d_idxFirstElem + numNodes, pList->d_idxFirstElem, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_numElems + numNodes, pList->d_numElems, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_nodeLevel + numNodes, pList->d_nodeLevel, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));

	if(hasInheritedBounds)
	{
		mncudaSafeCallNoSync(cudaMemcpy(d_aabbMin + numNodes, pList->d_aabbMinInherit, 
			pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
		mncudaSafeCallNoSync(cudaMemcpy(d_aabbMax + numNodes, pList->d_aabbMaxInherit, 
			pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
	}
	else
	{
		mncudaSafeCallNoSync(cudaMemcpy(d_aabbMin + numNodes, pList->d_aabbMinTight, 
			pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
		mncudaSafeCallNoSync(cudaMemcpy(d_aabbMax + numNodes, pList->d_aabbMaxTight, 
			pList->numNodes*sizeof(float4), cudaMemcpyDeviceToDevice));
	}

	// Copy split information.
	mncudaSafeCallNoSync(cudaMemcpy(d_splitAxis + numNodes, pList->d_splitAxis, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_splitPos + numNodes, pList->d_splitPos, 
		pList->numNodes*sizeof(float), cudaMemcpyDeviceToDevice));

	// Copy child relationship data. Need to update this after that cause the child relationship
	// indices are relative to the next node list's indices.
	mncudaSafeCallNoSync(cudaMemcpy(d_childLeft + numNodes, pList->d_childLeft, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaConstantOp<MNCuda_ADD, uint>(d_childLeft + numNodes, pList->numNodes, numNodes + pList->numNodes);
	mncudaSafeCallNoSync(cudaMemcpy(d_childRight + numNodes, pList->d_childRight, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaConstantOp<MNCuda_ADD, uint>(d_childRight + numNodes, pList->numNodes, numNodes + pList->numNodes);

	if(appendENA)
	{
		// Check if there's still enough space in the ENA.
		if(nextFreePos + pList->nextFreePos > maxElems)
			ResizeElementData(nextFreePos + pList->nextFreePos);

		// Copy in new other ENA.
		mncudaSafeCallNoSync(cudaMemcpy(d_elemNodeAssoc + nextFreePos, pList->d_elemNodeAssoc, 
			pList->nextFreePos*sizeof(uint), cudaMemcpyDeviceToDevice));
	
		// Shift first element indices in d_idxFirstElem for new nodes.
		if(nextFreePos != 0)
			mncudaConstantOp<MNCuda_ADD, uint>(d_idxFirstElem + numNodes, pList->numNodes, nextFreePos);
	}

	// Now update counts.
	if(appendENA)
		nextFreePos += pList->nextFreePos;
	numNodes += pList->numNodes;
}

void KDFinalNodeList::ResizeNodeData(uint required)
{
	MNAssert(required > maxNodes);

	// Add some space to avoid multiple resizes.
	uint newMax = std::max(2*maxNodes, required);

	// newMax should stay unchanged after first operation.
	newMax = mncudaResizeMNCudaMem(&d_idxFirstElem, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_numElems, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_nodeLevel, maxNodes, newMax);

	mncudaResizeMNCudaMem(&d_aabbMin, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_aabbMax, maxNodes, newMax);

	mncudaResizeMNCudaMem(&d_splitAxis, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_splitPos, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_childLeft, maxNodes, newMax);
	mncudaResizeMNCudaMem(&d_childRight, maxNodes, newMax);

	maxNodes = newMax;
	static StatCounter& ctrFNodeListResize = StatCounter::Create("KD-Tree Building", "Final node list resizes");
	ctrFNodeListResize += 1;
}

void KDFinalNodeList::ResizeElementData(uint required)
{
	MNAssert(required > maxElems);

	// Add some space to avoid multiple resizes.
	uint newMax = std::max(2*maxElems, required);
	
	// Element count might change due to alignment.
	maxElems = mncudaResizeMNCudaMem(&d_elemNodeAssoc, maxElems, newMax);

	static StatCounter& ctrFNodeListElResize = StatCounter::Create("KD-Tree Building", "Final node list resizes (elements)");
	ctrFNodeListElResize += 1;
}

void KDFinalNodeList::Clear()
{
	numNodes = 0;
	nextFreePos = 0;
}

void KDFinalNodeList::Free()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	mncudaSafeCallNoSync(pool.Release(d_idxFirstElem));
	mncudaSafeCallNoSync(pool.Release(d_numElems));
	mncudaSafeCallNoSync(pool.Release(d_nodeLevel));

	mncudaSafeCallNoSync(pool.Release(d_aabbMin));
	mncudaSafeCallNoSync(pool.Release(d_aabbMax));

	mncudaSafeCallNoSync(pool.Release(d_splitAxis));
	mncudaSafeCallNoSync(pool.Release(d_splitPos));
	mncudaSafeCallNoSync(pool.Release(d_childLeft));
	mncudaSafeCallNoSync(pool.Release(d_childRight));

	mncudaSafeCallNoSync(pool.Release(d_elemNodeAssoc));

	maxNodes = 0;
	maxElems = 0;
	numNodes = 0;
}



void KDChunkList::Initialize(uint _maxChunks)
{
	MNAssert(_maxChunks > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Start with 0 chunks.
	numChunks = 0;
	maxChunks = MNCUDA_ALIGN(_maxChunks);

	mncudaSafeCallNoSync(pool.Request((void**)&d_idxNode, maxChunks*sizeof(uint), "kd-tree misc"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxFirstElem, maxChunks*sizeof(uint), "kd-tree misc"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_numElems, maxChunks*sizeof(uint), "kd-tree misc"));

	// AABB minimum point.
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMin, maxChunks*sizeof(float4), "kd-tree misc", 256));
	// AABB maximum point.
	mncudaSafeCallNoSync(pool.Request((void**)&d_aabbMax, maxChunks*sizeof(float4), "kd-tree misc", 256));
}

void KDChunkList::Clear()
{
	numChunks = 0;
}

void KDChunkList::Free()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_idxNode));
	mncudaSafeCallNoSync(pool.Release(d_idxFirstElem));
	mncudaSafeCallNoSync(pool.Release(d_numElems));

	mncudaSafeCallNoSync(pool.Release(d_aabbMin));
	mncudaSafeCallNoSync(pool.Release(d_aabbMax));

	numChunks = 0;
	maxChunks = 0;
}


void KDSplitList::Initialize(KDNodeList* pSmallRoots)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Having a fixed size here.
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxFirstSplit, pSmallRoots->numNodes*sizeof(uint), "kd-tree misc"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_numSplits, pSmallRoots->numNodes*sizeof(uint), "kd-tree misc"));

	// Allocated later when size is known.
	d_splitPos = NULL;
	d_splitInfo = NULL;
	d_maskLeft = NULL;
	d_maskRight = NULL;
}

void KDSplitList::Free()
{
	if(!d_splitPos)
		return;
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	mncudaSafeCallNoSync(pool.Release(d_idxFirstSplit));
	mncudaSafeCallNoSync(pool.Release(d_numSplits));

	mncudaSafeCallNoSync(pool.Release(d_splitPos));
	mncudaSafeCallNoSync(pool.Release(d_splitInfo));
	mncudaSafeCallNoSync(pool.Release(d_maskLeft));
	mncudaSafeCallNoSync(pool.Release(d_maskRight));
}

void KDTreeData::Initialize(KDFinalNodeList* pList, float3 aabbMin, float3 aabbMax)
{
	MNAssert(pList);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	numNodes = pList->numNodes;

	// Set root bounds. Read them from root node's tight bounds.
	aabbRootMin = aabbMin;
	aabbRootMax = aabbMax;

	// Use texture request here since we use the data for linear texture memory and therefore
	// need special alignment.
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_numElems, numNodes*sizeof(uint), "kd-tree result"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeAddresses, numNodes*sizeof(uint), "kd-tree result"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeExtent, numNodes*sizeof(float4), "kd-tree result"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_childLeft, numNodes*sizeof(uint), "kd-tree result"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_childRight, numNodes*sizeof(uint), "kd-tree result"));

	// Copy in element counts and left/right indices. Other data is initialized later.
	mncudaSafeCallNoSync(cudaMemcpy(d_numElems, pList->d_numElems, 
		pList->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_childLeft, pList->d_childLeft, 
		pList->numNodes*sizeof(float), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_childRight, pList->d_childRight, 
		pList->numNodes*sizeof(float), cudaMemcpyDeviceToDevice));

	// Size unknown yet.
	sizeTree = 0;
	d_preorderTree = NULL;
}

void KDTreeData::Free()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	mncudaSafeCallNoSync(pool.Release(d_numElems));
	mncudaSafeCallNoSync(pool.Release(d_nodeAddresses));
	mncudaSafeCallNoSync(pool.Release(d_nodeExtent));
	mncudaSafeCallNoSync(pool.Release(d_childLeft));
	mncudaSafeCallNoSync(pool.Release(d_childRight));
	if(d_preorderTree)
		mncudaSafeCallNoSync(pool.Release(d_preorderTree));

	sizeTree = 0;
	numNodes = 0;
}