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

#include "KDTreeTri.h"
#include "../KernelDefs.h"
#include "../MNUtilities.h"
#include "../MNCudaUtil.h"

// kdtree_tri.cu
extern "C"
void KernelKDGenerateTriAABBs(const KDNodeList& lstRoot, const TriangleData& td);
extern "C"
void KernelKDPerformSplitClipping(const KDNodeList& lstActive, const KDNodeList& lstNext,
								  const KDChunkList& lstChunks, const TriangleData& td);


KDTreeTri::KDTreeTri(const TriangleData& td)
	: KDTreeGPU(td.numTris, 2, td.aabbMin, td.aabbMax)
{
	// Store pointer to data source.
	m_pTD = &td;

	SetEmptySpaceRatio(0.25f); // Zhou et al. proposed 0.25f
	SetSmallNodeMax(64);
}

KDTreeTri::~KDTreeTri(void)
{
}

void KDTreeTri::AddRootNode(KDNodeList* pList)
{
	mncudaSafeCallNoSync(cudaMemset(pList->d_idxFirstElem, 0, sizeof(uint)));
	mncudaSafeCallNoSync(cudaMemset(pList->d_nodeLevel, 0, sizeof(uint)));
	uint tmp = m_pTD->numTris;
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_numElems, &tmp, sizeof(uint), cudaMemcpyHostToDevice));

	// Set inherited bounds to scene bounds.
	float4 aabb = make_float4(m_rootAABBMin);
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_aabbMinInherit, &aabb, sizeof(float4), cudaMemcpyHostToDevice));
	aabb = make_float4(m_rootAABBMax);
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_aabbMaxInherit, &aabb, sizeof(float4), cudaMemcpyHostToDevice));

	// All elements are contained in the first node, therefore the list is just the identity relation.
	mncudaInitIdentity(pList->d_elemNodeAssoc, m_pTD->numTris);

	pList->numNodes = 1;
	// Align first free tri index.
	pList->nextFreePos = MNCUDA_ALIGN(m_pTD->numTris);

	// Compute AABBs for all triangles in root node in parallel. This initializes
	// the d_elemPoint1/2 members of pList.
	KernelKDGenerateTriAABBs(*pList, *m_pTD);
}

void KDTreeTri::PerformSplitClipping(KDNodeList* pListParent, KDNodeList* pListChild)
{
	// Construct a chunk list for the child list.
	CreateChunkList(pListChild);

	KernelKDPerformSplitClipping(*pListParent, *pListChild, *m_pChunkList, *m_pTD);
}