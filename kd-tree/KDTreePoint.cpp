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

#include "KDTreePoint.h"
#include "../MNCudaUtil.h"
#include "../MNStatContainer.h"

// Is multiplied with the maximum query radius to get a maximum radius for query radius
// estimation nodes, see KernelKDMarkEstimationNodes().
#define QRE_ALPHA		0.5f


extern "C"
void KDSetTreePoint(const KDTreeData& kdTree, float4* d_points, uint numPoints,
					uint knnRefineIters, uint knnTargetCount);
extern "C"
void KDUnsetTreePoint();
extern "C"
void KernelKDMarkEstimationNodes(const KDTreeData& pmap, float maxNodeRadius, uint* d_outIsQualified);
extern "C"
void KernelKDComputeNodeQR(uint* d_idxNode, float4* d_nodeExtent, uint numNodes, float globalQR,
						   float* d_ioNodeRadiusEstimate);extern "C"
void KernelKDEstimateRadii(float4* d_queryPoints, uint numQueryPoints, float* d_nodeRadiusEstimate,
						   float4* d_nodeExtents,
						   float globalQR, float* d_outRadiusEstimate);
extern "C"
void KernelKDRefineRadiusZhou(float4* q_queryPoints, uint numQueryPoints, float* d_ioQueryRadius);



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	WorkList
///
/// \brief	Work list structure to hold kd-tree nodes currently worked on within node query
/// 		radius precomputation. 
///
/// \author	Mathias Neumann
/// \date	08.07.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class WorkList
{
public:
	WorkList(uint maxNodes);
	~WorkList();
public:
	// Number of nodes stored.
	uint numNodes;

	// Index of the node.
	uint* d_idxNode;
	// Node extent (center + radius)
	float4* d_nodeExtent;

public:
	// Initializes the work list's memory.
	// Fills all nodes that are marked (1) in d_isWorkListEntry from the given kd-tree data
	// into this work list.
	void Fill(const KDTreeData& kdTree, uint* d_isWorkListEntry);
};

WorkList::WorkList(uint maxNodes)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_idxNode, maxNodes*sizeof(uint), "KNN radius est."));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeExtent, maxNodes*sizeof(float4), "KNN radius est."));
}

WorkList::~WorkList()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_idxNode));
	mncudaSafeCallNoSync(pool.Release(d_nodeExtent));
	numNodes = 0;
}


void WorkList::Fill(const KDTreeData& kdTree, uint* d_isWorkListEntry)
{
	numNodes = mncudaGenCompactAddresses(d_isWorkListEntry, kdTree.numNodes, d_idxNode);
	if(numNodes == 0)
		return;

	mncudaSetFromAddress(d_nodeExtent, d_idxNode, kdTree.d_nodeExtent, numNodes);
}






KDTreePoint::KDTreePoint(float4* _d_points, uint numPoints, 
						 float3 sceneAABBMin, float3 sceneAABBMax, float maxQueryRadius)
	: KDTreeGPU(numPoints, 1, sceneAABBMin, sceneAABBMax),
	  d_points(_d_points),
	  m_numPoints(numPoints)
{
	SetEmptySpaceRatio(0.1f); // Zhou proposed 0.1f
	// NOTE: Picking this lower than the value for triangle kd-trees helps with avoiding too large
	//       kd-tree nodes.
	SetSmallNodeMax(32);
	SetMaxQueryRadius(maxQueryRadius);
	m_knnRefineIters = 2;
	m_knnTargetCount = 5;
	d_nodeRadiusEstimate = NULL;
}

KDTreePoint::~KDTreePoint(void)
{
}

void KDTreePoint::AddRootNode(KDNodeList* pList)
{
	mncudaSafeCallNoSync(cudaMemset(pList->d_idxFirstElem, 0, sizeof(uint)));
	mncudaSafeCallNoSync(cudaMemset(pList->d_nodeLevel, 0, sizeof(uint)));
	uint tmp = m_numPoints;
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_numElems, &tmp, sizeof(uint), cudaMemcpyHostToDevice));

	// Set inherited bounds to scene bounds.
	float4 aabb = make_float4(m_rootAABBMin);
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_aabbMinInherit, &aabb, sizeof(float4), cudaMemcpyHostToDevice));
	aabb = make_float4(m_rootAABBMax);
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_aabbMaxInherit, &aabb, sizeof(float4), cudaMemcpyHostToDevice));

	// All elements are contained in the first node, therefore the list is just the identity relation.
	mncudaInitIdentity(pList->d_elemNodeAssoc, m_numPoints);

	// Need to initialize d_elemPoint1 member of pList using the our point list.
	// This can be done by just copying the data.
	mncudaSafeCallNoSync(cudaMemcpy(pList->d_elemPoint1, d_points, 
		m_numPoints*sizeof(float4), cudaMemcpyDeviceToDevice));

	pList->numNodes = 1;
	// Align first free tri index.
	pList->nextFreePos = MNCUDA_ALIGN(m_numPoints);
}

void KDTreePoint::Destroy()
{
	if(d_nodeRadiusEstimate)
	{
		MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
		mncudaSafeCallNoSync(pool.Release(d_nodeRadiusEstimate));
		d_nodeRadiusEstimate = NULL;
	}

	KDTreeGPU::Destroy();
}

void KDTreePoint::PostBuild()
{
	// Performs cleanup (Important!).
	KDTreeGPU::PostBuild();

	// Precompute a query radius for some nodes in the kd-tree.
	PrecomputeQueryRadii();
}

void KDTreePoint::PrecomputeQueryRadii()
{
	static StatTimer& s_timer = StatTimer::Create("Timers", "kd-tree: node query radius estimation", false);
	mncudaSafeCallNoSync(s_timer.Start(true));
	KDTreeData* pKDData = GetData();

	// Initialize radius estimate with infinity.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeRadiusEstimate, 
		pKDData->numNodes*sizeof(float), "KNN radius est."));
	mncudaInitConstant(d_nodeRadiusEstimate, pKDData->numNodes, MN_INFINITY);

	// Avoid estimating the radius when we have too few kd-tree nodes.
	if(pKDData->numNodes < 1000)
	{
		mncudaSafeCallNoSync(s_timer.Stop(true));
		return;
	}

	// Mark nodes for which to perform query radius estimation.
	MNCudaMemory<uint> d_isQualified(pKDData->numNodes);
	KernelKDMarkEstimationNodes(*pKDData, QRE_ALPHA*GetMaxQueryRadius(), d_isQualified);

	// Fill work list with initial nodes.
	WorkList* pWorkList = new WorkList(pKDData->numNodes);
	pWorkList->Fill(*pKDData, d_isQualified);
	//printf("NUM QNODE EST NODES: %d.\n", pWorkList->numNodes);

	// Perform estimation.
	if(pWorkList->numNodes > 0)
	{
		// Estimate query radius for each node in the work list using Zhou's algorithm.
		KDSetTreePoint(*pKDData, d_points, m_numPoints, m_knnRefineIters, m_knnTargetCount);
		KernelKDComputeNodeQR(pWorkList->d_idxNode, pWorkList->d_nodeExtent, pWorkList->numNodes, 
			GetMaxQueryRadius(), d_nodeRadiusEstimate);
		KDUnsetTreePoint();
	}

	SAFE_DELETE(pWorkList);
	mncudaSafeCallNoSync(s_timer.Stop(true));
}

void KDTreePoint::ComputeQueryRadii(float4* d_queryPoints, uint numQueryPoints, float* d_outRadii)
{
	KDTreeData* pKDData = GetData();
	mncudaInitConstant(d_outRadii, numQueryPoints, GetMaxQueryRadius());

	KDSetTreePoint(*pKDData, d_points, m_numPoints, m_knnRefineIters, m_knnTargetCount);

	// First estimate a query radius based on node query radii.
	KernelKDEstimateRadii(d_queryPoints, numQueryPoints, d_nodeRadiusEstimate, pKDData->d_nodeExtent,
		GetMaxQueryRadius(), d_outRadii);
	
	// Now refine the estimates using histogram based iteration.
	KernelKDRefineRadiusZhou(d_queryPoints, numQueryPoints, d_outRadii);

	KDUnsetTreePoint();

	/*float res;
	mncudaReduce(res, d_outQREstimate, numPoints, MNCuda_ADD, 0.f);
	printf("QR estimate average: %.03f.\n", res / float(numPoints));*/
}