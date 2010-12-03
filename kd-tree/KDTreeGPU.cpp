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

#include "KDTreeGPU.h"
#include "KDTreeListener.h"
#include "../MNCudaPrimitives.h"
#include "../MNCudaUtil.h"
#include "../MNStatContainer.h"

//#define PRINT_INFO
//#define PRINT_KDTREE_STATS
//#define TEST_KDTREE
//#define KDTREE_MEASURE_TIME

#ifdef KDTREE_MEASURE_TIME
#define CREATE_TIMER(var, name) static StatTimer& var = StatTimer::Create("Timers", "kd-tree: " name, false)
#define START_TIMER(var) mncudaSafeCallNoSync(var.Start(true));
#define STOP_TIMER(var) mncudaSafeCallNoSync(var.Stop(true));
#else
#define CREATE_TIMER(var, name) ((void)0)
#define START_TIMER(var) ((void)0)
#define STOP_TIMER(var) ((void)0)
#endif //KDTREE_MEASURE_TIME

extern "C"
void KDInitializeKernels();
extern "C"
void KDSetParameters(uint smallNodeMax);
extern "C"
void KernelKDGetChunkCounts(uint* d_numElemsNode, uint numNodes, uint* d_outChunkCounts);
extern "C"
void KernelKDGenerateChunks(uint* d_numElemsNode, uint* d_idxFirstElemNode, uint numNodes, 
							uint* d_offsets, KDChunkList& lstChunks);
extern "C"
void KernelKDCountElemsPerChunk(const KDChunkList& lstChunks, uint* d_validFlags, uint* d_outCountPerChunk);
extern "C++"
template <uint numElementPoints>
void KernelKDGenChunkAABB(const KDNodeList& lstActive, KDChunkList& lstChunks);
extern "C"
void KernelKDEmptySpaceCutting(KDNodeList& lstActive, KDFinalNodeList& lstFinal, float emptySpaceRatio,
							   uint* d_ioFinalListIndex);
extern "C"
void KernelKDSplitLargeNodes(const KDNodeList& lstActive, KDNodeList& lstNext);
extern "C"
void KernelKDUpdateFinalListChildInfo(const KDNodeList& lstActive, KDFinalNodeList& lstFinal,
								      uint* d_finalListIndex);
extern "C++"
template <uint numElementPoints>
void KernelKDMarkLeftRightElements(const KDNodeList& lstActive, const KDChunkList& lstChunks, 
						           uint* d_valid);
extern "C"
void KernelKDMarkSmallNodes(const KDNodeList& lstNext, uint* d_finalListIndex, uint* d_isSmall, 
						    uint* d_smallRootParent);
extern "C"
void KernelKDMarkElemsByNodeSize(const KDChunkList& lstChunks, uint* d_numElemsNext, 
							     uint* d_outIsSmallElem, uint* d_outIsLargeElem);
extern "C"
void KernelKDMoveNodes(const KDNodeList& lstSource, KDNodeList& lstTarget, uint* d_move, uint* d_offsets,
					   bool bTargetIsSmall);
extern "C++"
template <uint numElementPoints>
void KernelKDCreateSplitCandidates(const KDNodeList& lstSmall, KDSplitList& lstSplit);
extern "C++"
template <uint numElementPoints>
void KernelKDInitSplitMasks(const KDNodeList& lstSmall, uint smallNodeMax, KDSplitList& lstSplit);
extern "C"
void KernelKDUpdateSmallRootParents(const KDFinalNodeList& lstNodes, uint* d_smallRootParents, uint numSmallNodes);
extern "C++"
template<uint numElementPoints>
void KernelKDFindBestSplits(const KDNodeList& lstActive, const KDSplitList& lstSplit, float maxQueryRadius,
						    uint* d_outBestSplit, float* d_outSplitCost);
extern "C"
void KernelKDSplitSmallNodes(const KDNodeList& lstActive, const KDSplitList& lstSplit, const KDNodeList& lstNext,
						     uint* d_inBestSplit, float* d_inSplitCost, uint* d_outIsSplit);
extern "C"
void KernelKDGenerateENAFromMasks(KDNodeList& lstActive, const KDNodeList& lstSmallRoots);
extern "C"
void KernelKDTraversalUpPath(const KDFinalNodeList& lstFinal, uint curLevel, uint* d_sizes);
extern "C"
void KernelKDTraversalDownPath(const KDFinalNodeList& lstFinal, uint curLevel, uint* d_sizes,
							   uint* d_addresses, KDTreeData& kdData);
extern "C++"
template <uint numElementPoints>
void KernelKDTestNodeList(const KDNodeList& lstNodes, uint* d_valid, float3 rootMin, float3 rootMax,
						  bool useTightBounds);
extern "C"
void KernelKDSetCustomBit(const KDTreeData& kdData, uint bitNo, uint* d_values);


KDTreeGPU::KDTreeGPU(size_t numInputElems, uint numElementPoints, float3 rootAABBMin, float3 rootAABBMax)
	: d_tempVal(1),
	  m_pCurrentChunkListSource(NULL),
	  m_numInputElements(numInputElems),
	  m_numElementPoints(numElementPoints),
	  m_rootAABBMin(rootAABBMin),
	  m_rootAABBMax(rootAABBMax)
{
	m_pListFinal = NULL;
	m_pListActive = NULL;
	m_pListSmall = NULL;
	m_pListNext = NULL;
	m_pChunkList = NULL;
	m_pSplitList = NULL;
	m_pKDData = NULL;

	d_smallRootParents = NULL;

	KDInitializeKernels();

	m_fEmptySpaceRatio = 0.25f;
	m_nSmallNodeMax = 64;
	m_fMaxQueryRadius = 1.f;
}

KDTreeGPU::~KDTreeGPU(void)
{
	Destroy();
}

void KDTreeGPU::PreBuild()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	m_pListFinal = new KDFinalNodeList();
	m_pListFinal->Initialize(2*m_numInputElements, 16*m_numInputElements);
	m_pListActive = new KDNodeList();
	m_pListActive->Initialize(m_numInputElements, 4*m_numInputElements, m_numElementPoints);
	m_pListSmall = new KDNodeList();
	m_pListSmall->Initialize(m_numInputElements, 4*m_numInputElements, m_numElementPoints);
	m_pListNext = new KDNodeList();
	m_pListNext->Initialize(m_numInputElements, 4*m_numInputElements, m_numElementPoints);
	m_pChunkList = new KDChunkList();
	m_pChunkList->Initialize(m_numInputElements);

	// Do not initialize here since size is unknown.
	m_pSplitList = NULL;
	m_pKDData = NULL;

	// Initialize small root parent vector. We need at most x entries, where x is the
	// maximum number of nodes in the small (root) list.
	mncudaSafeCallNoSync(pool.Request((void**)&d_smallRootParents, m_numInputElements*sizeof(uint)));

	KDSetParameters(m_nSmallNodeMax);
}

bool KDTreeGPU::BuildTree()
{
	CREATE_TIMER(s_timer, "kd-tree: total build time");
	START_TIMER(s_timer);

	PreBuild();

	// Reset lists.
	m_pListFinal->Clear();
	m_pListActive->Clear();
	m_pListSmall->Clear();
	m_pListNext->Clear();
	m_pChunkList->Clear();

	// Create and add root node.
	AddRootNode(m_pListActive);

	// Large node stage.
	LargeNodeStage();

	// Test the small node list here.
	TestNodeList(m_pListSmall, "Small node list (final roots)", false, true, true);

	// Small node stage.
	SmallNodeStage();

	// Generates final node list m_pKDData.
	PreorderTraversal();

	PostBuild();

	STOP_TIMER(s_timer);

	return true;
}

/*virtual*/ void KDTreeGPU::PostBuild()
{
	if(!d_smallRootParents)
		return;

	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	mncudaSafeCallNoSync(pool.Release(d_smallRootParents));
	d_smallRootParents = NULL;

	m_pListFinal->Free();
	SAFE_DELETE(m_pListFinal);
	m_pListActive->Free();
	SAFE_DELETE(m_pListActive);
	m_pListSmall->Free();
	SAFE_DELETE(m_pListSmall);
	m_pListNext->Free();
	SAFE_DELETE(m_pListNext);
	m_pChunkList->Free();
	SAFE_DELETE(m_pChunkList);

	if(m_pSplitList)
		m_pSplitList->Free();
	SAFE_DELETE(m_pSplitList);
}

void KDTreeGPU::Destroy()
{
	// Ensure stuff is destroyed.
	PostBuild();

	if(m_pKDData)
		m_pKDData->Free();
	SAFE_DELETE(m_pKDData);
}

void KDTreeGPU::LargeNodeStage()
{
	CREATE_TIMER(s_timer, "kd-tree: large node stage");
	START_TIMER(s_timer);

	// Iterate until the active list is empty, that is until there are
	// no more large nodes to work on.
	while(!m_pListActive->IsEmpty())
	{
		// Append the active list to the final node list. This is done even if the child information
		// aren't available, yet. It is fixed later.
		m_pListFinal->AppendList(m_pListActive, false, true);

		// Keeps track of where the current active list's nodes are in the final node list.
		MNCudaMemory<uint> d_finalListIndex(m_pListActive->numNodes);
		mncudaInitIdentity(d_finalListIndex, m_pListActive->numNodes);
		mncudaConstantOp<MNCuda_ADD, uint>(d_finalListIndex, m_pListActive->numNodes, 
			m_pListFinal->numNodes - m_pListActive->numNodes);

		// Clear the next list which stores the nodes for the next step.
		m_pListNext->Clear();

		// Process the active nodes. This generated both small nodes and new
		// next nodes.
		ProcessLargeNodes(d_finalListIndex);

		// Swap active and next list for next pass.
		KDNodeList* pTemp = m_pListActive;
		m_pListActive = m_pListNext;
		m_pListNext = pTemp;

#ifdef PRINT_INFO
		//printf("KD - Final nodes: %d.\n", m_pListFinal->numNodes);
#endif // PRINT_INFO
	}

	STOP_TIMER(s_timer);
}

void KDTreeGPU::SmallNodeStage()
{
	CREATE_TIMER(s_timer, "kd-tree: small node stage");
	START_TIMER(s_timer);

	// Check average small node size.
	/*uint* triCounts = new uint[m_pListSmall->numNodes];
	mncudaSafeCallNoSync(cudaMemcpy(triCounts, m_pListSmall->d_numElems, m_pListSmall->numNodes*sizeof(uint), 
		cudaMemcpyDeviceToHost));
	uint totalTris = 0;
	for(uint i=0; i<m_pListSmall->numNodes; i++)
		totalTris += triCounts[i];
	printf("Small root average tri count: %.2f.\n", (float)totalTris/(float)m_pListSmall->numNodes);
	SAFE_DELETE_ARRAY(triCounts);*/

	// Preprocess...
	PreProcessSmallNodes();

	m_pListActive->Clear();
	m_pListActive->AppendList(m_pListSmall, true);
	while(!m_pListActive->IsEmpty())
	{
		// NOTE: The paper tells to append the active list to the final node list here.
		//		 I do not follow since this way the changes (children, ...) won't
		//		 get into the final node list. Instead I moved this to the end of the
		//		 small processing stage.

		// Clear the next list which stores the nodes for the next step.
		m_pListNext->Clear();
		ProcessSmallNodes();

		// No swapping required here. This is done in ProcessSmallNodes to avoid
		// using temporary memory.

		// Test next node list (allready swapped).
		TestNodeList(m_pListActive, "Next list (small node stage)", false, true, false);
	}

	STOP_TIMER(s_timer);
}

bool KDTreeGPU::TestNodeAABB(float4 aabbMinI, float4 aabbMaxI, float4 aabbMinT, float4 aabbMaxT,
							   bool bHasInheritedBounds, bool bHasTightBounds)
{
	float3 rootMin = m_rootAABBMin;
	float3 rootMax = m_rootAABBMax;

	bool result = true;

	// Allow slight errors due to floating point arithmetic.
	float relErrorAllowed = 1e-6f;

	if(bHasInheritedBounds)
	{
		// Check if inherited bounds lie within root bounds.
		result &= fabsf(fmaxf(0.f, rootMin.x - aabbMinI.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, rootMin.y - aabbMinI.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, rootMin.z - aabbMinI.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
		result &= fabsf(fmaxf(0.f, aabbMaxI.x - rootMax.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, aabbMaxI.y - rootMax.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, aabbMaxI.z - rootMax.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
	}

	if(bHasTightBounds)
	{
		// Check if tight bounds lie within root bounds.
		result &= fabsf(fmaxf(0.f, rootMin.x - aabbMinT.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, rootMin.y - aabbMinT.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, rootMin.z - aabbMinT.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
		result &= fabsf(fmaxf(0.f, aabbMaxT.x - rootMax.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, aabbMaxT.y - rootMax.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, aabbMaxT.z - rootMax.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
	}

	if(bHasTightBounds && bHasInheritedBounds)
	{
		// Check if tight bounds lie within inherited bounds.
		result &= fabsf(fmaxf(0.f, aabbMinI.x - aabbMinT.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, aabbMinI.y - aabbMinT.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, aabbMinI.z - aabbMinT.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
		result &= fabsf(fmaxf(0.f, aabbMaxT.x - aabbMaxI.x)) < relErrorAllowed * (rootMax.x - rootMin.x);
		result &= fabsf(fmaxf(0.f, aabbMaxT.y - aabbMaxI.y)) < relErrorAllowed * (rootMax.y - rootMin.y);
		result &= fabsf(fmaxf(0.f, aabbMaxT.z - aabbMaxI.z)) < relErrorAllowed * (rootMax.z - rootMin.z);
	}

	return result;
}

void KDTreeGPU::TestNodeList(KDNodeList* pList, char* strTest,
							   bool bHasInheritedBounds, bool bHasTightBounds, bool bCheckElements)
{
#ifdef TEST_KDTREE
	if(pList->IsEmpty())
		return;

	printf("KD - Testing node list: %s.\n", strTest);

	// Check whether the node bounds lie within the scene bounds.
	float4* aabbInheritMin = new float4[pList->numNodes];
	float4* aabbInheritMax = new float4[pList->numNodes];
	float4* aabbTightMin = new float4[pList->numNodes];
	float4* aabbTightMax = new float4[pList->numNodes];
	mncudaSafeCallNoSync(cudaMemcpy(aabbInheritMin, pList->d_aabbMinInherit, pList->numNodes*sizeof(float4), 
		cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(aabbInheritMax, pList->d_aabbMaxInherit, pList->numNodes*sizeof(float4), 
		cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(aabbTightMin, pList->d_aabbMinTight, pList->numNodes*sizeof(float4), 
		cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(aabbTightMax, pList->d_aabbMaxTight, pList->numNodes*sizeof(float4), 
		cudaMemcpyDeviceToHost));

	bool ok = true;
	for(uint i=0; i<pList->numNodes; i++)
	{
		if(!TestNodeAABB(aabbInheritMin[i], aabbInheritMax[i], aabbTightMin[i], aabbTightMax[i], 
			bHasInheritedBounds, bHasTightBounds))
		{
			printf("Scene min: %.3f, %.3f, %.3f\n", m_rootAABBMin.x, m_rootAABBMin.y, m_rootAABBMin.z);
			printf("Scene max: %.3f, %.3f, %.3f\n", m_rootAABBMax.x, m_rootAABBMax.y, m_rootAABBMax.z);
			if(bHasInheritedBounds)
			{
				printf("Inherit min: %.3f, %.3f, %.3f\n", aabbInheritMin[i].x, aabbInheritMin[i].y, aabbInheritMin[i].z);
				printf("Inherit max: %.3f, %.3f, %.3f\n", aabbInheritMax[i].x, aabbInheritMax[i].y, aabbInheritMax[i].z);
			}
			if(bHasTightBounds)
			{
				printf("Tight min: %.3f, %.3f, %.3f\n", aabbTightMin[i].x, aabbTightMin[i].y, aabbTightMin[i].z);
				printf("Tight max: %.3f, %.3f, %.3f\n", aabbTightMax[i].x, aabbTightMax[i].y, aabbTightMax[i].z);
			}
			ok = false;
			break;
		}
	}

	delete [] aabbInheritMin;
	delete [] aabbInheritMax;
	delete [] aabbTightMin;
	delete [] aabbTightMax;

	if(!ok)
		MNFatal("%s check: illegal node bounds.\n", strTest);
	
	// Check if all elements lie within their node bounds.
	// WARNING: This won't work for small nodes as we do not perform split clipping for them.
	if(bCheckElements)
	{
		if(pList->numElementPoints == 1)
			KernelKDTestNodeList<1>(*pList, d_tempVal, m_rootAABBMin, m_rootAABBMax, bHasTightBounds);
		else
			KernelKDTestNodeList<2>(*pList, d_tempVal, m_rootAABBMin, m_rootAABBMax, bHasTightBounds);
		uint isValid;
		mncudaSafeCallNoSync(cudaMemcpy(&isValid, d_tempVal, sizeof(uint), cudaMemcpyDeviceToHost));
		if(!isValid)
			MNFatal("%s check: illegal elements detected.\n", strTest);
	}
#endif
}

void KDTreeGPU::ProcessLargeNodes(uint* d_finalListIndexActive)
{
	MNAssert(!m_pListActive->IsEmpty());
	
	// Group elements into chunks.
	CreateChunkList(m_pListActive);

	// Compute per node bounding boxes.
	ComputePerNodeAABBs();

	// Now tight node bounds are available.
	/*if(m_pListFinal->numNodes == 0) // Then m_pListActive contains the root node only.
	{
		float4 aabbMin, aabbMax;
		mncudaSafeCallNoSync(cudaMemcpy(&aabbMin, m_pListActive->d_aabbMinTight, sizeof(float4), cudaMemcpyDeviceToHost));
		mncudaSafeCallNoSync(cudaMemcpy(&aabbMax, m_pListActive->d_aabbMaxTight, sizeof(float4), cudaMemcpyDeviceToHost));
		printf("KDROOT - x: %.3f - %.3f.\n", aabbMin.x, aabbMax.x);
		printf("KDROOT - y: %.3f - %.3f.\n", aabbMin.y, aabbMax.y);
		printf("KDROOT - z: %.3f - %.3f.\n", aabbMin.z, aabbMax.z);
	}*/

	// Test the node list here. This is the first point where we have tight AABBs.
	TestNodeList(m_pListActive, "Active list (large node stage)", true, true, true);

	// Split large nodes.
	SplitLargeNodes(d_finalListIndexActive);

	// Sort and clip elements to child nodes.
	SortAndClipToChildNodes();

	// Now we have unclipped element bounds, so perform split clipping. Per default, this
	// does nothing. Clipping has to be realized in subclasses.
	CREATE_TIMER(s_timer, "kd-tree: split clipping");
	START_TIMER(s_timer);
	PerformSplitClipping(m_pListActive, m_pListNext);
	STOP_TIMER(s_timer);

	// Test before updating lists.
	TestNodeList(m_pListNext, "Next list (children before update)", true, false, true);

	// Now update lists for next run.
	UpdateSmallList(d_finalListIndexActive);
}

void KDTreeGPU::CreateChunkList(KDNodeList* pList)
{
	if(m_pCurrentChunkListSource == pList)
		return;

	MNCudaMemory<uint> d_counts(pList->numNodes);
	MNCudaMemory<uint> d_offsets(pList->numNodes);

	// Clear old list first.
	m_pChunkList->Clear();

	// Check if the chunk list is large enough. For now we do NO resizing since we allocated a
	// big enough chunk list (probably too big in most cases).
	uint maxChunks = pList->nextFreePos / KD_CHUNKSIZE + pList->numNodes;
	MNAssert(maxChunks <= m_pChunkList->maxChunks);
	if(maxChunks > m_pChunkList->maxChunks)
		MNFatal("Chunk list too small (max: %d; need: %d).\n", m_pChunkList->maxChunks, maxChunks);

	// Get the count of chunks for each node. Store them in d_counts.
	KernelKDGetChunkCounts(pList->d_numElems, pList->numNodes, d_counts);

	// Scan the counts to d_offsets. Use exclusive scan cause then we have
	// the start index for the i-th node in the i-th element of d_offsets.
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	cp.Scan(d_counts, pList->numNodes, false, d_offsets);

	// Generate chunk list.
	KernelKDGenerateChunks(pList->d_numElems, pList->d_idxFirstElem, pList->numNodes, 
		d_offsets, *m_pChunkList);

	// Set number of chunks.
	mncudaReduce(m_pChunkList->numChunks, (uint*)d_counts, pList->numNodes, MNCuda_ADD, (uint)0);

#ifdef PRINT_INFO
	//printf("Chunk list created. Num chunks: %d.\n", m_pChunkList->numChunks);
#endif
}

void KDTreeGPU::ComputePerNodeAABBs()
{
	CREATE_TIMER(s_timer, "kd-tree: node AABB computation");
	START_TIMER(s_timer);

	// First compute the bounding boxes of all chunks in parallel.
	if(m_pListActive->numElementPoints == 1)
		KernelKDGenChunkAABB<1>(*m_pListActive, *m_pChunkList);
	else
		KernelKDGenChunkAABB<2>(*m_pListActive, *m_pChunkList);

	/*float4 aabbMin;
	float4 aabbMax;
	mncudaSafeCallNoSync(cudaMemcpy(&aabbMin, m_pChunkList->d_aabbMin, sizeof(float4), cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(&aabbMax, m_pChunkList->d_aabbMax, sizeof(float4), cudaMemcpyDeviceToHost));
	printf("Chunk min: %.3f, %.3f, %.3f\n", aabbMin.x, aabbMin.y, aabbMin.z);
	printf("Chunk max: %.3f, %.3f, %.3f\n", aabbMax.x, aabbMax.y, aabbMax.z);*/

	// Now compute the tight bounding boxes of all nodes in parallel using
	// segmented reduction.
	mncudaSegmentedReduce(m_pChunkList->d_aabbMin, m_pChunkList->d_idxNode, 
		m_pChunkList->numChunks, MNCuda_MIN, make_float4(MN_INFINITY), 
		m_pListActive->d_aabbMinTight, m_pListActive->numNodes);
	mncudaSegmentedReduce(m_pChunkList->d_aabbMax, m_pChunkList->d_idxNode, 
		m_pChunkList->numChunks, MNCuda_MAX, make_float4(-MN_INFINITY), 
		m_pListActive->d_aabbMaxTight, m_pListActive->numNodes);

	STOP_TIMER(s_timer);
}

void KDTreeGPU::SplitLargeNodes(uint* d_finalListIndexActive)
{	
	MNAssert(m_pListFinal->numNodes >= m_pListActive->numNodes);
	CREATE_TIMER(s_timer, "kd-tree: large node splitting");
	START_TIMER(s_timer);

	// Cut of empty space. This updates the final list to include the cut-off empty space nodes
	// as well as the updated nodes, that are in the active list, too. It keeps track where the
	// active list nodes reside in the final list by updating the parent index array appropriately.
	KernelKDEmptySpaceCutting(*m_pListActive, *m_pListFinal, m_fEmptySpaceRatio, d_finalListIndexActive);


	// Now we can perform real spatial median splitting to create exactly two child nodes for
	// each active list node (into the next list).

	// Check if there is enough space in the next list.
	if(m_pListNext->maxNodes < 2*m_pListActive->numNodes)
		m_pListNext->ResizeNodeData(2*m_pListActive->numNodes);

	// Perform splitting. Also update the final node child relationship.
	KernelKDSplitLargeNodes(*m_pListActive, *m_pListNext);

	// Set new number of nodes.
	m_pListNext->numNodes = 2*m_pListActive->numNodes;

	STOP_TIMER(s_timer);
}

void KDTreeGPU::SortAndClipToChildNodes()
{
	MNAssert(m_pChunkList->numChunks > 0);

	CREATE_TIMER(s_timer, "kd-tree: large sort and clip");
	START_TIMER(s_timer);

	uint nextFreeL;
	MNCudaMemory<uint> d_countsUnaligned(2*m_pListActive->numNodes);
	MNCudaMemory<uint> d_chunkCounts(2*MNCUDA_ALIGN(m_pChunkList->numChunks));
	// Size: 2*m_pListActive->nextFreePos, first half for left marks, second for right marks.
	MNCudaMemory<uint> d_elemMarks(2*m_pListActive->nextFreePos);

	// Ensure the next's ENA is large enough.
	if(m_pListNext->maxElems < 2*m_pListActive->nextFreePos)
		m_pListNext->ResizeElementData(2*m_pListActive->nextFreePos);

	// We virtually duplicate the TNA of the active list and write it virtually twice into the
	// temporary TNA of the next list which we do not store explicitly.

	// Zero the marks. This is required since not all marks represent valid elements.
	mncudaSafeCallNoSync(cudaMemset(d_elemMarks, 0, 2*m_pListActive->nextFreePos*sizeof(uint)));

	// Mark the valid elements in the virtual TNA. This is required since not all elements are
	// both in left and right child. The marked flags are of the same size as the virtual TNA and
	// hold 1 for valid tris, else 0.
	if(m_pListActive->numElementPoints == 1)
		KernelKDMarkLeftRightElements<1>(*m_pListActive, *m_pChunkList, d_elemMarks);
	else
		KernelKDMarkLeftRightElements<2>(*m_pListActive, *m_pChunkList, d_elemMarks);

	// Determine per chunk element count for nodes using per block reduction.
	// ... left nodes
	KernelKDCountElemsPerChunk(*m_pChunkList, d_elemMarks, d_chunkCounts);
	// ... right nodes
	KernelKDCountElemsPerChunk(*m_pChunkList, d_elemMarks + m_pListActive->nextFreePos, 
		d_chunkCounts + MNCUDA_ALIGN(m_pChunkList->numChunks));

	// Perform segmented reduction on per chunk results to get per child nodes results. The owner
	// list is the chunk's idxNode list.
	// ... left nodes
	mncudaSafeCallNoSync(mncudaSegmentedReduce((uint*)d_chunkCounts, m_pChunkList->d_idxNode, m_pChunkList->numChunks,
		MNCuda_ADD, (uint)0, (uint*)d_countsUnaligned, m_pListActive->numNodes));
	// ... right nodes
	mncudaSafeCallNoSync(mncudaSegmentedReduce((uint*)d_chunkCounts + MNCUDA_ALIGN(m_pChunkList->numChunks), 
		m_pChunkList->d_idxNode, m_pChunkList->numChunks,
		MNCuda_ADD, (uint)0, (uint*)d_countsUnaligned + m_pListActive->numNodes, m_pListActive->numNodes));


	// Compact both parts together using two segments.
	nextFreeL = CompactElementData(m_pListNext, 0, 0, m_pListActive, 0, 2*m_pListActive->numNodes,
					d_elemMarks, d_countsUnaligned, 2);

#ifdef PRINT_INFO
	// Get element counts since they're not stored explicitly.
	uint numElemsParent, numElemsChildren;
	mncudaReduce(numElemsParent, m_pListActive->d_numElems, m_pListActive->numNodes, MNCuda_ADD, (uint)0);
	mncudaReduce(numElemsChildren, m_pListNext->d_numElems, m_pListNext->numNodes, MNCuda_ADD, (uint)0);

	uint numMarked;
	mncudaReduce(numMarked, (uint*)d_elemMarks, 2*m_pListActive->nextFreePos, MNCuda_ADD, (uint)0);

	printf("KD - Splitting: %d.\n", numElemsParent);
	printf("KD - Children elements: %d (marks: %d).\n", numElemsChildren, numMarked);

	// Test if there is a node without elements.
	uint minElemCount;
	mncudaReduce(minElemCount, m_pListNext->d_numElems, m_pListNext->numNodes, MNCuda_MIN, 0xffffffff);
	printf("KD - Min element count: %d.\n", minElemCount);
	printf("KD - Avg. element count: %.2f.\n", float(numElemsChildren) / m_pListNext->numNodes);
	/*if(minElemCount == 0)
		MNFatal("Illegal split in large node stage detected.");*/
#endif

	// Update element total.
	m_pListNext->nextFreePos = nextFreeL; // nextFreeR is aligned!

	STOP_TIMER(s_timer);
}

uint KDTreeGPU::CompactElementData(KDNodeList* pListDest, uint destOffset, uint nodeOffset,
									 KDNodeList* pListSrc, uint srcOffset, uint numSourceNodes,
									 uint* d_validMarks, uint* d_countsUnaligned, uint numSegments/* = 1*/)
{
	MNAssert(numSegments > 0);

	CREATE_TIMER(s_timer, "kd-tree: compact element data");
	START_TIMER(s_timer);

	MNCudaMemory<uint> d_offsetsUnaligned(numSourceNodes);
	MNCudaMemory<uint> d_countsAligned(numSourceNodes);
	MNCudaMemory<uint> d_offsetsAligned(numSourceNodes);

	// Get unaligned offsets.
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	cp.Scan(d_countsUnaligned, numSourceNodes, false, d_offsetsUnaligned);

	// Get aligned counts to temp buffer to avoid uncoalesced access (here and later).
	/*mncudaSafeCallNoSync(cudaMemcpy(d_countsAligned, d_countsUnaligned, 
		numSourceNodes*sizeof(uint), cudaMemcpyDeviceToDevice));*/
	mncudaAlignCounts(d_countsAligned, d_countsUnaligned, numSourceNodes);

	// Get aligned offsets.
	cp.Scan(d_countsAligned, numSourceNodes, false, d_offsetsAligned);

	// Now copy in resulting *unaligned* counts and aligned offsets.
	mncudaSafeCallNoSync(cudaMemcpy(pListDest->d_numElems+nodeOffset, d_countsUnaligned, 
		numSourceNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(pListDest->d_idxFirstElem+nodeOffset, d_offsetsAligned, 
		numSourceNodes*sizeof(uint), cudaMemcpyDeviceToDevice));

	// Offset d_idxFirstElem by destOffset.
	if(destOffset > 0)
		mncudaConstantOp<MNCuda_ADD, uint>(pListDest->d_idxFirstElem+nodeOffset, numSourceNodes, destOffset);

	// Get next free position by reduction. Using two device-to-host memcpys were slower than this!
	uint alignedSum, nextFreePos;
	mncudaReduce(alignedSum, (uint*)d_countsAligned, numSourceNodes, MNCuda_ADD, (uint)0);
	nextFreePos = alignedSum + destOffset; // Add destination offset.

	// Move elements only if there are any!
	if(alignedSum == 0)
		printf("kd - WARNING: Splitting nodes and zero elements on one side!\n");
	else
	{
		// Algorithm to compute addresses (where to put which compacted element).
		// N        -  N  -  N        -  (node starts)
		// 0  0  0  0 -1  0 -1  0  0  0  (inverse difference of counts at aligned offsets, offsetted by one node!)
		// 0  0  0  0 -1 -1 -2 -2 -2 -2  (inclusive scan to distribute values)
		// 0  1  2  3  3  4  4  5  6  7  (AddIdentity)
		// x  x  x  -  x  -  x  x  x  -
		// -> Addresses where to put which compact elements.
		MNCudaMemory<uint> d_bufTemp(alignedSum);
		MNCudaMemory<uint> d_addresses(alignedSum);
		MNCudaMemory<uint> d_srcAddr(numSegments*pListSrc->nextFreePos);
		
		// Nothing to compute if there is only one source node!
		if(numSourceNodes == 1)
			mncudaInitIdentity(d_addresses, alignedSum);
		else
		{
			// 1. Get inverse count differences (that is unaligned - aligned = -(aligned - unaligned)).
			MNCudaMemory<uint> d_invCountDiffs(numSourceNodes);
			mncudaSafeCallNoSync(cudaMemcpy(d_invCountDiffs, d_countsUnaligned, 
					numSourceNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
			mncudaArrayOp<MNCuda_SUB, uint>(d_invCountDiffs, d_countsAligned, numSourceNodes);

			// 2. Copy inverse count differences to node starts in d_bufTemp, but offset them by one node
			//    so that the first node gets zero as difference.
			mncudaSafeCallNoSync(cudaMemset(d_addresses, 0, alignedSum*sizeof(uint)));
			//    Only need to set numSourceNode-1 elements, starting with d_offsetsAligned + 1 as
			//    first address.
			mncudaSetAtAddress((uint*)d_addresses, d_offsetsAligned + 1, (uint*)d_invCountDiffs, numSourceNodes - 1);

			// 3. Scan the differences to distribute them to the other node elements.
			//    Do this inplace in d_addresses.
			cp.Scan(d_addresses, alignedSum, true, d_addresses);

			// 4. Add identity
			mncudaAddIdentity(d_addresses, alignedSum);
		}

		// To avoid multiple calls of compact we just compact an identity array once
		// to generate the source addresses. 
		for(uint seg=0; seg<numSegments; seg++)
			mncudaInitIdentity(d_srcAddr + seg*pListSrc->nextFreePos, pListSrc->nextFreePos);
		mncudaSafeCallNoSync(cudaMemset(d_bufTemp, 0, alignedSum*sizeof(uint)));
		cp.Compact(d_srcAddr, d_validMarks, numSegments*pListSrc->nextFreePos, 
			d_bufTemp, d_tempVal);

		// Ensure the destination list element data is large enough.
		if(pListDest->maxElems < nextFreePos)
			pListDest->ResizeElementData(nextFreePos);
		
		// Now we can generate the source addresses by setting the compacted data
		// at the positions defined by d_addresses:
		// N     - N   N     -  (node starts)
		// 0 1 2 3 3 4 5 6 7 8  (identity - seg-scan result)
		// A B C D E F G H 0 0  (compact)
		// A B C D D E F G H 0	(SetFromAddress(d_srcAddr, d_addresses, compact)
		//       -           -
		// NOTE: We assume here that the compacted array has at least as many elements
		//       as the address array. Therefore the 0-read is possible. It doesn't
		//       destroy the result because the 0 values don't matter.
		mncudaSetFromAddress((uint*)d_srcAddr, d_addresses, (uint*)d_bufTemp, alignedSum);

		mncudaSetFromAddress(pListDest->d_elemNodeAssoc+destOffset, d_srcAddr,
			pListSrc->d_elemNodeAssoc, alignedSum);
		mncudaSetFromAddress(pListDest->d_elemPoint1+destOffset, d_srcAddr,
			pListSrc->d_elemPoint1, alignedSum);
		if(pListDest->numElementPoints == 2)
			mncudaSetFromAddress(pListDest->d_elemPoint2+destOffset, d_srcAddr,
				pListSrc->d_elemPoint2, alignedSum);
	}

	STOP_TIMER(s_timer);

	return nextFreePos;
}

void KDTreeGPU::UpdateSmallList(uint* d_finalListIndexActive)
{
	CREATE_TIMER(s_timer, "kd-tree: update small list");
	START_TIMER(s_timer);

	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	MNCudaMemory<uint> d_countsUnaligned(m_pListNext->numNodes);
	MNCudaMemory<uint> d_nodeMarks(m_pListNext->numNodes);
	MNCudaMemory<uint> d_smallParents(m_pListNext->numNodes);
	MNCudaMemory<uint> d_nodeListOffsets(m_pListNext->numNodes);

	uint numSmall, numLarge, nextFreeSmall, nextFreeLarge;

	// Group next list elements into chunks.
	CreateChunkList(m_pListNext);

	// Mark small nodes. Result to d_nodeMarks. Small node parent array to d_smallParents.
	KernelKDMarkSmallNodes(*m_pListNext, d_finalListIndexActive, d_nodeMarks, d_smallParents);

	// Compact small root parents array to get d_smallRootParents for new small roots.
	cp.Compact(d_smallParents, d_nodeMarks, m_pListNext->numNodes, 
		d_smallRootParents+m_pListSmall->numNodes, d_tempVal);
	// Store the number of small nodes, but do not update the list's value for now.
	// This ensures we still have the old value.
	mncudaSafeCallNoSync(cudaMemcpy(&numSmall, d_tempVal, sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef PRINT_INFO
	printf("KD - New small nodes: %d.\n", numSmall);
#endif

	// Get element markers to d_elemMarks. Zero first to avoid marked empty space.
	MNCudaMemory<uint> d_elemMarks(2*m_pListNext->nextFreePos);
	mncudaSafeCallNoSync(cudaMemset(d_elemMarks, 0, 2*m_pListNext->nextFreePos*sizeof(uint)));
	uint* d_isSmallElem = d_elemMarks;
	uint* d_isLargeElem = d_elemMarks + m_pListNext->nextFreePos;
	KernelKDMarkElemsByNodeSize(*m_pChunkList, m_pListNext->d_numElems, d_isSmallElem, d_isLargeElem);

/*#ifdef PRINT_INFO
	uint numElemsSmall;
	mncudaReduce(numElemsSmall, d_isSmallElem, m_pListNext->nextFreePos, MNCuda_ADD, (uint)0);
	printf("KD - Small elem marks: %d.\n", numElemsSmall);
#endif*/

	if(numSmall == 0)
		nextFreeSmall = m_pListSmall->nextFreePos;
	else
	{
		// Compact element count array to get d_numElems for small list.
		cp.Compact(m_pListNext->d_numElems, d_nodeMarks, m_pListNext->numNodes, 
			d_countsUnaligned, d_tempVal);

		// Scan nodes marks to get node list offsets.
		cp.Scan(d_nodeMarks, m_pListNext->numNodes, false, d_nodeListOffsets);

		// Resize small list if required.
		if(m_pListSmall->numNodes + numSmall > m_pListSmall->maxNodes)
			m_pListSmall->ResizeNodeData(m_pListSmall->numNodes + numSmall);

		// Now remove small nodes and add them to the small list.
		KernelKDMoveNodes(*m_pListNext, *m_pListSmall, d_nodeMarks, d_nodeListOffsets, true);

		// Need to update left & right child pointers in current active list here. This is
		// neccessary since we remove the small nodes and the current pointers point to
		// an array enriched with those small nodes.
		// Scan isSmall array d_nodeMarks to get an array we can subtract from the left/right indices
		// to get the final positions of the large nodes. Example:
		//
		// 0 1 2 3 4 5 6  (d_childLeft)
		// 0 1 1 0 1 0 0  (d_nodeMarks)
		// 0 0 1 2 2 3 3  (Scan d_nodeMarks -> d_nodeListOffsets)
		// 0 1 1 1 2 2 3  (d_childLeft - d_nodeListOffsets)
		mncudaArrayOp<MNCuda_SUB, uint>(m_pListActive->d_childLeft, 
			d_nodeListOffsets, m_pListActive->numNodes);
		mncudaArrayOp<MNCuda_SUB, uint>(m_pListActive->d_childRight, 
			d_nodeListOffsets+m_pListActive->numNodes, m_pListActive->numNodes);

		// Update ENA of small list. This can be done by compacting the next list ENA
		// using the marks for demarcation.
		nextFreeSmall = CompactElementData(m_pListSmall, m_pListSmall->nextFreePos, m_pListSmall->numNodes,
							m_pListNext, 0, numSmall,
							d_isSmallElem, d_countsUnaligned);
#ifdef PRINT_INFO
		uint numSmallTris;
		mncudaReduce(numSmallTris, (uint*)d_countsUnaligned, numSmall, MNCuda_ADD, (uint)0);
		printf("Small tris: %d.\n", numSmallTris);
#endif
	}

	// Do the same with the remaining large nodes. But do not calculate the markers
	// as they can be computed by inversion.
	//  - Large node markers.
	mncudaInverseBinary(d_nodeMarks, m_pListNext->numNodes);

	// Now we have updated child and split information. We need to update the corresponding
	// final node list entries to reflect these changes.
	KernelKDUpdateFinalListChildInfo(*m_pListActive, *m_pListFinal, d_finalListIndexActive);

	// Compact next list. This is slightly difficult since we cannot do this inplace.
	// Instead we abuse the active list as temporary target. This is no problem since
	// this is the last stage of the large node processing. The active list is cleared
	// after this (under the name next list, as they are swapped).
	m_pListActive->Clear();


	// Compact element count array to get d_numElems for large list.
	cp.Compact(m_pListNext->d_numElems, d_nodeMarks, m_pListNext->numNodes, 
		d_countsUnaligned, d_tempVal);
	mncudaSafeCallNoSync(cudaMemcpy(&numLarge, d_tempVal, sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef PRINT_INFO
	printf("KD - Remaining large nodes: %d.\n", numLarge);
#endif

	if(numLarge == 0)
		nextFreeLarge = 0;
	else
	{
		// Scan nodes marks to get node list offsets.
		MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
		cp.Scan(d_nodeMarks, m_pListNext->numNodes, false, d_nodeListOffsets);

		// Resize active list if required.
		if(numLarge > m_pListActive->maxNodes)
			m_pListActive->ResizeNodeData(numLarge);

		// Now move large nodes.
		KernelKDMoveNodes(*m_pListNext, *m_pListActive, d_nodeMarks, d_nodeListOffsets, false);

		// Compact ENA of next list.
		nextFreeLarge = CompactElementData(m_pListActive, 0, 0,
							m_pListNext, 0, numLarge,
							d_isLargeElem, d_countsUnaligned);
#ifdef PRINT_INFO
		uint numLargeTris;
		mncudaReduce(numLargeTris, (uint*)d_countsUnaligned, numLarge, MNCuda_ADD, (uint)0);
		printf("Large tris: %d.\n", numLargeTris);
#endif
	}


	// Now the new next list is the active list as we used it to temporarily build the
	// next list to avoid overwriting. We make the active list real by swapping it with
	// the next list. This avoids copying stuff back.
	KDNodeList* pTemp = m_pListActive;
	m_pListActive = m_pListNext;
	m_pListNext = pTemp;
	m_pListActive->Clear();


	// Update counts.
	m_pListSmall->numNodes += numSmall;  // Note the "+"
	m_pListSmall->nextFreePos = nextFreeSmall; // Aligned. No need for +.
	m_pListNext->numNodes = numLarge;
	m_pListNext->nextFreePos = nextFreeLarge; // Aligned.

	TestNodeList(m_pListSmall, "Small node list (roots)", false, true, true);

	STOP_TIMER(s_timer);
}


// This generates the split candidate list.
void KDTreeGPU::PreProcessSmallNodes()
{
	MNCudaMemory<uint> d_alignedSplitCounts(m_pListSmall->numNodes);

	m_pSplitList = new KDSplitList();
	m_pSplitList->Initialize(m_pListSmall);

	/*uint smallRootElems;
	mncudaReduce(smallRootElems, m_pListSmall->d_numElems, m_pListSmall->numNodes, MNCuda_ADD, (uint)0);
	printf("SMALL ROOT ELEMS: %d.\n", smallRootElems);*/

	// Compute candidate number. This is exactly numElementPoints the number of
	// elements times three axes for each node.
	mncudaSafeCallNoSync(cudaMemcpy(m_pSplitList->d_numSplits, m_pListSmall->d_numElems, 
		m_pListSmall->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaConstantOp<MNCuda_MUL, uint>(m_pSplitList->d_numSplits, 
		m_pListSmall->numNodes, m_pListSmall->numElementPoints*3);

	// Align split counts before scanning to get aligned split offsets.
	mncudaAlignCounts(d_alignedSplitCounts, m_pSplitList->d_numSplits, m_pListSmall->numNodes);

	// Compute offsets from counts using scan.
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	cp.Scan(d_alignedSplitCounts, m_pListSmall->numNodes, false, m_pSplitList->d_idxFirstSplit);

	// Get number of entries required for split list.
	// NOTE: Reduction is currently required because alignment prevents from simply
	//       calculating the total number.
	uint numSplitTotal;
	mncudaReduce(numSplitTotal, (uint*)d_alignedSplitCounts, m_pListSmall->numNodes, MNCuda_ADD, (uint)0);

	// Allocate split memory. Use 128 byte alignment for 64 bit element masks.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Request((void**)&m_pSplitList->d_splitPos, numSplitTotal*sizeof(float), 
		"kd-tree misc"));
	mncudaSafeCallNoSync(pool.Request((void**)&m_pSplitList->d_splitInfo, numSplitTotal*sizeof(uint), 
		"kd-tree misc"));
	mncudaSafeCallNoSync(pool.Request((void**)&m_pSplitList->d_maskLeft, numSplitTotal*sizeof(ElementMask), 
		"kd-tree misc", 128));
	mncudaSafeCallNoSync(pool.Request((void**)&m_pSplitList->d_maskRight, numSplitTotal*sizeof(ElementMask), 
		"kd-tree misc", 128));

	// Create split candidates (inits split list). Parallelized over nodes.
	if(m_pListSmall->numElementPoints == 1)
		KernelKDCreateSplitCandidates<1>(*m_pListSmall, *m_pSplitList);
	else
		KernelKDCreateSplitCandidates<2>(*m_pListSmall, *m_pSplitList);

	// Now init split masks. Parallelized over split candidates.
	if(m_pListSmall->numElementPoints == 1)
		KernelKDInitSplitMasks<1>(*m_pListSmall, m_nSmallNodeMax, *m_pSplitList);
	else
		KernelKDInitSplitMasks<2>(*m_pListSmall, m_nSmallNodeMax, *m_pSplitList);

	// Initialize small list extra data (d_elemMask, d_idxSmallRoot). We just set the masks
	// completely (even if there are less elements in the node) to allow the use of
	// cudaMemset.
	mncudaInitIdentity(m_pListSmall->d_idxSmallRoot, m_pListSmall->numNodes);
	mncudaSafeCallNoSync(cudaMemset(m_pListSmall->d_elemMask, 0xFF, m_pListSmall->numNodes*sizeof(ElementMask)));

	// We now need to update small root parents in m_pListFinal. This is neccessary to
	// retain the connection between large tree and small root trees. It is assumed
	// here that the small root nodes are added right after the current m_pListFinal
	// last node.
	KernelKDUpdateSmallRootParents(*m_pListFinal, d_smallRootParents, m_pListSmall->numNodes);
}

void KDTreeGPU::ProcessSmallNodes()
{
	MNAssert(m_pListActive->numNodes > 0);
	MNCudaMemory<uint> d_isSplit(m_pListActive->numNodes);
	MNCudaMemory<uint> d_bestSplits(m_pListActive->numNodes);
	MNCudaMemory<float> d_minSAHs(m_pListActive->numNodes);

	if(m_pListActive->numElementPoints == 1)
		KernelKDFindBestSplits<1>(*m_pListActive, *m_pSplitList, m_fMaxQueryRadius, d_bestSplits, d_minSAHs);
	else
		KernelKDFindBestSplits<2>(*m_pListActive, *m_pSplitList, m_fMaxQueryRadius, d_bestSplits, d_minSAHs);

	// Resize node data if required.
	if(m_pListNext->maxNodes < 2*m_pListActive->numNodes)
		m_pListNext->ResizeNodeData(2*m_pListActive->numNodes);

	// Generate children into next list. The result contains the left result in the first
	// m_pListActive->numNodes elements and the right results in the following elements.
	// It contains holes where no children were generated. Therefore we need to compact
	// the result and require a isSplit array in d_isSplit.
	KernelKDSplitSmallNodes(*m_pListActive, *m_pSplitList, *m_pListNext, 
		d_bestSplits, d_minSAHs, d_isSplit);

	// Just reuse buffer for child offsets.
	uint* d_childOffsets = d_bestSplits;

	// The child data is invalid cause we compact the next list later. Therefore both left
	// and right child indices have to be updated. This can be done by scanning the inverse
	// of the isSplit array. Example:
	//
	// 0 1 2 3 4 5 (Identity)
	// 0 0 1 0 0 1 (isSplit)
	// 0 0 2 0 0 5 (isSplit * Identity)
	// 1 1 0 1 1 0 (not isSplit)
	// 0 1 2 2 3 4 (scan not isSplit)
	// 0 0 2 0 0 4 (isSplit * (scan not isSplit))
	// 0 0 0 0 0 1 (Left := Identity - scan not isSplit)
	mncudaInverseBinary(d_isSplit, m_pListActive->numNodes);

	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	cp.Scan(d_isSplit, m_pListActive->numNodes, false, d_childOffsets);

	mncudaInverseBinary(d_isSplit, m_pListActive->numNodes);
	mncudaArrayOp<MNCuda_MUL, uint>(d_childOffsets, d_isSplit, m_pListActive->numNodes);

	mncudaInitIdentity(m_pListActive->d_childLeft, m_pListActive->numNodes);
	mncudaArrayOp<MNCuda_MUL, uint>(m_pListActive->d_childLeft, d_isSplit, m_pListActive->numNodes);
	mncudaArrayOp<MNCuda_SUB>(m_pListActive->d_childLeft, d_childOffsets, m_pListActive->numNodes);

	// Get left (and right) child count.
	uint numSplits;
	mncudaReduce(numSplits, (uint*)d_isSplit, m_pListActive->numNodes, MNCuda_ADD, (uint)0);

#ifdef PRINT_INFO
	printf("KD - Small nodes splitted into %d nodes.\n", 2*numSplits);
#endif

	// Left child indices are OK now. Right indices we get the following way:
	//
	// 0 0 0 0 0 1 (Left)
	// 0 0 1 0 0 1 (isSplit)
	// 2 2 2 2 2 2 (numSplits)
	// 0 0 2 0 0 2 (isSplit * numSplits)
	// 0 0 2 0 0 3 (Right = Left + (isSplit * numSplits))
	mncudaSafeCallNoSync(cudaMemcpy(m_pListActive->d_childRight, m_pListActive->d_childLeft, 
		m_pListActive->numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaInitConstant(d_childOffsets, m_pListActive->numNodes, numSplits);
	if(numSplits > 0)
	{
		mncudaArrayOp<MNCuda_MUL, uint>(d_childOffsets, d_isSplit, m_pListActive->numNodes);
		mncudaArrayOp<MNCuda_ADD>(m_pListActive->d_childRight, d_childOffsets, m_pListActive->numNodes);
	}

	// Child data is up to date. Append to final node list.
	TestNodeList(m_pListActive, "Active list (small node stage)", false, true, false);
	m_pListFinal->AppendList(m_pListActive, true, false);
	
	uint numNodesOld = m_pListActive->numNodes;
	m_pListActive->Clear();

	// Compact the result into the active list. This avoids using temporary buffers
	// and is possible since the active list is no longer needed. The following elements
	// have to be compacted:
	// d_aabbMinTight, d_aabbMaxTight, d_idxSmallRoot, d_numElems, d_elemMask
	if(numSplits > 0)
	{
		// Resize node data if required.
		if(m_pListActive->maxNodes < 2*numSplits)
			m_pListActive->ResizeNodeData(2*numSplits);

		MNCudaMemory<uint> d_srcAddr(numNodesOld);
		mncudaGenCompactAddresses(d_isSplit, numNodesOld, d_srcAddr);

		for(uint j=0; j<2; j++)
		{
			uint offset = j * numNodesOld;

			mncudaSetFromAddress(m_pListActive->d_aabbMinTight+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_aabbMinTight+offset, numSplits);
			mncudaSetFromAddress(m_pListActive->d_aabbMaxTight+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_aabbMaxTight+offset, numSplits);

			mncudaSetFromAddress(m_pListActive->d_idxSmallRoot+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_idxSmallRoot+offset, numSplits);
			mncudaSetFromAddress(m_pListActive->d_numElems+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_numElems+offset, numSplits);
			mncudaSetFromAddress(m_pListActive->d_nodeLevel+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_nodeLevel+offset, numSplits);

			mncudaSetFromAddress(m_pListActive->d_elemMask+m_pListActive->numNodes, d_srcAddr, 
				m_pListNext->d_elemMask+offset, numSplits);

			// Do not move this out of the loop as it is used as offset.
			m_pListActive->numNodes += numSplits;
		}
	}

	if(m_pListActive->numNodes > 0)
	{
		// Generate TNA for the new nodes. Currently we only have the masks.
		// At first compute the offsets (idxFirstTri) into the TNA by scanning the
		// element numbers.
		cp.Scan(m_pListActive->d_numElems, m_pListActive->numNodes, false, m_pListActive->d_idxFirstElem);

		// Get element total.
		uint numElems;
		mncudaReduce(numElems, (uint*)m_pListActive->d_numElems, 
			m_pListActive->numNodes, MNCuda_ADD, (uint)0);

		// Ensure the active's ENA is large enough.
		if(m_pListActive->maxElems < 2*MNCUDA_ALIGN(numElems))
			m_pListActive->ResizeElementData(2*MNCUDA_ALIGN(numElems));

		m_pListActive->nextFreePos = MNCUDA_ALIGN(numElems);

#ifdef PRINT_INFO
		printf("KD - Small nodes splitted into %d elements.\n", numElems);
#endif

		// Generate ENA from masks.
		KernelKDGenerateENAFromMasks(*m_pListActive, *m_pListSmall);
	}
}

void KDTreeGPU::PreorderTraversal()
{
	CREATE_TIMER(s_timer, "kd-tree: preorder traversal");
	START_TIMER(s_timer);

	MNCudaMemory<uint> d_nodeSizes(m_pListFinal->numNodes);

	// Now we have the final tree node list. Notify listeners.
	for(uint i=0; i<m_vecListeners.size(); i++)
		m_vecListeners[i]->OnFinalNodeList(m_pListFinal);

	// Initialize final node list now.
	m_pKDData = new KDTreeData();
	m_pKDData->Initialize(m_pListFinal, m_rootAABBMin, m_rootAABBMax);

	// Compute maximim node level.
	uint maxLevel;
	mncudaReduce(maxLevel, m_pListFinal->d_nodeLevel, m_pListFinal->numNodes, MNCuda_MAX, (uint)0);

	// At first perform a bottom-top traversal to determine the size of the
	// preorder tree structure.
	for(int lvl=maxLevel; lvl>=0; lvl--)
	{
		// Write sizes into d_nodeSizes.
		KernelKDTraversalUpPath(*m_pListFinal, lvl, d_nodeSizes);
	}

	//mncudaSafeCallNoSync(mncudaPrintArray(m_pListFinal->d_numElems, m_pKDData->numNodes, false, "NODE TRI COUNTS:"));

	// Now we have the total tree size in root's size. 
	m_pKDData->sizeTree = d_nodeSizes.Read(0);

	// Allocate preorder tree. Use special request for alignment.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&m_pKDData->d_preorderTree, 
		m_pKDData->sizeTree*sizeof(uint), "kd-tree result"));

	// Initialize root nodes address.
	mncudaSafeCallNoSync(cudaMemset(m_pKDData->d_nodeAddresses, 0, sizeof(uint)));

	// Top-down traversal to generate tree from sizes.
	for(uint lvl=0; lvl<=maxLevel; lvl++)
	{
		// Generate preorder tree.
		KernelKDTraversalDownPath(*m_pListFinal, lvl, d_nodeSizes, m_pKDData->d_nodeAddresses, *m_pKDData);
	}

	STOP_TIMER(s_timer);

	// DEBUG
#ifdef PRINT_KDTREE_STATS
	PrintTree();
#endif // PRINT_KDTREE_STATS
}

void KDTreeGPU::PrintTree()
{
	uint* preorderTree = new uint[m_pKDData->sizeTree];

	// Read data into host memory.
	mncudaSafeCallNoSync(cudaMemcpy(preorderTree, m_pKDData->d_preorderTree, m_pKDData->sizeTree*sizeof(uint), 
		cudaMemcpyDeviceToHost));

	// Traversal stack.
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoLevel[KD_MAX_HEIGHT];
	uint todoPos = 0;
	int addrNode = 0;
	int curLevel = -1;
	int maxLevelTrav = -1;

	uint numLeafs = 0;
	uint numLeafElems = 0;

	while(addrNode != -1)
	{
		uint idxNode = preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;
		curLevel++;
		maxLevelTrav = std::max(maxLevelTrav, curLevel);

		if(!isLeaf && idxNode >= m_pKDData->numNodes)
			MNFatal("Illegal node index %d at address %d.\n", idxNode, addrNode);

		if(!isLeaf)
		{
			MNAssert(todoPos < KD_MAX_HEIGHT);

			// Internal node.
			uint left = addrNode + 1 + 2;
			uint right = preorderTree[addrNode+1] & 0x0FFFFFFF;
			todoAddr[todoPos] = right;
			todoLevel[todoPos] = curLevel;
			todoPos++;

			//printf("A(%4d, %4d): L %4d R %4d A %d P %.3f \n", addrNode, numTri, left, right, 
			//	splitAxis[idxNode], splitPos[idxNode]);

			addrNode = left;
		}
		else
		{
			uint num = preorderTree[addrNode+1];

			numLeafs++;
			numLeafElems += num;

			//printf("A(%4d, %4d): Child\n", addrNode, numTri);

			if(todoPos > 0)
			{
				todoPos--;
				addrNode = todoAddr[todoPos];
				curLevel = todoLevel[todoPos];
			}
			else
				break;
		}
	}

	uint maxLevel;
	mncudaReduce(maxLevel, m_pListFinal->d_nodeLevel, m_pListFinal->numNodes, MNCuda_MAX, (uint)0);

	printf("KD-Tree: nodes %d, size %d, height: %d\n", m_pKDData->numNodes, m_pKDData->sizeTree, maxLevel+1);
	printf("KD-Tree: leafs %d, avg.elems: %.2f, total elems: %d.\n", numLeafs, 
		(float)numLeafElems/(float)numLeafs, numLeafElems);

	printf("\n");

	// Test query...
	/*float p[3] = {-1.40349317f, 10.15778828f, -7.559999660f};
	todoPos = 0;
	addrNode = 0;

	while(addrNode != -1)
	{
		uint idxNode = preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		while(!isLeaf)
		{
			uint left = addrNode + 1 + 2;
			uint right = preorderTree[addrNode+1] & 0x0FFFFFFF;
			uint splitAxis = preorderTree[addrNode+1] >> 30;
			float splitPos = *(float*)&preorderTree[addrNode+2];

			// Compute squared distance on split axis from query point to splitting plane.
			float distSqr = (p[splitAxis] - splitPos) * (p[splitAxis] - splitPos);

			// Advance to next child node, possibly enqueue other child.
			addrNode = left;
			uint addrOther = right;
			if(p[splitAxis] > splitPos)
			{
				// Next: right node.
				addrNode = right;
				addrOther = left;
			}

			// Enqueue other if required.
			if(distSqr < 1.0f)
			{
				// Enqueue second child in todo list.
				todoAddr[todoPos++] = addrOther;
			}

			// Read node index + leaf info (MSB) for new node.
			idxNode = preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		addrNode = -1;
		if(todoPos > 0)
		{
			todoPos--;
			addrNode = todoAddr[todoPos];
		}
	}

	printf("TEST QUERY SUCCESSFUL.\n");*/

	// Test query end...

	SAFE_DELETE_ARRAY(preorderTree);
}

void KDTreeGPU::AddListener(KDTreeListener* pListener)
{
	MNAssert(pListener);

	// Check if listener allready contained.
	for(uint i=0; i<m_vecListeners.size(); i++)
	{
		if(pListener == m_vecListeners[i])
			return;
	}

	m_vecListeners.push_back(pListener);
}

void KDTreeGPU::RemoveListener(KDTreeListener* pListener)
{
	MNAssert(pListener);

	for(uint i=0; i<m_vecListeners.size(); i++)
	{
		if(pListener == m_vecListeners[i])
		{
			m_vecListeners.erase(m_vecListeners.begin() + i);
			break;
		}
	}
}

void KDTreeGPU::SetCustomBits(uint bitNo, uint* d_values)
{
	MNAssert(bitNo < 2 && d_values);
	if(!m_pKDData)
		MNFatal("Setting custom bits requires final kd-tree data.");

	KernelKDSetCustomBit(*m_pKDData, bitNo, d_values);
}

void KDTreeGPU::SetSmallNodeMax(uint maximum) 
{ 
	if(maximum > KD_SMALLNODEMAX)
		MNFatal("Illegal small node maximum chosen. Maximum is: %d.", KD_SMALLNODEMAX);
	m_nSmallNodeMax = maximum; 
}