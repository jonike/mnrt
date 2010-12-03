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

#include "PhotonMap.h"
#include "RayPool.h"
#include "kd-tree/KDTreePoint.h"
#include "MNCudaMT.h"
#include "MNStatContainer.h"
#include "MNRTSettings.h"

// photon_knn.cu
extern "C"
void PMSetPhotonMap(const KDTreeData& pmap, const PhotonData& photons, uint knnTargetCount);
extern "C"
void PMUnsetPhotonMap();
extern "C"
void ICutSetInterpolData(float4* d_nodeCenterNormal, float4* d_nodeIrrExact, uint numNodes);
extern "C"
void ICutUnsetInterpolData();
extern "C"
void KernelPMPhotonGather(const ShadingPoints& shadingPts, const RayChunk& srcRays,
						  float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint,
						  float4* d_ioRadiance, float* d_ioInvDistAccu, bool bVisualizePhotons);
extern "C"
void KernelICutGatherByInterpolation(const ShadingPoints& shadingPts, const RayChunk& srcRays,
								     float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint, 
								     float4* d_ioRadiance, float* d_ioInvDistAccu);
extern "C"
void KernelICutIrrAtNodeCenters(float4* d_workQueryPoint, float* d_queryRadius, uint* d_workNodeIdx,
							    uint numWorkNodes, float4* d_nodeCenterNormals, float4* d_ioIrrExact);

// photon_build.cu
extern "C"
void KernelPMScaleFlux(PhotonData& ioPhotons, float scale);

// raytracing.cu
extern "C"
void KernelRTGetDiffuseColors(int* d_triHitIndices, float2* d_baryHit, uint numPoints, float4* d_outClrDiffHit);
extern "C"
void KernelRTApproximateNormalAt(const KDFinalNodeList& lstFinal, float queryRadiusMax,
							     float4* d_outNormals);

// bsdf.cu
extern "C"
void KernelBSDFGetNormalsAtHit(uint count, int* d_triHitIndices, float2* d_hitBaryCoords,
							   float4* d_outNormalsG, float4* d_outNormalsS);

// illumcut.cu
extern "C"
void KernelICutWriteNodeSideLength(const KDFinalNodeList& lstFinal, float4* d_ioNodeNormals);
extern "C"
void KernelICutEstimateNodeIrradiance(const KDFinalNodeList& lstFinal, float4* d_nodeNormals,
							          const PhotonData& photons,
									  float4* d_outNodeIrrEstimate);
extern "C"
float3 KernelICutGetAverageIrradianceOnLevel(uint* d_nodeLevel, float4* d_nodeIrrEst, 
											 uint numNodes, uint level);
extern "C"
void KernelICutMarkCoarseCut(const KDFinalNodeList& lstFinal, float4* d_nodeIrrEst, float3 Emin,
						     uint* d_outMarks);
extern "C"
void KernelICutMarkLeafs(const KDFinalNodeList& lstFinal, uint* d_outMarks);
extern "C"
void KernelICutUpdateFinalCut(const KDTreeData& pmap, float4* d_nodeIrrEst, float4* d_nodeIrrExact,
							  uint* d_workNodeIdx, uint numWorkNodes, float reqAccuracy,
							  uint* d_ioWorkMarks, uint* d_ioFinalCutMarks);



////////////////////////////////////////////////////////////////////////////////////////////////////
// PhotonMap implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

PhotonMap::PhotonMap(uint maxPhotons, float3 sceneAABBMin, float3 sceneAABBMax, float globalQRadiusMax,
		MNRTSettings* pSettings)
{
	MNAssert(maxPhotons > 0 && pSettings);
	m_sceneAABBMin = sceneAABBMin;
	m_sceneAABBMax = sceneAABBMax;
	m_pSettings = pSettings;

	m_knnTargetCount = 100;
	m_fGlobalQRadiusMax = globalQRadiusMax;

	m_Photons.Initialize(maxPhotons);
	m_pPhotonMap = NULL;

	d_nodeIrrEstimate = NULL;
	d_nodeIrrExact = NULL;
	d_nodeCenterNormals = NULL;
	d_isCoarseTreeCutNode = NULL;
}

PhotonMap::~PhotonMap(void)
{
	Destroy();
}

void PhotonMap::Destroy()
{
	PMUnsetPhotonMap();

	m_Photons.Destroy();
	DestroyTree();
}

void PhotonMap::DestroyTree()
{
	if(!m_pPhotonMap)
		return;

	SAFE_DELETE(m_pPhotonMap);

	if(d_nodeCenterNormals)
	{
		MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
		mncudaSafeCallNoSync(pool.Release(d_nodeCenterNormals));
		d_nodeCenterNormals = NULL;
		mncudaSafeCallNoSync(pool.Release(d_nodeIrrEstimate));
		d_nodeIrrEstimate = NULL;
		mncudaSafeCallNoSync(pool.Release(d_nodeIrrExact));
		d_nodeIrrExact = NULL;
		mncudaSafeCallNoSync(pool.Release(d_isCoarseTreeCutNode));
		d_isCoarseTreeCutNode = NULL;
	}
}

bool PhotonMap::BuildMap()
{
	// Destroy previous photon map and radius estimate.
	DestroyTree();

	if(m_Photons.numPhotons == 0)
		return true; // Nothing to do.

	m_pPhotonMap = new KDTreePoint(m_Photons.d_positions, m_Photons.numPhotons,
		m_sceneAABBMin, m_sceneAABBMax, m_fGlobalQRadiusMax);
	m_pPhotonMap->AddListener(this);
	m_pPhotonMap->SetKNNRefineIters(m_pSettings->GetKNNRefineIters());
	m_pPhotonMap->SetKNNTargetCount(m_knnTargetCount);
	if(!m_pPhotonMap->BuildTree())
		return false;

	// Now we have built the tree, we can perform exact KNN search for each cut node.
	// Using that, generate the final illumination cut.
	if(m_pSettings->GetUseIllumCuts())
		GenerateFinalIllumCut();

	return true;
}

uint PhotonMap::Merge(const PhotonData& other, uint* d_isValid)
{
	return m_Photons.Merge(other, d_isValid);
}

void PhotonMap::ScaleFlux(float scale)
{
	if(m_Photons.numPhotons == 0)
		return; // Nothing to do.

	// Do not scale all components as w contains spherical polar angle!
	KernelPMScaleFlux(m_Photons, scale);
}

void PhotonMap::Gather(const ShadingPoints& sp, const RayChunk& rayChunk,
					   float4* d_clrDiffHit, 
					   float4* d_ioRadiance)
{
	if(m_Photons.numPhotons == 0)
		return; // Nothing to do.

	KDTreeData* pKDData = m_pPhotonMap->GetData();
	MNCudaMemory<float> d_queryRadius(sp.numPoints);

	// WARNING: Cannot use illumination cuts for direct gathering as current algorithm doesn't
	//          handle boundaries very well.

	/*if(m_pSettings->GetUseIllumCuts())
	{
		// Use fixed query radius.
		d_queryRadius.InitConstant(m_fGlobalQRadiusMax);

		// Use illumination cut to interpolate precomputed illumination.
		PMSetPhotonMap(*pKDData, m_Photons, m_pPhotonMap->GetKNNTargetCount());
		ICutSetInterpolData(d_nodeCenterNormals, d_nodeIrrExact, pKDData->numNodes);
		KernelICutGatherByInterpolation(sp, rayChunk, d_queryRadius, 
			d_clrDiffHit, NULL, d_ioRadiance, NULL);
		ICutUnsetInterpolData();
		PMUnsetPhotonMap();
	}
	else*/
	{
		// Get estimated query radius for each shading point.
		m_pPhotonMap->ComputeQueryRadii(sp.d_ptInter, sp.numPoints, d_queryRadius);

		// Just perform full final gathering.
		PMSetPhotonMap(*pKDData, m_Photons, m_pPhotonMap->GetKNNTargetCount());
		KernelPMPhotonGather(sp, rayChunk, d_queryRadius, 
			d_clrDiffHit, NULL, d_ioRadiance, NULL, false);
		PMUnsetPhotonMap();
	}
}

void PhotonMap::GatherFor(const ShadingPoints& sp, const RayChunk& rayChunk,
						  float4* d_clrDiffHit, 
						  uint* d_idxShadingPts,
						  float4* d_ioRadiance, float* d_ioReciDistAccu)
{
	if(m_Photons.numPhotons == 0)
		return; // Nothing to do.

	MNCudaMemory<float> d_queryRadius(sp.numPoints);
	KDTreeData* pKDData = m_pPhotonMap->GetData();

	if(m_pSettings->GetUseIllumCuts())
	{
		// Use fixed query radius.
		d_queryRadius.InitConstant(m_fGlobalQRadiusMax);

		// Use illumination cut to interpolate precomputed illumination.
		PMSetPhotonMap(*pKDData, m_Photons, m_pPhotonMap->GetKNNTargetCount());
		ICutSetInterpolData(d_nodeCenterNormals, d_nodeIrrExact, pKDData->numNodes);
		KernelICutGatherByInterpolation(sp, rayChunk, d_queryRadius, 
			d_clrDiffHit, d_idxShadingPts, d_ioRadiance, d_ioReciDistAccu);
		ICutUnsetInterpolData();
		PMUnsetPhotonMap();
	}
	else
	{
		// Get estimated query radius for each shading point.
		//d_queryRadius.InitConstant(m_fGlobalQRadiusMax);
		m_pPhotonMap->ComputeQueryRadii(sp.d_ptInter, sp.numPoints, d_queryRadius);

		// Just perform full final gathering.
		PMSetPhotonMap(*pKDData, m_Photons, m_pPhotonMap->GetKNNTargetCount());
		KernelPMPhotonGather(sp, rayChunk, d_queryRadius, 
			d_clrDiffHit, d_idxShadingPts, d_ioRadiance, d_ioReciDistAccu, false);
		PMUnsetPhotonMap();
	}
}

void PhotonMap::Visualize(const ShadingPoints& sp, const RayChunk& rayChunk,
						  float4* d_ioRadiance)
{
	if(m_Photons.numPhotons == 0)
		return; // Nothing to do.

	// Get query radius for each shading point.
	MNCudaMemory<float> d_queryRadius(sp.numPoints);
	d_queryRadius.InitConstant(m_fGlobalQRadiusMax*0.05f);

	PMSetPhotonMap(*m_pPhotonMap->GetData(), m_Photons, 1);
	KernelPMPhotonGather(sp, rayChunk, d_queryRadius, NULL, NULL, d_ioRadiance, NULL, true);
	PMUnsetPhotonMap();
}

float3 PhotonMap::GetEminForCoarseCut(KDFinalNodeList* pNodesFinal)
{
	return KernelICutGetAverageIrradianceOnLevel(pNodesFinal->d_nodeLevel, 
		d_nodeIrrEstimate, pNodesFinal->numNodes, m_pSettings->GetICutLevelEmin());
}

// Wang et al. tell someting about computing estimated radiance values during kd-tree construction
// and not separately. I opted to move this to the end of the construction, where the final node
// list is just available.
void PhotonMap::OnFinalNodeList(KDFinalNodeList* pNodesFinal)
{
	if(!m_pSettings->GetUseIllumCuts())
		return;
	MNAssert(pNodesFinal && pNodesFinal->numNodes > 0);

	//mncudaPrintArray<uint>((uint*)pNodesFinal->d_nodeLevel, pNodesFinal->numNodes, false, "");

	// Request memory now we not the node list size.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeCenterNormals, pNodesFinal->numNodes*sizeof(float4), 
		"Illumination cuts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeIrrEstimate, pNodesFinal->numNodes*sizeof(float4), 
		"Illumination cuts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_nodeIrrExact, pNodesFinal->numNodes*sizeof(float4), 
		"Illumination cuts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_isCoarseTreeCutNode, pNodesFinal->numNodes*sizeof(uint), 
		"Illumination cuts"));

	// Estimate normals and BSDF at node centers.
	//   This step is somewhat imprecisely described in Wang et al.'s paper. They just refer to
	//   the photon map node's center normal. However, a normal could only be obtained using
	//   geometry information. For lower level kd-tree nodes this might make sense since those
	//   nodes cover photons from one surface only in most cases.
	//   However for nodes closer to the root there is no real normal estimate. Just consider an
	//   empty room with uniform photon distribution. The root's center would have no normal
	//   as it lies directly in empty space. Same problems apply to BSDF information.
	//
	//   My current approach uses the geometry kd-tree for a range search to find the closest
	//   triangle to the given node center. This triangle's normal and BSDF information are used
	//   to estimate stuff. However, I only employ this technique for a given fixed search radius.
	//   In case there are no triangles within that search radius, I just use fixed normal
	//   and BSDF for the given node center. This would lead to a useless estimated radiance, however
	//   this should be the case anyways.
	MNCudaMemory<int> d_triHitIndices(pNodesFinal->numNodes);
	MNCudaMemory<float2> d_hitBaryCoords(pNodesFinal->numNodes, "Temporary", 128);

	// Approximate normals at node centers.
	// WARNING: This is critical for the performance illumination cuts. I still need to find
	//          a way to pick a good radius for each center, especially when the nodes are
	//          picked on smaller levels.
	float queryRadiusMax = m_fGlobalQRadiusMax;
	KernelRTApproximateNormalAt(*pNodesFinal, queryRadiusMax, d_nodeCenterNormals);
	//mncudaPrintArray<uint>((uint*)(int*)d_triHitIndices, pNodesFinal->numNodes, false, "");

	// Write node side length into w-component.
	KernelICutWriteNodeSideLength(*pNodesFinal, d_nodeCenterNormals);

	if(m_pSettings->GetICutUseLeafs())
		KernelICutMarkLeafs(*pNodesFinal, d_isCoarseTreeCutNode);
	else
	{
		//static StatTimer& s_timer = StatTimer::Create("Timers", "ICUT: Coarse Cut Selection", false);
		//mncudaSafeCallNoSync(s_timer.Start(true));

		// Estimate radiance at node centers.
		// NOTE: This doesn't require the final kd-tree data!
		static StatTimer& s_timer = StatTimer::Create("Timers", "ICUT: Estimation", false);
		mncudaSafeCallNoSync(s_timer.Start(true));
		KernelICutEstimateNodeIrradiance(*pNodesFinal, d_nodeCenterNormals,
			m_Photons, d_nodeIrrEstimate);
		mncudaSafeCallNoSync(s_timer.Stop(true));

		// Estimate E_min.
		float3 Emin = GetEminForCoarseCut(pNodesFinal);
		//MNMessage("ICUT - Level %i E_min: %.8f, %.8f, %.8f.", m_pSettings->GetICutLevelEmin(), 
		//	Emin.x, Emin.y, Emin.z);

		// Mark coarse cut nodes.
		KernelICutMarkCoarseCut(*pNodesFinal, d_nodeIrrEstimate, Emin, d_isCoarseTreeCutNode);

		//mncudaSafeCallNoSync(s_timer.Stop(true));
	}
}

void PhotonMap::ComputExactNodeRadiances(uint* d_workNodeIdx, uint numWorkNodes)
{
	MNAssert(m_pPhotonMap);
	KDTreeData* pKDData = m_pPhotonMap->GetData();

	MNCudaMemory<float4> d_workQueryPoint(numWorkNodes, "Temporary", 256);
	mncudaSetFromAddress((float4*)d_workQueryPoint, d_workNodeIdx, pKDData->d_nodeExtent, numWorkNodes);

	// Compute query radii first!
	MNCudaMemory<float> d_queryRadius(numWorkNodes);
	m_pPhotonMap->ComputeQueryRadii(d_workQueryPoint, numWorkNodes, d_queryRadius);

	PMSetPhotonMap(*pKDData, m_Photons, m_pPhotonMap->GetKNNTargetCount());
	KernelICutIrrAtNodeCenters(d_workQueryPoint, d_queryRadius, d_workNodeIdx, numWorkNodes, 
		d_nodeCenterNormals, d_nodeIrrExact);
	PMUnsetPhotonMap();
}

void PhotonMap::GenerateFinalIllumCut()
{
	static StatTimer& s_timer = StatTimer::Create("Timers", "ICUT: Final Cut Generation", false);
	mncudaSafeCallNoSync(s_timer.Start(true));

	KDTreeData* pKDData = m_pPhotonMap->GetData();	

	MNCudaMemory<uint> d_isFinalTreeCutNode(pKDData->numNodes);
	MNCudaMemory<uint> d_workNodeIdx(pKDData->numNodes);
	uint* d_workMarks = d_isCoarseTreeCutNode;
	uint numWorkNodes = mncudaGenCompactAddresses(d_workMarks, pKDData->numNodes, d_workNodeIdx);

	//MNMessage("ICUT - Coarse cut nodes: %d.", numWorkNodes);
	MNAssert(numWorkNodes > 0);

	// Initialize exact radiance with zeroes.
	mncudaSafeCallNoSync(cudaMemset(d_nodeIrrExact, 0, pKDData->numNodes*sizeof(float4)));
	// Initialize final cut markers.
	mncudaSafeCallNoSync(cudaMemset(d_isFinalTreeCutNode, 0, pKDData->numNodes*sizeof(uint)));

	uint iter = 0;
	while(iter < m_pSettings->GetICutRefineIters() && numWorkNodes > 0)
	{
		// Compute exact radiance R_p at each cut node center.
		ComputExactNodeRadiances(d_workNodeIdx, numWorkNodes);

		// Update final cut and work markers based on difference between estimated and exact radiance.
		KernelICutUpdateFinalCut(*pKDData, d_nodeIrrEstimate, d_nodeIrrExact,
			d_workNodeIdx, numWorkNodes, m_pSettings->GetICutAccuracy(), d_workMarks, d_isFinalTreeCutNode);

		/*uint finalCutNodes;
		mncudaReduce(finalCutNodes, (uint*)d_isFinalTreeCutNode, pKDData->numNodes, MNCuda_ADD, (uint)0);
		MNMessage("ICUT - final cut nodes after iter %d: %d.", iter, finalCutNodes);*/

		// Update work node indices from new marks. Also update work node count.
		numWorkNodes = mncudaGenCompactAddresses(d_workMarks, pKDData->numNodes, d_workNodeIdx);
		//MNMessage("ICUT - Work nodes after iter %d: %d.", iter, numWorkNodes);
		iter++;
	}

	// Add missing work list nodes to final cut in case there are any.
	// This is required due to the fact that we just perform some iterations.
	if(numWorkNodes > 0)
	{
		// Compute exact values for missing nodes.
		ComputExactNodeRadiances(d_workNodeIdx, numWorkNodes);

		// This works because no final cut node can be a work list node.
		mncudaArrayOp<MNCuda_ADD, uint>(d_isFinalTreeCutNode, d_workMarks, pKDData->numNodes);
	}

	/*uint finalCutNodes;
	mncudaReduce(finalCutNodes, (uint*)d_isFinalTreeCutNode, pKDData->numNodes, MNCuda_ADD, (uint)0);
	MNMessage("ICUT - Final cut nodes: %d.", finalCutNodes);*/
	/*mncudaPrintArray((uint*)d_isFinalTreeCutNode, pKDData->numNodes, false, "");
	exit(-1);*/

	// Now write final cut node markers to parent info (custom bits!) in kd-tree. Note that this does
	// not affect leafs as leafs have no parent info. That's no problem as if we get to a leaf during
	// traversal, we can assume it's a cut node.
	m_pPhotonMap->SetCustomBits(0, d_isFinalTreeCutNode);

	mncudaSafeCallNoSync(s_timer.Stop(true));
}