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

#include "RTCore.h"
#include "CameraModel.h"
#include "kd-tree/KDTreeTri.h"
#include "kd-tree/KDTreePoint.h"
#include "RayPool.h"
#include "MNCudaMT.h"
#include "MNStatContainer.h"
#include "MNCudaPrimitives.h"
#include "PhotonMap.h"
#include "ProgressListener.h"
#include "MNRTSettings.h"
#include "SceneConfig.h"

// Kernel methods
extern "C"
void RTInitializeKernels();
extern "C"
void RTUpdateKernelData(const LightData& lights, const TriangleData& tris, 
						const MaterialData& mats, const KDTreeData& kdTree,
						float fRayEpsilon);
extern "C"
void RTUpdateRayGenKernels(const TriangleData& tris, const MaterialData& mats);
extern "C"
void KernelRTFinalGatherRays(const ShadingPoints& shadingPts,
						     float4* d_clrDiffHit,
						     float* d_randoms1, float* d_randoms2,
						     uint idxFGRayX, uint numFGRaysX,
						     uint idxFGRayY, uint numFGRaysY,
						     RayChunk& outChunk);
extern "C"
void RTCleanupKernelData();
extern "C"
void KernelRTTraceRays(const RayChunk& rayChunk, ShadingPoints& outInters, uint* d_outIsValid);
extern "C"
void KernelRTEvaluateLTE(const RayChunk& rayChunk, const ShadingPoints& shadingPts, const LightData& lights,
					     float4* d_radianceIndirect,
					     bool bDirectRT, bool bTraceShadowRays, uint2 areaLightSamples, 
					     float4* d_ioLuminance);
extern "C"
void KernelRTTracePhotons(PhotonData& photons, uint* d_outIsValid, uint* d_outHasNonSpecular, 
						  int* d_outTriHitIndex, float2* d_outHitBary, 
						  float4* d_outHitDiffClr, float4* d_outHitSpecClr);
extern "C"
void KernelRTGetDiffuseColors(int* d_triHitIndices, float2* d_baryHit, uint numPoints, float4* d_outClrDiffHit);

extern "C"
void PMUpdateBuildData(const LightData& lights);
extern "C"
void KernelPMSpawnLightPhotons(LightType type, uint photonOffset, uint numToSpawn,
							 float3 worldCenter, float worldRadius,
							 PhotonData& outPhotonSpawn);
extern "C"
void KernelPMRussianRoulette(float* d_randoms, float contProbability,
						     PhotonData& ioPhotons, uint* d_ioIsValid);
extern "C"
void KernelPMSpawnScatteredPhotons(PhotonData& ioPhotons,
								   float4* d_normalsG, float4* d_normalsS,
								   float4* d_hitDiffClrs, float4* d_hitSpecClrs,
								   float* d_randoms1, float* d_randoms2, float* d_randoms3,
								   uint* d_outIsLastSpecular, uint* d_outIsValid);

extern "C"
void PMInitializeKernelData();
extern "C"
void PMCleanupKernelData();

// Adaptive Final Gathering (adaptiveFG.cu).
extern "C"
void AFGSetGeoVarAlpha(float geoVarAlpha);
extern "C"
void AFGSetClusterData(const KDTreeData& kdCluster, float4* d_posCluster, float4* d_normCluster,
					   uint numClusters);
extern "C"
void AFGCleanupClusterData();
extern "C"
void AFGSetInterpolationData(float* d_meanReciDists, float4* d_irrCluster, uint numClusters);
extern "C"
void AFGCleanupInterpolationData();
extern "C"
void KernelAFGConstructQT2SPAssoc(const ShadingPoints& spHits, uint numQTLevels,
							      PairList& outAssoc);
extern "C"
void KernelAFGComputeGeometricVariation(const ShadingPoints& spHits, const QuadTreeSP& quadTree,
									    float* d_outGeoVar);
extern "C"
void KernelAFGNormalizeGeoVariation(const QuadTreeSP& quadTree, float fPropagationFactor);
extern "C"
void KernelAFGDistributeSamplesToLevel(float* d_geoVars, float* d_qtNodeCounts,
									   float* d_randoms,
									   uint lvlSrc, uint idxStartLvl, uint numNodesLvl,
									   uint* d_numSamplesSrc, uint* d_outNumSamplesDest);
extern "C"
void KernelAFGCreateInitialClusterList(const QuadTreeSP& quadTree, uint idxStartLeafs, uint numLeafs,
									   uint* d_numSamplesLeaf, uint* d_clusterOffsets, 
									   ClusterList& outClusters);
extern "C"
void KernelAFGConstructCluster2SPAssoc(const ShadingPoints& spHits, const ClusterList& clusters,
									   float queryRadius,
									   PairList& outAssoc,
									   float* d_outGeoVar);
extern "C"
void KernelAFGCheckReclassifciation(const ClusterList& clustersOld, const PairList& cluster2sp,
									float* d_geoVarsPairs,
									uint* d_outIsSPUnclassified, uint* d_outIsClusterNonEmpty);
extern "C"
void KernelAFGVisualizeClusters(const ShadingPoints& spHits, const PairList& cluster2sp, uint numClusters,
							    float4* d_outScreenBuffer);
extern "C"
void KernelAFGMarkNonEmptyClusters(float* d_spCounts, uint numClusters, 
							       uint* d_outClusterMarks);
extern "C"
void KernelAFGGetFinalClusterIndices(float* d_geoVars, float* d_geoVarMinima, uint numClusters,
								     const PairList& cluster2sp, uint* d_outMinIndices);
extern "C"
void KernelAFGVisualizeClusterCenters(const ShadingPoints& clusters, float4* d_outScreenBuffer);
extern "C"
void KernelAFGVisualizeInitialDistribution(ShadingPoints spHits, uint* d_idxShadingPt,
										   uint* d_outNumSamplesQTLeaf, 
										   uint numLeafs, 
										   float4* d_outscreenbuffer);
extern "C"
void KernelAFGBestFitIrradianceDistrib(const ShadingPoints& spHits, const PairList& cluster2sp,
									   uint numClusters, float4* d_irrCluster, float4* d_outIrrSP);
extern "C"
void KernelAFGWangIrradianceDistrib(const ShadingPoints& spHits, float* d_queryRadii, float4* d_outIrrSP);


// BSDF kernels
extern "C"
void BSDFUpdateKernelData(const TriangleData& tris, const MaterialData& mats);
extern "C"
void BSDFCleanupKernelData();
extern "C"
void KernelBSDFGetNormalsAtHit(uint count, int* d_triHitIndices, float2* d_hitBaryCoords,
							   float4* d_outNormalsG, float4* d_outNormalsS);
extern "C"
void KernelBSDFIrradianceToRadiance(ShadingPoints shadingPts, float4* d_inIrradiance,
									float4* d_clrDiffuse, float4* d_ioRadiance);

// Image kernels.
extern "C"
void KernelIMGConvertToRGBA8(float4* d_inRadiance, uint numPixel, uchar4* d_outScreenBuffer);


RTCore::RTCore(MNRTSettings* pSettings)
{
	MNAssert(pSettings);
	m_pSettings = pSettings;
	m_pSC = NULL;
	m_bInited = false;

	m_pRayPool = NULL;
	m_pKDTree = NULL;
	m_pPMGlobal = NULL;
	m_pPMCaustics = NULL;
}

RTCore::~RTCore(void)
{
	Destroy();
}

bool RTCore::Initialize(uint screenSize, SceneConfig* pSceneConfig)
{
	MNAssert(pSceneConfig && IsPowerOf2(screenSize));
	m_pSC = pSceneConfig;

	srand(1337);

	// Initialize and seed Mersenne Twister RNG.
	MNCudaMT& mtw = MNCudaMT::GetInstance();
	if(!mtw.Init("MersenneTwister.dat"))
		return false;
	if(!mtw.Seed(1337))
		return false;

	m_pRayPool = new RayPool();

	// Retrieve scene data.
	m_Tris.Initialize(m_pSC->GetScene());
	m_Materials.Initialize(m_pSC->GetScene());

	// Move the camera so that the scene is visible.
	m_pSC->GetCamera()->SetScreenDimension(screenSize, screenSize);

	m_pRayPool->Initialize(256*1024);

	// Initialize shading point structures.
	m_spShade.Initialize(screenSize*screenSize);
	m_spFinalGather.Initialize(screenSize*screenSize);
	m_spClusters.Initialize(m_pSC->GetWangInitialSamples());
	m_clusterList.Initialize(m_pSC->GetWangInitialSamples());

	// Update kernel constants and data.
	RTInitializeKernels();
	RTUpdateRayGenKernels(m_Tris, m_Materials);
	PMInitializeKernelData();
	PMUpdateBuildData(m_pSC->GetLight());
	BSDFUpdateKernelData(m_Tris, m_Materials);

	MNMessage("Core initialized.");
	m_bInited = true;
	return true;
}

bool RTCore::RebuildObjectKDTree()
{
	if(!m_pSettings->GetDynamicScene() && m_pKDTree)
		return true; // No need to rebuild for static scenes.

	static StatTimer& s_timer = StatTimer::Create("Timers", "Object kd-tree construction", false);
	mncudaSafeCallNoSync(s_timer.Start(true));

	// Destroy old kd-tree first.
	SAFE_DELETE(m_pKDTree);

	m_pKDTree = new KDTreeTri(m_Tris);

	if(!m_pKDTree->BuildTree())
	{
		mncudaSafeCallNoSync(s_timer.Stop(true));
		return false;
	}
	mncudaSafeCallNoSync(s_timer.Stop(true));

	RTUpdateKernelData(m_pSC->GetLight(), m_Tris, m_Materials, *m_pKDTree->GetData(),
		m_pSC->GetRayEpsilon());

	return true;
}

void RTCore::Destroy()
{
	if(!m_bInited)
		return;

	m_spShade.Destroy();
	m_spClusters.Destroy();
	m_spFinalGather.Destroy();

	RTCleanupKernelData();
	PMCleanupKernelData();
	AFGCleanupClusterData();
	BSDFCleanupKernelData();

	m_Materials.Destroy();
	m_Tris.Destroy();

	SAFE_DELETE(m_pPMGlobal);
	SAFE_DELETE(m_pPMCaustics);
	SAFE_DELETE(m_pRayPool);
	SAFE_DELETE(m_pKDTree);
}

bool RTCore::BenchmarkKDTreeCon(uint numWarmup, uint numRuns, float* outTotal_s, float* outAvg_s,
		ProgressListener* pListener/* = NULL*/)
{
	MNAssert(m_Tris.numTris > 0 && numRuns > 0 && outTotal_s && outAvg_s);

	uint timer;
	mncudaCheckErrorCUtil(cutCreateTimer(&timer));

	if(pListener)
		pListener->SetMaximum(numWarmup+numRuns);

	bool bAborted = false;
	for(uint i=0; i<numWarmup+numRuns; i++)
	{
		KDTreeTri kd(m_Tris);

		if(i >= numWarmup)
			mncudaCheckErrorCUtil(cutStartTimer(timer));
		kd.BuildTree();
		if(i >= numWarmup)
			mncudaCheckErrorCUtil(cutStopTimer(timer));

		kd.Destroy();

		if(pListener)
		{
			if(!pListener->Update(i+1))
			{
				bAborted = true;
				break;
			}
		}
	}

	*outTotal_s = cutGetTimerValue(timer);
	*outAvg_s = cutGetAverageTimerValue(timer);

	mncudaCheckErrorCUtil(cutDeleteTimer(timer));

	return bAborted;
}

uint RTCore::CompactPhotonTrace(PhotonData& photonsTrace, int* d_triHitIndices, 
							    float2* d_hitBaryCoords, float4* d_hitDiffClrs, float4* d_hitSpecClrs, 
							    uint* d_isValid)
{
	uint countOld = photonsTrace.numPhotons;

	MNCudaMemory<uint> d_srcAddr(countOld);
	uint countNew = mncudaGenCompactAddresses(d_isValid, countOld, d_srcAddr);
	if(countNew == 0)
		return 0;

	photonsTrace.CompactSrcAddr(d_srcAddr, countNew);
	
	// Compact misc data.
	mncudaCompactInplace(d_triHitIndices, d_srcAddr, countOld, countNew);
	mncudaCompactInplace(d_hitBaryCoords, d_srcAddr, countOld, countNew);
	mncudaCompactInplace(d_hitDiffClrs, d_srcAddr, countOld, countNew);
	mncudaCompactInplace(d_hitSpecClrs, d_srcAddr, countOld, countNew);

	return countNew;
}

uint RTCore::CompactClusters(ClusterList& clusters, uint* d_isValid)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	MNCudaMemory<uint> d_srcAddr(clusters.numClusters);
	uint countNew = mncudaGenCompactAddresses(d_isValid, clusters.numClusters, d_srcAddr);

	if(countNew == 0)
	{
		clusters.numClusters = 0;
		return 0;
	}
	if(countNew == clusters.numClusters)
		return clusters.numClusters;

	// Now move source data to destination data.
	mncudaCompactInplace(clusters.d_positions, d_srcAddr, clusters.numClusters, countNew);
	mncudaCompactInplace(clusters.d_normals, d_srcAddr, clusters.numClusters, countNew);

	// No need to compact this as it is calculated later and not yet available.
	//mncudaCompactInplace(clusters.d_idxShadingPt, d_srcAddr, clusters.numClusters, countNew);

	// Update count.
	clusters.numClusters = countNew;

	return countNew;
}

bool RTCore::BuildPhotonMapKDTrees()
{
	static StatTimer& s_timer = StatTimer::Create("Timers", "Photon map construction (+ICUT)", false);
	mncudaSafeCallNoSync(s_timer.Start(true));

	// Build photon maps (kd-trees).
	if(m_pPMCaustics)
		if(!m_pPMCaustics->BuildMap())
			return false;
	if(m_pPMGlobal)
		if(!m_pPMGlobal->BuildMap())
			return false;

	mncudaSafeCallNoSync(s_timer.Stop(true));

	static StatCounter& ctrGlobal = StatCounter::Create("Photon map", "Stored photons (global)");
	if(m_pPMGlobal)
		ctrGlobal += m_pPMGlobal->GetPhotonCount();
	static StatCounter& ctrCaustics = StatCounter::Create("Photon map", "Stored photons (caustics)");
	if(m_pPMCaustics)
		ctrCaustics += m_pPMCaustics->GetPhotonCount();

	return true;
}

uint RTCore::TraceRays(RayChunk* pRayChunk, ShadingPoints* pSP, const std::string& strCategory, uint* d_outSrcAddr/* = NULL*/)
{
	StatCounter& ctrTracedCat = StatCounter::Create("Ray tracing", "Traced rays (" + strCategory + ")");
	MNCudaMemory<uint> d_isValid(pRayChunk->numRays);
	StatTimer& s_tRT = StatTimer::Create("Timers", "Raytracing (" + strCategory + ")", false);
	mncudaSafeCallNoSync(s_tRT.Start(true));

	// Do tracing.
	KernelRTTraceRays(*pRayChunk, *pSP, d_isValid);
	ctrTracedCat += pRayChunk->numRays;
	uint traced = pRayChunk->numRays;

	// Compact shading points and ray chunk to avoid rays that hit nothing.
	if(d_outSrcAddr)
	{
		uint countNew = mncudaGenCompactAddresses(d_isValid, pSP->numPoints, d_outSrcAddr);
		pSP->CompactSrcAddr(d_outSrcAddr, countNew);
		pRayChunk->CompactSrcAddr(d_outSrcAddr, countNew);
	}
	else
	{
		MNCudaMemory<uint> d_srcAddr(pSP->numPoints);
		uint countNew = mncudaGenCompactAddresses(d_isValid, pSP->numPoints, d_srcAddr);
		pSP->CompactSrcAddr(d_srcAddr, countNew);
		pRayChunk->CompactSrcAddr(d_srcAddr, countNew);
	}

	// Create normals for shading points.
	if(pSP->numPoints != 0)
		KernelBSDFGetNormalsAtHit(pSP->numPoints, pSP->d_idxTris, pSP->d_baryHit,
			pSP->d_normalsG, pSP->d_normalsS);

	mncudaSafeCallNoSync(s_tRT.Stop(true));

	return traced;
}

void RTCore::SpawnLightPhotons(PhotonData& outPhotons, uint photonOffset, uint numToSpawn)
{
	// Get world center and radius.
	MNBBox bounds = m_pSC->GetSceneBounds();
	float3 worldCenter = *(float3*)&bounds.GetCenter();
	float worldRadius = (bounds.ptMax - bounds.ptMin).Length() * 0.5f;

	// Spawn light photons into outPhotons.
	outPhotons.numPhotons = 0;
	KernelPMSpawnLightPhotons(m_pSC->GetLight().type, photonOffset, numToSpawn, worldCenter, worldRadius, 
		outPhotons);
}

bool RTCore::RebuildPhotonMaps()
{
	if(!m_pSettings->GetDynamicScene() && (m_pPMGlobal || m_pPMCaustics))
		return true; // No need to rebuild for static scenes.

	// Destroy old photon maps.
	SAFE_DELETE(m_pPMGlobal);
	SAFE_DELETE(m_pPMCaustics);

	if(m_pSettings->GetPhotonMapMode() == PhotonMap_Disabled)
		return true; // Nothing to do.

	// Just spawn a fixed number of photons per pass. This allows us to approximate the target counts.
	uint nPhotonsPerLight = 20000;
	int targetSC;
	targetSC = m_pSC->GetTargetCountGlobal();
	uint targetGlobal = targetSC;
	if(targetSC < 0)
		targetGlobal = m_pSettings->GetTargetCountGlobal();
	targetSC = m_pSC->GetTargetCountCaustics();
	uint targetCaustics = targetSC;
	if(targetSC < 0)
		targetCaustics = m_pSettings->GetTargetCountCaustics();
	bool bGlobal = targetGlobal > 0;
	// Disable caustics photon map for now.
	bool bCaustics = false; //m_pSC->GetHasSpecular() && targetCaustics > 0;

	// Initialize photon maps.
	uint maxGlobal = targetGlobal + nPhotonsPerLight; // Add some space in case we get more photons.
	uint maxCaustics = targetCaustics + nPhotonsPerLight; 
	if(bGlobal)
	{
		m_pPMGlobal = new PhotonMap(maxGlobal, m_Tris.aabbMin, m_Tris.aabbMax, 
			m_pSC->GetRadiusPMapMax(), m_pSettings);
		m_pPMGlobal->SetKNNTargetCount(m_pSettings->GetKinKNNSearchGlobal());
	}
	if(bCaustics)
	{
		m_pPMCaustics = new PhotonMap(maxCaustics, m_Tris.aabbMin, m_Tris.aabbMax, 
			m_pSC->GetRadiusPMapMax(), m_pSettings);
		m_pPMCaustics->SetKNNTargetCount(m_pSettings->GetKinKNNSearchCaustics());
	}

	// Spawned photon storage.
	PhotonData photonSpawn;
	photonSpawn.Initialize(nPhotonsPerLight);

	// Get number of randoms to compute. Has to be aligned.
	MNCudaMT& mtw = MNCudaMT::GetInstance();
	uint seed = 545;
	uint numRnd = mtw.GetAlignedCount(nPhotonsPerLight);

	MNCudaMemory<uint> d_isValid(nPhotonsPerLight);
	MNCudaMemory<uint> d_hasNonSpecular(nPhotonsPerLight);
	MNCudaMemory<uint> d_isLastSpecular(nPhotonsPerLight);
	MNCudaMemory<int> d_triHitIndices(nPhotonsPerLight);
	MNCudaMemory<float2> d_hitBaryCoords(nPhotonsPerLight, "Temporary", 128);
	MNCudaMemory<float4> d_hitDiffClrs(nPhotonsPerLight, "Temporary", 256);
	MNCudaMemory<float4> d_hitSpecClrs(nPhotonsPerLight, "Temporary", 256);
	MNCudaMemory<float> d_randoms(3*numRnd);
	MNCudaMemory<uint> d_doMerge(nPhotonsPerLight);
	MNCudaMemory<float4> d_normalsG(nPhotonsPerLight, "Temporary", 256), 
		d_normalsS(nPhotonsPerLight, "Temporary", 256);

	static StatCounter& ctrTraced = StatCounter::Create("Photon map", "Traced photons (total)");
	static StatRatio& ctrPhPerBounce = StatRatio::Create("Photon map", "Photons stored per bounce");
	static StatTimer& s_tPTracing = StatTimer::Create("Timers", "Photon tracing", false);
	mncudaSafeCallNoSync(s_tPTracing.Start(true));

	// Stop after some rounds to avoid endless loop.
	bool recordGlobal = bGlobal, recordCaustics = bCaustics;
	uint shot = 0, shotForGlobal = 0, shotForCaustics = 0;
	while((recordGlobal || recordCaustics) && shot < 100*std::max(targetGlobal, targetCaustics))
	{
		// Spawn light photons.
		SpawnLightPhotons(photonSpawn, shotForCaustics, nPhotonsPerLight);
		shot += photonSpawn.numPhotons;
		if(recordGlobal)
			shotForGlobal += photonSpawn.numPhotons;
		if(recordGlobal)
			shotForCaustics += photonSpawn.numPhotons;

		// +1 since we need to trace after the last bounce, too.
		for(uint i=0; i<m_pSettings->GetMaxPhotonBounces()+1; i++)
		{
			// Trace photons. This works inplace for photonSpawn.
			KernelRTTracePhotons(photonSpawn, d_isValid, d_hasNonSpecular, d_triHitIndices, 
				d_hitBaryCoords, d_hitDiffClrs, d_hitSpecClrs);
			ctrTraced += photonSpawn.numPhotons;

			// After a few bounces, perform Russian roulette to terminate paths randomly.
			if(i > 3)
			{
				// Pregenerate random numbers using MT.
				mtw.Seed(seed++);
				uint numRndRR = mtw.GetAlignedCount(photonSpawn.numPhotons);
				mncudaSafeCallNoSync(mtw.Generate(d_randoms, numRndRR));

				KernelPMRussianRoulette(d_randoms, 0.5f, photonSpawn, d_isValid);
			}

			uint newPhotonsC = 0, newPhotonsG = 0;
			if(i == 0 && recordGlobal)
			{
				// Direct illumination goes to global photon map since we perform final gathering.
				newPhotonsG = m_pPMGlobal->Merge(photonSpawn, d_hasNonSpecular);
			}
			else if(i > 0)
			{
				// Merge into caustics photon map data first when
				// isLastSpecular && hasNonSpecular.
				mncudaSafeCallNoSync(cudaMemcpy(d_doMerge, d_isLastSpecular, 
					photonSpawn.numPhotons*sizeof(uint), cudaMemcpyDeviceToDevice));
				mncudaArrayOp<MNCuda_MUL, uint>(d_doMerge, d_hasNonSpecular, photonSpawn.numPhotons);

				if(recordCaustics)
					newPhotonsC = m_pPMCaustics->Merge(photonSpawn, d_doMerge);

				if(recordGlobal)
				{
					// Merge new photon hits into global photon map data in case
					//   !(isLastSpecular && hasNonSpecular) && hasNonSpecular
					// = (!isLastSpecular || !hasNonSpecular) && hasNonSpecular
					// = !isLastSpecular && hasNonSpecular.
					mncudaInverseBinary(d_doMerge, photonSpawn.numPhotons);
					mncudaArrayOp<MNCuda_MUL, uint>(d_doMerge, d_hasNonSpecular, photonSpawn.numPhotons);
					newPhotonsG = m_pPMGlobal->Merge(photonSpawn, d_doMerge);
				}
			}
			ctrPhPerBounce.Increment(newPhotonsC + newPhotonsG, 1);

			// Update states.
			recordGlobal = bGlobal && m_pPMGlobal->GetPhotonCount() < targetGlobal;
			recordCaustics = bCaustics && m_pPMCaustics->GetPhotonCount() < targetCaustics;
			if(!recordGlobal && !recordCaustics)
				break; // No more photons needed.

			// Build list of photons we still have to trace. Some photons could have left 
			// the scene and are invalid then. They are marked by d_isValid[i] = 0.
			// Also compact the other data, so we can operate on valid hits only.
			uint numToTrace = CompactPhotonTrace(photonSpawn, d_triHitIndices, 
				d_hitBaryCoords, d_hitDiffClrs, d_hitSpecClrs, d_isValid);
			if(numToTrace == 0)
				break; // No more photons left. So spawn new photons.

			// Now spawn scattered photons.
			if(i < m_pSettings->GetMaxPhotonBounces())
			{
				// Generate normals for spawning new photons.
				KernelBSDFGetNormalsAtHit(photonSpawn.numPhotons, d_triHitIndices, d_hitBaryCoords,
					d_normalsG, d_normalsS);

				// Pregenerate random numbers using MT.
				mtw.Seed(seed++);
				mncudaSafeCallNoSync(mtw.Generate(d_randoms, 3*numRnd));

				// Spawn new photons inplace.
				KernelPMSpawnScatteredPhotons(photonSpawn, d_normalsG, d_normalsS, d_hitDiffClrs, d_hitSpecClrs,
					d_randoms, d_randoms+numRnd, d_randoms+2*numRnd, d_isLastSpecular, d_isValid);
				// Compact to remove invalid photons.
				uint* d_srcAddr = d_doMerge;
				uint countNew = mncudaGenCompactAddresses(d_isValid, photonSpawn.numPhotons, d_srcAddr);
				if(countNew != 0)
					photonSpawn.CompactSrcAddr(d_srcAddr, countNew);
			}
		}
	}

	photonSpawn.Destroy();

	// Scale photon flux by number of photons shot to get to the number of photons in
	// each photon map.
	if(bGlobal)
		m_pPMGlobal->ScaleFlux(1.f / float(shotForGlobal));
	if(bCaustics)
		m_pPMCaustics->ScaleFlux(1.f / float(shotForCaustics));

	mncudaSafeCallNoSync(s_tPTracing.Stop(true));

	// Finally, build kd-trees from photons.
	if(!BuildPhotonMapKDTrees())
		return false;

	return true;
}

bool RTCore::RenderToBuffer(float4* d_outLuminance)
{
	MNAssert(d_outLuminance);
	CameraModel* pCamera = m_pSC->GetCamera();
	uint screenW = pCamera->GetScreenWidth();
	uint screenH = pCamera->GetScreenHeight();

	// Reset target buffer to black.
	mncudaSafeCallNoSync(cudaMemset(d_outLuminance, 0, screenW*screenH*sizeof(float4)));
	
	// Generate primary rays for current camera position.
	bool useMultiSample = m_pSettings->GetViewMode() == MNRTView_Result;
	m_pRayPool->AddPrimaryRays(pCamera, useMultiSample);

	uint2 areaLightSamples = make_uint2(m_pSettings->GetAreaLightSamplesX(), m_pSettings->GetAreaLightSamplesY());

	while(m_pRayPool->hasMoreRays())
	{
		// Get more rays from ray pool. This performs synchronization.
		RayChunk* pChunk = m_pRayPool->GetNextChunk();

		// Trace rays. Result is compacted.
		TraceRays(pChunk, &m_spShade, "Primary, secondary");
		if(m_spShade.numPoints == 0)
		{
			// Nothing to do.
			m_pRayPool->FinalizeChunk(pChunk);
			continue;
		}

		PairList cluster2sp;
		cluster2sp.Initialize(m_spShade.numPoints);

		// Final gather result.
		MNCudaMemory<float4> d_radianceIndirect(m_spShade.numPoints, "Temporary", 256); // 256 byte alignment!
		mncudaSafeCallNoSync(cudaMemset(d_radianceIndirect, 0, m_spShade.numPoints*sizeof(float4)));

		uint numInitSamples = m_pSC->GetWangInitialSamples();
		if(m_pSettings->GetViewMode() == MNRTView_Cluster)
		{
			SelectIrradianceSamples(m_spShade, numInitSamples, m_clusterList, cluster2sp, m_spClusters, NULL);
			KernelAFGVisualizeClusters(m_spShade, cluster2sp, m_clusterList.numClusters, d_outLuminance);
		}
		else if(m_pSettings->GetViewMode() == MNRTView_ClusterCenters)
		{
			SelectIrradianceSamples(m_spShade, numInitSamples, m_clusterList, 
				cluster2sp, m_spClusters, d_outLuminance);
			KernelRTEvaluateLTE(*pChunk, m_spShade, m_pSC->GetLight(), d_radianceIndirect, 
				m_pSettings->GetEnableDirectRT(), m_pSettings->GetEnableShadowRays(), 
				areaLightSamples, d_outLuminance);
			KernelAFGVisualizeClusterCenters(m_spClusters, d_outLuminance);
		}
		else if(m_pSettings->GetViewMode() == MNRTView_InitialSamples)
		{
			SelectIrradianceSamples(m_spShade, numInitSamples, m_clusterList, 
				cluster2sp, m_spClusters, d_outLuminance);
		}
		else
		{
			if(m_pSettings->GetPhotonMapMode() == PhotonMap_Visualize)
			{
				if(m_pPMCaustics)
					m_pPMCaustics->Visualize(m_spShade, *pChunk, d_radianceIndirect);
				if(m_pPMGlobal)
					m_pPMGlobal->Visualize(m_spShade, *pChunk, d_radianceIndirect);
			}
			else if(m_pSettings->GetPhotonMapMode() != PhotonMap_Disabled)
			{
				bool doFinalGather = m_pSettings->GetPhotonMapMode() == PhotonMap_FullFinalGather ||
									 m_pSettings->GetPhotonMapMode() == PhotonMap_AdaptiveSamplesBestFit ||
									m_pSettings-> GetPhotonMapMode() == PhotonMap_AdaptiveSamplesWang;


				// Indirect illumination using final gathering and global photon map.
				if(doFinalGather)
				{
					if(!DoFinalGathering(m_spShade, d_radianceIndirect))
					{
						cluster2sp.Destroy();
						m_pRayPool->FinalizeChunk(pChunk);
						return false;
					}
				}
				
				// Caustics using caustic photon map.
				if(m_pPMCaustics)
				{
					static StatTimer& s_tCaustics = StatTimer::Create("Timers", "Caustics gathering", false);
					mncudaSafeCallNoSync(s_tCaustics.Start(true));

					// Get diffuse colors at found shading points for gathering.
					MNCudaMemory<float4> d_clrDiffGather(m_spShade.numPoints, "Temporary", 256);
					KernelRTGetDiffuseColors(m_spShade.d_idxTris, m_spShade.d_baryHit, 
						m_spShade.numPoints, d_clrDiffGather);

					m_pPMCaustics->Gather(m_spShade, *pChunk, d_clrDiffGather, d_radianceIndirect);

					mncudaSafeCallNoSync(s_tCaustics.Stop(true));
				}
			}

			static StatTimer& s_tDirect = StatTimer::Create("Timers", "Shadow rays + Shading", false);
			mncudaSafeCallNoSync(s_tDirect.Start(true));

			// Shade using raytracing.
			KernelRTEvaluateLTE(*pChunk, m_spShade, m_pSC->GetLight(), d_radianceIndirect, 
				m_pSettings->GetEnableDirectRT(), m_pSettings->GetEnableShadowRays(), 
				areaLightSamples, d_outLuminance);

			mncudaSafeCallNoSync(s_tDirect.Stop(true));

			// Transform the intersections into new child rays.
			if(m_pSettings->GetEnableDirectRT())
				m_pRayPool->GenerateChildRays(pChunk, m_spShade, m_Tris, 
					m_pSettings->GetEnableSpecReflect(), m_pSettings->GetEnableSpecTransmit());
		}

		cluster2sp.Destroy();
		m_pRayPool->FinalizeChunk(pChunk);
	}

	// Scale result by sum of filter weights.
	uint samplesPerPixel = 1;
	if(useMultiSample)
		samplesPerPixel = m_pRayPool->GetSamplesPerPixel();
	if(samplesPerPixel > 1)
		mncudaScaleVectorArray((float4*)d_outLuminance, screenW*screenH, 
			1.f / float(samplesPerPixel));

	m_pRayPool->Clear();

	return true;
}

bool RTCore::RenderScene(uchar4* d_screenBuffer)
{
	if(!m_bInited)
		return false;

	// Rebuild if required.
	if(!RebuildObjectKDTree())
	{
		MNError("Building object kd-tree failed.");
		return false;
	}	
	if(!RebuildPhotonMaps()) // We have to have the triangle data inside of the kernels to do this!
	{
		MNError("Building photon maps failed.");
		return false;
	}
	
	// Render to luminance buffer.
	uint screenW = m_pSC->GetCamera()->GetScreenWidth();
	uint screenH = m_pSC->GetCamera()->GetScreenHeight();
	MNCudaMemory<float4> d_Luminance(screenW*screenH, "Temporary", 256);
	if(!RenderToBuffer(d_Luminance))
		return false;
	
	// Convert luminance (float4 array) to screen buffer (uchar4).
	KernelIMGConvertToRGBA8(d_Luminance, screenW*screenH, d_screenBuffer);

	static StatCounter& ctrFrames = StatCounter::Create("General", "Frames Rendered");
	++ctrFrames;

	return true;
}

bool RTCore::DoFinalGathering(const ShadingPoints& spHits, float4* d_ioRadiance)
{
	MNCudaMemory<float4> d_irradianceSP(spHits.numPoints, "Temporary", 256); // 256 byte alignment!
	mncudaSafeCallNoSync(cudaMemset(d_irradianceSP, 0, spHits.numPoints*sizeof(float4)));

	// NOTE: We use full final gathering anytime when the sample count is too small.
	if(m_pSettings->GetPhotonMapMode() == PhotonMap_FullFinalGather || 
		spHits.numPoints < 2*m_pSC->GetWangInitialSamples())
	{
		FullFinalGathering(spHits, m_pSettings->GetFinalGatherRaysX(), 
			m_pSettings->GetFinalGatherRaysY(), d_irradianceSP);
	}
	else if(m_pSettings->GetPhotonMapMode() == PhotonMap_AdaptiveSamplesBestFit || 
			m_pSettings->GetPhotonMapMode() == PhotonMap_AdaptiveSamplesWang)
	{
		UpdateIrradianceSamples(spHits);
		if(!SelectiveFinalGathering(spHits, m_spClusters, d_irradianceSP))
			return false;
	}

	// Get diffuse hit colors for BRDF computations.
	MNCudaMemory<float4> d_clrDiffHit(spHits.numPoints, "Temporary", 256); // 256 byte alignment!
	KernelRTGetDiffuseColors(spHits.d_idxTris, spHits.d_baryHit, spHits.numPoints, d_clrDiffHit);

	// Convert irradiance to radiance by combining it with the BSDF at the hit point.
	KernelBSDFIrradianceToRadiance(spHits, d_irradianceSP, d_clrDiffHit, d_ioRadiance);

	return true;
}

void RTCore::FullFinalGathering(const ShadingPoints& shadingPts,
							    uint numFGSamplesX, uint numFGSamplesY,
							    float4* d_outIrradiance, float* d_outMeanReciDists/* = NULL*/)
{
	MNAssert(numFGSamplesX > 0 && numFGSamplesY > 0 && shadingPts.numPoints > 0);
	MNAssert(m_pPMGlobal != NULL || m_pPMCaustics != NULL);
	MNCudaMT& mtw = MNCudaMT::GetInstance();

	static StatTimer& s_tFG = StatTimer::Create("Timers", "Final gathering", false);
	mncudaSafeCallNoSync(s_tFG.Start(true));

	uint numTotalSamples = numFGSamplesX*numFGSamplesY;

	// Get number of randoms to compute. Has to be aligned.
	uint offsetRnd = MNCUDA_ALIGN(shadingPts.numPoints);
	// NOTE: I get curious bars within the irradiance result for one FG sample when using
	//       two Generate(). Those bars are reduced when using only one Generate(). Seems
	//		 that the MT series from two seeds are somehow dependent. Furthermore elimination
	//		 of patterns is given when avoiding the first numbers in the series.
	// NOTE: I got patterns (bars) in the generated ray visualization when just using the first
	//		 generated randoms from the CUDA SDK MT implementation. Better results were obtained
	//		 when just using some later values of the MT series.
	uint numRnd = mtw.GetAlignedCount(2*offsetRnd);
	MNCudaMemory<float> d_randoms(numRnd);
	MNCudaMemory<uint> d_idxShadingPoint(shadingPts.numPoints);

	// Initialize mean reciprocal accumulator with zero.
	if(d_outMeanReciDists)
		mncudaSafeCallNoSync(cudaMemset(d_outMeanReciDists, 0, shadingPts.numPoints*sizeof(float)));

	// Create a ray chunk for the final gather rays.
	RayChunk raysFG;
	raysFG.Initialize(shadingPts.numPoints);

	// Get diffuse hit colors for BRDF computations.
	MNCudaMemory<float4> d_clrDiffHit(shadingPts.numPoints, "Temporary", 256); // 256 byte alignment!
	KernelRTGetDiffuseColors(shadingPts.d_idxTris, shadingPts.d_baryHit, shadingPts.numPoints, d_clrDiffHit);

	static StatCounter& ctrTracedFG = StatCounter::Create("Ray tracing", "Traced rays (final gathering)");
	for(uint x=0; x<numFGSamplesX; x++)
	{
		for(uint y=0; y<numFGSamplesY; y++)
		{
			// Pregenerate random numbers using MT.
			mtw.Seed(rand());
			mncudaSafeCallNoSync(mtw.Generate(d_randoms, numRnd));

			// Sample random directions for gather rays.
			static StatTimer& s_tFGRTCreate = StatTimer::Create("Timers", "Final gathering (ray creation)", false);
			mncudaSafeCallNoSync(s_tFGRTCreate.Start(true));
			KernelRTFinalGatherRays(shadingPts, d_clrDiffHit, d_randoms, d_randoms+offsetRnd, 
				x, numFGSamplesX, y, numFGSamplesY, raysFG);
			mncudaSafeCallNoSync(s_tFGRTCreate.Stop(true));

			// Trace final gather rays. Result is compacted. Also get the source addresses, that is
			// the source shading point indices.
			static StatTimer& s_tFGRT = StatTimer::Create("Timers", "Final gathering (ray tracing)", false);
			mncudaSafeCallNoSync(s_tFGRT.Start(true));
			m_spFinalGather.Clear();
			ctrTracedFG += TraceRays(&raysFG, &m_spFinalGather, "Final gathering", d_idxShadingPoint);
			if(m_spFinalGather.numPoints == 0)
			{
				// No valid rays...
				mncudaSafeCallNoSync(s_tFGRT.Stop(true));
				m_spFinalGather.Clear();
				continue;
			}
			mncudaSafeCallNoSync(s_tFGRT.Stop(true));

			// Get diffuse colors at found shading points for gathering.
			MNCudaMemory<float4> d_clrDiffGatherAt(m_spFinalGather.numPoints, "Temporary", 256);
			KernelRTGetDiffuseColors(m_spFinalGather.d_idxTris, m_spFinalGather.d_baryHit, 
				m_spFinalGather.numPoints, d_clrDiffGatherAt);

			static StatTimer& s_tFGGather = StatTimer::Create("Timers", "Final gathering (gathering)", false);
			mncudaSafeCallNoSync(s_tFGGather.Start(true));

			// Since intersections are the same, there is no need to record the
			// mean reciprocal distances twice.

			// Global photon map gathering (direct and indirect photons).
			if(m_pPMGlobal)
				m_pPMGlobal->GatherFor(m_spFinalGather, raysFG, d_clrDiffGatherAt,
					d_idxShadingPoint, d_outIrradiance, d_outMeanReciDists);

			// Caustics photon map gathering.
			if(m_pPMCaustics)
				m_pPMCaustics->GatherFor(m_spFinalGather, raysFG, d_clrDiffGatherAt,
					d_idxShadingPoint, d_outIrradiance, (m_pPMGlobal ? NULL : d_outMeanReciDists));

			mncudaSafeCallNoSync(s_tFGGather.Stop(true));

			m_spFinalGather.Clear();
		}
	}

	// Now we only have sums of reciprocal distances in d_outMeanReciDists. We need to divide them
	// by the sample count to get the inverse of the harmonic mean.
	// See Ward's "A Ray Tracing Solution for Diffuse Interreflection", 1988.
	if(d_outMeanReciDists)
	{
		// Any ray that hits nothing would have a 1/distance of zero and is handled in the correct way!
		uint mrdSamples = numTotalSamples;

		// If there are no samples taken, do nothing.
		float invFGSamples = 1.f / float(mrdSamples);
		mncudaConstantOp<MNCuda_MUL, float>(d_outMeanReciDists, shadingPts.numPoints, invFGSamples);
	}

	raysFG.Destroy();	

	mncudaSafeCallNoSync(s_tFG.Stop(true));
}

bool RTCore::SelectiveFinalGathering(const ShadingPoints& spHits,
									 const ShadingPoints& spClusters, 
							         float4* d_outIrradiance)
{
	if(m_clusterList.numClusters == 0 || m_clusterList.numClusters != spClusters.numPoints)
	{
		MNError("Illegal cluster center list for selective final gathering.");
		return false;
	}

	// Sample radiance for selected clusters only.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	MNCudaMemory<float4> d_fgIrradiance(spClusters.numPoints, "Temporary", pool.GetTextureAlignment());
	mncudaSafeCallNoSync(cudaMemset(d_fgIrradiance, 0, spClusters.numPoints*sizeof(float4)));
	MNCudaMemory<float> d_meanReciDists(spClusters.numPoints, "Temporary", pool.GetTextureAlignment());

	FullFinalGathering(spClusters, m_pSettings->GetFinalGatherRaysX(), 
		m_pSettings->GetFinalGatherRaysY(), d_fgIrradiance, d_meanReciDists);
	
	static StatTimer& s_tInterpol = StatTimer::Create("Timers", "FG sample interpolation", false);
	mncudaSafeCallNoSync(s_tInterpol.Start(true));

	if(m_pSettings->GetPhotonMapMode() == PhotonMap_AdaptiveSamplesBestFit)
	{
		PairList cluster2sp;
		cluster2sp.Initialize(spHits.numPoints);
		ClassifyShadingPoints(spHits, m_clusterList, cluster2sp, NULL);

		// Assign cluster's irradiance to all cluster members.
		KernelAFGBestFitIrradianceDistrib(spHits, cluster2sp, m_clusterList.numClusters,
			d_fgIrradiance, d_outIrradiance);

		cluster2sp.Destroy();
	}
	else if(m_pSettings->GetPhotonMapMode() == PhotonMap_AdaptiveSamplesWang)
	{
		// kd-tree for cluster centers.
		KDTreePoint kdtreeClusters(spClusters.d_ptInter, spClusters.numPoints,
			m_Tris.aabbMin, m_Tris.aabbMax, m_pSC->GetRadiusWangInterpolMax());
		kdtreeClusters.SetKNNTargetCount(20);
		kdtreeClusters.BuildTree();

		MNCudaMemory<float> d_queryRadii(spHits.numPoints);
		kdtreeClusters.ComputeQueryRadii(spHits.d_ptInter, spHits.numPoints, d_queryRadii);

		AFGSetClusterData(*kdtreeClusters.GetData(), spClusters.d_ptInter, spClusters.d_normalsS,
			spClusters.numPoints);
		AFGSetInterpolationData(d_meanReciDists, d_fgIrradiance, spClusters.numPoints);

		KernelAFGWangIrradianceDistrib(spHits, d_queryRadii, d_outIrradiance);

		AFGCleanupClusterData();
		AFGCleanupInterpolationData();
		kdtreeClusters.Destroy();
	}

	mncudaSafeCallNoSync(s_tInterpol.Stop(true));
	return true;
}

void RTCore::UpdateIrradianceSamples(const ShadingPoints& spHits)
{
	uint maxSamples = m_pSC->GetWangInitialSamples();

	PairList cluster2SP;
	cluster2SP.Initialize(spHits.numPoints);

	// Just rebuild the complete cluster list in case we have no clusters, e.g. first frame.
	if(m_clusterList.numClusters == 0)
		SelectIrradianceSamples(spHits, maxSamples, m_clusterList, cluster2SP, m_spClusters, NULL);
	else
	{
		MNCudaMemory<float> d_geoVarsPairs(spHits.numPoints);
		MNCudaMemory<uint> d_isSPUnclassified(spHits.numPoints);
		MNCudaMemory<uint> d_isClusterNonEmpty(m_clusterList.numClusters);
		MNCudaMemory<uint> d_srcAddr(spHits.numPoints); // Used for both sp and clusters.

		// Try to classify new shading points to old clusters. Shading points that could not be
		// classified are moved into the virutal cluster m_clusterList.numClusters+1.
		ClassifyShadingPoints(spHits, m_clusterList, cluster2SP, d_geoVarsPairs);

		// Check reclassification and identify unclassified shading points and empty old clusters.
		KernelAFGCheckReclassifciation(m_clusterList, cluster2SP, d_geoVarsPairs,
			d_isSPUnclassified, d_isClusterNonEmpty);

		// Remove empty clusters from cluster list and corresponding shading point list.
		uint countNew = mncudaGenCompactAddresses(d_isClusterNonEmpty, m_clusterList.numClusters, d_srcAddr);
		m_clusterList.CompactSrcAddr(d_srcAddr, countNew);
		m_spClusters.CompactSrcAddr(d_srcAddr, countNew);

		// Check how many shading points are left unassociated. Accept old cluster list when there
		// are just to few of such shading points.
		uint countUnclass = mncudaGenCompactAddresses(d_isSPUnclassified, spHits.numPoints, d_srcAddr);
		//MNMessage("Unclassified: %d.", countUnclass);
		if(countUnclass == 0 || countUnclass < 2000) // TODO: Improve this.
		{
			cluster2SP.Destroy();
			return;
		}

		// Check if there are not enough free cluster entries for countUnclass. In that case, just
		// recompute the full cluster list.
		if(countUnclass > uint(0.5f*float(spHits.numPoints)))
		{
			SelectIrradianceSamples(spHits, maxSamples, m_clusterList, cluster2SP, m_spClusters, NULL);
			cluster2SP.Destroy();
			return;
		}
		// Check if we still have enough clusters. Avoid overhead of computing only very few
		// new clusters.
		if(m_clusterList.numClusters > uint(0.90f*float(maxSamples)))
		{
			cluster2SP.Destroy();
			return;
		}

		// Create a temporary shading point structure for those shading points that are
		// left unclassified.
		ShadingPoints spUnclassified;
		spUnclassified.Initialize(countUnclass);
		spUnclassified.SetFrom(spHits, d_srcAddr, countUnclass);

		// Generate new clusters (samples) for remaining shading points.
		uint maxNewSamples = maxSamples - m_clusterList.numClusters;
		ClusterList clustersNew;
		ShadingPoints spClustersNew;
		clustersNew.Initialize(maxNewSamples);
		spClustersNew.Initialize(maxNewSamples);
		SelectIrradianceSamples(spUnclassified, maxNewSamples, clustersNew,
			cluster2SP, spClustersNew, NULL);

		// Merge new clusters/shading points into old cluster structure.
		m_clusterList.Add(clustersNew);
		m_spClusters.Add(spClustersNew);

		spClustersNew.Destroy();
		clustersNew.Destroy();
		spUnclassified.Destroy();
	}

	cluster2SP.Destroy();
}

void RTCore::SelectIrradianceSamples(const ShadingPoints& spHits, uint numSamplesToSelect, 
									 ClusterList& outClusters, PairList& outCluster2SP, 
									 ShadingPoints& outSPClusters, float4* d_outScreenBuffer)
{
	static StatTimer& s_tAdSeed = StatTimer::Create("Timers", "Adaptive Sample Seeding", false);
	static StatTimer& s_tKMeans = StatTimer::Create("Timers", "k-Means Clustering", false);

	// Ensure output structures are empty.
	outClusters.Clear();
	outCluster2SP.Clear();
	outSPClusters.Clear();

	// Set parameters.
	MNBBox boundsScene = m_pSC->GetSceneBounds();
	AFGSetGeoVarAlpha(m_pSettings->GetGeoVarAlpha(boundsScene));

	// Seed initial clusters.
	mncudaSafeCallNoSync(s_tAdSeed.Start());
	PerformAdaptiveSeeding(spHits, numSamplesToSelect, outClusters, d_outScreenBuffer);
	mncudaSafeCallNoSync(s_tAdSeed.Stop());

	// Perform some k-means iterations.
	mncudaSafeCallNoSync(s_tKMeans.Start());
	PerformKMeansClustering(spHits, outClusters, outCluster2SP);
	mncudaSafeCallNoSync(s_tKMeans.Stop());

	// Convert the clusters to a shading point list. This is more useful for later processing as
	// it contains all relevant data (triangle indices, u_hit, v_hit, pixel index, ...).
	outSPClusters.SetFrom(spHits, outClusters.d_idxShadingPt, outClusters.numClusters);

	// Also move data into cluster list for reclassification (next frame).
	mncudaSafeCallNoSync(cudaMemcpy(outClusters.d_positions, outSPClusters.d_ptInter,
		outClusters.numClusters*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(outClusters.d_normals, outSPClusters.d_normalsG,
		outClusters.numClusters*sizeof(float4), cudaMemcpyDeviceToDevice));
}

void RTCore::PerformAdaptiveSeeding(const ShadingPoints& spHits, uint numSamplesToSelect, 
									ClusterList& outClusters, float4* d_outScreenBuffer)
{
	CameraModel* pCamera = m_pSC->GetCamera();
	uint sizeScreen = pCamera->GetScreenWidth();

	// Construct quad tree for shading points.
	QuadTreeSP qtreeSP;
	qtreeSP.Initialize(sizeScreen);
	//MNMessage("Quadtree: %d nodes for screen size %d x %d.", qtreeSP.numNodes, 
	//	pCamera->GetScreenWidth(), pCamera->GetScreenHeight());

	// Pair list to store quad tree node to shading point association.
	PairList qt2sp;
	qt2sp.Initialize(qtreeSP.numLevels*spHits.numPoints);

	// Construct association list and sort it by node idx (first elem). Store sort addresses
	// to use them later to sort geometric variations.
	KernelAFGConstructQT2SPAssoc(spHits, qtreeSP.numLevels, qt2sp);

	// Sort level by level...
	MNCudaMemory<uint> d_sortAddr(qt2sp.numPairs);
	qt2sp.SortByFirst(qtreeSP.numNodes, spHits.numPoints, d_sortAddr);

	// Compute average position and normal for each quad tree node using segmented reduction.
	// To avoid adding storage for position and normal for pair lists, I use the set from
	// address utilitiy function to move the data into a temporary array and apply segmented
	// reduction on this array to get the result for the quad tree.
	MNCudaMemory<float4> d_toReduce4(qt2sp.numPairs, "Temporary", 256);
	// Positions
	mncudaSetFromAddress((float4*)d_toReduce4, qt2sp.d_second, spHits.d_ptInter, qt2sp.numPairs);
	mncudaSegmentedReduce((float4*)d_toReduce4, qt2sp.d_first, qt2sp.numPairs, MNCuda_ADD, make_float4(0.f, 0.f, 0.f, 0.f), 
		qtreeSP.d_positions, qtreeSP.numNodes);
	// Normals
	mncudaSetFromAddress((float4*)d_toReduce4, qt2sp.d_second, spHits.d_normalsG, qt2sp.numPairs);
	mncudaSegmentedReduce((float4*)d_toReduce4, qt2sp.d_first, qt2sp.numPairs, MNCuda_ADD, make_float4(0.f, 0.f, 0.f, 0.f), 
		qtreeSP.d_normals, qtreeSP.numNodes);

	// Divide by number of points per node to get average. 
	// WARNING: Note that there might be qt nodes without any shading point as there might be intersection-less 
	//          areas. Furthermore the given shading points might not always represent the full screen!
	MNCudaMemory<float> d_qtNodeCounts(qtreeSP.numNodes);
	{
		// We don't know the point per node counts, so calculate them using segmented reduction.
		// NOTE: I use float as type here to allow later division.
		MNCudaMemory<float> d_ones(qt2sp.numPairs);
		mncudaInitConstant<float>(d_ones, qt2sp.numPairs, 1.f);
		mncudaSegmentedReduce<float>(d_ones, qt2sp.d_first, qt2sp.numPairs, MNCuda_ADD, 0.f, 
			d_qtNodeCounts, qtreeSP.numNodes);

		//mncudaPrintArray((uint*)qt2sp.d_first, qt2sp.numPairs, false, "TEST");
		//mncudaPrintArray((float*)d_qtNodeCounts, qtreeSP.numNodes, true, "TEST");

		float m;
		mncudaReduce(m, (float*)d_qtNodeCounts, qtreeSP.numNodes, MNCuda_MIN, MN_INFINITY);
		//if(m == 0.f)
		//	MNWarning("QT node without shading points!");

		// Now divide. Empty node positions stay unchanged.
		mncudaAverageArray<float4, float>(qtreeSP.d_positions, qtreeSP.numNodes, d_qtNodeCounts);
		// Normal averaging requires normalization.
		mncudaNormalize(qtreeSP.d_normals, qtreeSP.numNodes);
	}

	// Construct geometric variations per sample point and per tree level. Avoid using a new
	// pair list, so just use an array.
	MNCudaMemory<float> d_temp(qt2sp.numPairs), d_geomVariation(qt2sp.numPairs);
	KernelAFGComputeGeometricVariation(spHits, qtreeSP, d_temp);

	//mncudaPrintArray((float*)d_temp, qt2sp.numPairs, true, "TEST");

	// Sort to move into correct order.
	mncudaSetFromAddress<float>(d_geomVariation, d_sortAddr, d_temp, qt2sp.numPairs);

	// Compute geometric variation per node by *adding up* using segmented reduction.
	mncudaSegmentedReduce<float>(d_geomVariation, qt2sp.d_first, qt2sp.numPairs, MNCuda_ADD, 0.f, 
			qtreeSP.d_geoVars, qtreeSP.numNodes);

	KernelAFGNormalizeGeoVariation(qtreeSP, m_pSettings->GetGeoVarPropagation());

	// Generate random numbers using MT for sample distribution.
	MNCudaMT& mtw = MNCudaMT::GetInstance();
	uint maxRnd = mtw.GetAlignedCount(qtreeSP.numNodes - sizeScreen*sizeScreen);
	MNCudaMemory<float> d_randoms(maxRnd);
	mtw.Seed(rand());
	mncudaSafeCallNoSync(mtw.Generate(d_randoms, maxRnd));

	// Distribute seeding samples according to geometric variation, using top-down approach. We need
	// two num number arrays since we cannot synchronize the reads and writes to those arrays within
	// all threads. With one array, a thread block might write before another thread block can read.
	MNCudaMemory<uint> d_numSamples1(sizeScreen*sizeScreen);
	MNCudaMemory<uint> d_numSamples2(sizeScreen*sizeScreen);
	uint* d_numSrc = d_numSamples1;
	uint* d_numDest = d_numSamples2;
	uint initialSamples = numSamplesToSelect;
	mncudaSafeCallNoSync(cudaMemcpy(d_numSrc, &initialSamples, sizeof(uint), cudaMemcpyHostToDevice));
	uint idxLvlStart = 0, idxStartLeafs, numLeafs;
	uint* d_numLeaf = NULL;
	for(uint lvl=0; lvl<qtreeSP.numLevels-1; lvl++)
	{
		uint numNodesLvl = (1 << lvl)*(1 << lvl); // 4^lvl
		uint idxChildStart = idxLvlStart + numNodesLvl;
		uint numNodesChild = numNodesLvl << 2;

		//mncudaPrintArray(qtreeSP.d_geoVars+idxChildStart, 4, true, "TEST");

		// Distribute lvl's samples relative to normalized geometric variation to child nodes.
		KernelAFGDistributeSamplesToLevel(qtreeSP.d_geoVars, d_qtNodeCounts, d_randoms+idxLvlStart, 
			lvl, idxLvlStart, numNodesLvl,
			d_numSrc, d_numDest);

		/*if(lvl == 2)
		{
			mncudaPrintArray(qtreeSP.d_geoVars+13, 4, true, "TEST");
			mncudaPrintArray(d_numDest, 4, false, "TEST");
		}*/

		// Swap source and destination.
		if(d_numSrc == d_numSamples1)
		{
			d_numSrc = d_numSamples2; 
			d_numDest = d_numSamples1;
		}
		else
		{
			d_numSrc = d_numSamples1; 
			d_numDest = d_numSamples2;
		}

		if(lvl == qtreeSP.numLevels - 2)
		{
			d_numLeaf = d_numSrc;
			idxStartLeafs = idxChildStart;
			numLeafs = numNodesChild;
		}
		idxLvlStart = idxChildStart;
	}

	// Visualize initial sample distribution if required.
	if(m_pSettings->GetViewMode() == MNRTView_InitialSamples && d_outScreenBuffer)
		KernelAFGVisualizeInitialDistribution(spHits, qt2sp.d_second+qt2sp.numPairs-numLeafs,
			d_numLeaf, numLeafs, d_outScreenBuffer);


	// Now d_numSamples contains the number of samples for each leaf quad tree node, that is for
	// each screen pixel. Build cluster list.
	outClusters.Clear();

	// Scan per-leaf counts to generate offsets.
	MNCudaMemory<uint> d_clusterOffsets(numLeafs);
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	cp.Scan(d_numLeaf, numLeafs, false, d_clusterOffsets);

	//mncudaPrintArray<uint>(d_numLeaf, numLeafs, false, "NUM LEAFS");
	//mncudaPrintArray<float>(qtreeSP.d_geoVars+idxStartLeafs, 200, true, "GEO VAR LEAFS");

	// Fill cluster list.
	KernelAFGCreateInitialClusterList(qtreeSP, idxStartLeafs, numLeafs,
								   d_numLeaf, d_clusterOffsets, outClusters);

	// Get cluster count.
	uint lastIdx, lastNum;
	mncudaSafeCallNoSync(cudaMemcpy(&lastIdx, d_clusterOffsets+(numLeafs-1), sizeof(uint), cudaMemcpyDeviceToHost));
	mncudaSafeCallNoSync(cudaMemcpy(&lastNum, d_numLeaf+(numLeafs-1), sizeof(uint), cudaMemcpyDeviceToHost));
	outClusters.numClusters = lastIdx + lastNum;

	// Destroy temp stuff.
	qt2sp.Destroy();
	qtreeSP.Destroy();
}

void RTCore::ClassifyShadingPoints(const ShadingPoints& spHits, const ClusterList& clusters, 
								   PairList& outCluster2SP, float* d_outGeoVarsSorted)
{
	MNAssert(clusters.numClusters > 0);
	MNCudaMemory<float> d_geoVarsUnsorted(spHits.numPoints);
	MNCudaMemory<uint> d_sortAddr(spHits.numPoints);

	static StatTimer& s_tKD = StatTimer::Create("Timers", "k-Means Clustering (kd-tree building)", false);
	static StatTimer& s_tCluster = StatTimer::Create("Timers", "k-Means Clustering (Cluster search)", false);

	// kd-tree for cluster centers.
	mncudaSafeCallNoSync(s_tKD.Start(true));
	KDTreePoint kdtreeClusters(clusters.d_positions, clusters.numClusters,
		m_Tris.aabbMin, m_Tris.aabbMax, m_pSC->GetRadiusWangKMeansMax());
	kdtreeClusters.SetKNNTargetCount(5);
	kdtreeClusters.BuildTree();
	mncudaSafeCallNoSync(s_tKD.Stop(true));

	// Update kernel data first.
	AFGSetClusterData(*kdtreeClusters.GetData(), clusters.d_positions, clusters.d_normals,
		clusters.numClusters);

	// Construct association between clusters and shading points. Note that there might be
	// clusters without any shading points.
	mncudaSafeCallNoSync(s_tCluster.Start(true));
	outCluster2SP.Clear();
	KernelAFGConstructCluster2SPAssoc(spHits, clusters, m_pSC->GetRadiusWangKMeansMax(), 
		outCluster2SP, d_geoVarsUnsorted);
	mncudaSafeCallNoSync(s_tCluster.Stop(true));

	// Sort by cluster index. We also need to sort the geometric variations, if required.
	// Note that we have one additional, virtual cluster, that is created for unclassified
	// shading points.
	outCluster2SP.SortByFirst(clusters.numClusters + 1, 0, d_sortAddr);
	if(d_outGeoVarsSorted != NULL)
		mncudaSetFromAddress<float>(d_outGeoVarsSorted, d_sortAddr, d_geoVarsUnsorted, spHits.numPoints);

	AFGCleanupClusterData();
	kdtreeClusters.Destroy();
}

void RTCore::PerformKMeansClustering(const ShadingPoints& spHits, ClusterList& ioClusters,
		PairList& outCluster2SP)
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	MNCudaMemory<float> d_geoVars(spHits.numPoints);
	MNCudaMemory<float> d_geoVarsMinima(ioClusters.numClusters+1, "Temporary", pool.GetTextureAlignment());

	// Run two more iterations to finalize cluster construction:
	uint maxIters = m_pSettings->GetKMeansItersMax();
	// maxIters+1: Build cluster2sp and compact, but no k-means.
	// maxIters+2: Build cluster2sp only.
	// TODO: Improve this...
	uint iteration = 1;
	while(iteration <= maxIters+2)
	{
		// Classify shading points into clusters.
		ClassifyShadingPoints(spHits, ioClusters, outCluster2SP, d_geoVars);
		uint numClustersP1 = ioClusters.numClusters + 1; // + virtual cluster.

		if(iteration <= maxIters)
		{
			// Compute average position and normal for each cluster using segmented reduction.
			MNCudaMemory<float4> d_toReduce4(outCluster2SP.numPairs, "Temporary", 256);
			// Positions
			mncudaSetFromAddress((float4*)d_toReduce4, outCluster2SP.d_second, 
				spHits.d_ptInter, outCluster2SP.numPairs);
			mncudaSegmentedReduce((float4*)d_toReduce4, outCluster2SP.d_first, outCluster2SP.numPairs, 
				MNCuda_ADD, make_float4(0.f), ioClusters.d_positions, numClustersP1);
			// Normals
			mncudaSetFromAddress((float4*)d_toReduce4, outCluster2SP.d_second, 
				spHits.d_normalsS, outCluster2SP.numPairs);
			mncudaSegmentedReduce((float4*)d_toReduce4, outCluster2SP.d_first, outCluster2SP.numPairs, 
				MNCuda_ADD, make_float4(0.f), ioClusters.d_normals, numClustersP1);
		}

		//uint countOld, countNew;
		if(iteration <= maxIters+1)
		{
			// We don't know the point per node counts, so calculate them using segmented reduction.
			// NOTE: I use float as type here to allow later division.
			MNCudaMemory<float> d_ones(outCluster2SP.numPairs), d_counts(numClustersP1);
			mncudaInitConstant<float>(d_ones, outCluster2SP.numPairs, 1.f);
			mncudaSegmentedReduce<float>(d_ones, outCluster2SP.d_first, outCluster2SP.numPairs, MNCuda_ADD, 0.f, 
				d_counts, numClustersP1);

			// No need to include virtual cluster in next steps.

			if(iteration <= maxIters)
			{
				// Divide by number of points per node to get average.
				mncudaAverageArray<float4, float>(ioClusters.d_positions, ioClusters.numClusters, d_counts);
				// Normal averaging requires normalization
				mncudaNormalize(ioClusters.d_normals, ioClusters.numClusters);
			}

			// A consequence of the averaging process is that there might be cluster centers without any
			// points. Remove these cluster centers when we are in the n+1 iteration.
			if(iteration == maxIters+1)
			{
				MNCudaMemory<uint> d_isValid(ioClusters.numClusters);
				KernelAFGMarkNonEmptyClusters(d_counts, ioClusters.numClusters, d_isValid);
				uint countOld = ioClusters.numClusters;
				uint countNew = CompactClusters(ioClusters, d_isValid);
				//MNMessage("Removed clusters (iter %d): old %d new %d.", iteration, countOld, countNew);
			}
		}

		iteration++;

		// Avoid last run for cluster-sp-assoc generation in case the compact didn't remove anything.
		/*if(iteration == maxIters+1 && countOld == countNew)
			break;*/
	}

	// In theory, as we removed empty clusters, now every cluster should contain at least one shading
	// point. However, I detected cases where there are still clusters that do not get their shading
	// points assigned below. To avoid resulting problems with illegal shading point indices, I
	// for now initialize the array to contain something valid.
	// TODO: Eliminate this problem. Maybe it's related to failing segmented reduction.
	mncudaSafeCallNoSync(cudaMemset(ioClusters.d_idxShadingPt, 0, ioClusters.numClusters*sizeof(uint)));

	// Now we choose one irradiance sample within each cluster of shading points. To do this, we pick
	// the shading point with the minimal geometric variation (error) with regard to the cluster center.
	// The geometric variations are stored in d_geoVars, so we can get per-cluster minima by segmented
	// reduction.
	// DEBUG This shows if we have empty clusters. Then there would be an invalid shading point index!
	//mncudaInitConstant((float*)d_geoVars, outCluster2SP.numPairs, 1.f);
	mncudaSegmentedReduce<float>(d_geoVars, outCluster2SP.d_first, outCluster2SP.numPairs, MNCuda_MIN, MN_INFINITY, 
		d_geoVarsMinima, ioClusters.numClusters + 1); // + virtual cluster
	//mncudaPrintArray((float*)d_geoVarsMinima, ioClusters.numClusters+1, true, "GEOVAR MINIMA");
	// Get minima indices and store them in the cluster list.
	KernelAFGGetFinalClusterIndices(d_geoVars, d_geoVarsMinima, ioClusters.numClusters,
		outCluster2SP, ioClusters.d_idxShadingPt);

	//mncudaPrintArray((uint*)ioClusters.d_idxShadingPt, ioClusters.numClusters, false, "SP INDICES");

	// Store maximum geometric variation within each cluster center.
	mncudaSegmentedReduce<float>(d_geoVars, outCluster2SP.d_first, outCluster2SP.numPairs, 
		MNCuda_MAX, -MN_INFINITY, ioClusters.d_geoVarMax, ioClusters.numClusters + 1); // + virtual cluster
}