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
/// \file	MNRT\PhotonMap.h
///
/// \brief	Declares the PhotonMap class. 
/// \author	Mathias Neumann
/// \date	07.07.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_PHOTONMAP_H__
#define __MN_PHOTONMAP_H__

#pragma once

#include <vector_types.h>
#include "KernelDefs.h"
#include "kd-tree/KDTreeListener.h"

// Forward declarations
class RayChunk;
class KDTreePoint;
class MNRTSettings;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	PhotonMap
///
/// \brief	Representation for photon maps.
///
///			Hides point kd-tree and photon data. Provides methods for photon map construction and
///			illumination cuts construction. For the latter, the class implements the KDTreeListener.
///
/// \author	Mathias Neumann
/// \date	07.07.2010
/// \see	KDTreePoint, PhotonData
////////////////////////////////////////////////////////////////////////////////////////////////////
class PhotonMap : public KDTreeListener
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	PhotonMap(uint maxPhotons, float3 sceneAABBMin, float3 sceneAABBMax,
	/// 	float globalQRadiusMax, MNRTSettings* pSettings)
	///
	/// \brief	Initializes the photon map object.
	///
	///			To build the concrete kd-tree, you first have to add photon data using Merge().
	///			Then call BuildMap().
	///
	/// \author	Mathias Neumann
	/// \date	07.07.2010
	///
	/// \param	maxPhotons			The maximum number of photons to store. 
	/// \param	sceneAABBMin		Scene bounding box minimum
	/// \param	sceneAABBMax		Scene bounding box maximum. 
	/// \param	globalQRadiusMax	Global query radius maximum.
	/// \param [in]	pSettings		MNRT application settings object.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	PhotonMap(uint maxPhotons, float3 sceneAABBMin, float3 sceneAABBMax, float globalQRadiusMax,
		MNRTSettings* pSettings);
	/// Destructs this object. Calls Destroy().
	~PhotonMap(void);

// Attributes
private:
	// Photon data.
	PhotonData m_Photons;
	// KD-tree for photon map.
	KDTreePoint* m_pPhotonMap;

	// Scene AABB minimum/maximum.
	float3 m_sceneAABBMin;
	float3 m_sceneAABBMax;

	// Settings
	MNRTSettings* m_pSettings;
	// KNN target count.
	uint m_knnTargetCount;
	// Global query radius maximum.
	float m_fGlobalQRadiusMax;

	// Approximated normals at node centers.
	// w: Contains maximum side length of node's bounding box.
	float4* d_nodeCenterNormals;
	// Approximated irradiance values E~_p, one for each node. Computed at node center.
	float4* d_nodeIrrEstimate;
	// Exact irradiance values E_p, only valid for cut nodes. Computed at node center.
	float4* d_nodeIrrExact;
	// Coarse tree cut node markers. Stored as they are required later when kd-tree build done.
	uint* d_isCoarseTreeCutNode;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool BuildMap()
	///
	/// \brief	Builds photon map kd-tree from current photon data.
	/// 		
	/// 		Previous kd-tree is destroyed using Destroy(), if any. This method requires that you
	/// 		added photons using Merge(). Besides building the kd-tree it also handles
	/// 		illumination cut construction, if required. 
	///
	/// \author	Mathias Neumann
	/// \date	July 2010
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool BuildMap();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint Merge(const PhotonData& other, uint* d_isValid)
	///
	/// \brief	Merges given photon data into photon data of this object. 
	///
	///			The provided binary array \a d_isValid is used to control which photons to merge.
	///
	/// \author	Mathias Neumann
	/// \date	July 2010
	/// \see	PhotonData::Merge()
	///
	/// \param	other				The other photon data. 
	/// \param [in]	d_isValid		Binary 0/1 array with as many entries as there are photons in
	///								\a other. Only photons with \a d_isValid[i] = 1 are merged
	///								into this photon map.
	///
	/// \return	The number of merged photons. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint Merge(const PhotonData& other, uint* d_isValid);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ScaleFlux(float scale)
	///
	/// \brief	Scale flux of all photons.
	///
	/// \author	Mathias Neumann
	/// \date	April 2010
	///
	/// \param	scale	Constant scale factor. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ScaleFlux(float scale);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Gather(const ShadingPoints& sp, const RayChunk& rayChunk, float4* d_clrDiffHit,
	/// 	float4* d_ioRadiance)
	///
	/// \brief	Gathers nearby photons an performs density estimation.
	///
	///			This operation updates the incoming radiance at each shading point by estimating the
	///			radiance using this photon map. This is done using direct density estimation without
	///			sending out gather ray to gather the radiance in the environment. Currently this
	///			operation does not use illumination cuts to retain high quality for caustics. That's
	///			because this operation is used for caustics only.
	///
	///			It is assumed that all data was compacted before so that only valid hits, i.e. valid
	///			shading points are given.
	///
	/// \author	Mathias Neumann
	/// \date	June 2010
	///
	/// \param	sp						The shading points at which to perform the density estimation.
	/// \param	rayChunk				The rays that were traced to the shading points \a sp.
	/// \param [in]		d_clrDiffHit	Diffuse hit colors for each shading point.
	/// \param [in,out]	d_ioRadiance	Incoming radiance accumulator at each shading point.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Gather(const ShadingPoints& sp, const RayChunk& rayChunk,
				float4* d_clrDiffHit, float4* d_ioRadiance);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void GatherFor(const ShadingPoints& sp, const RayChunk& rayChunk, float4* d_clrDiffHit,
	/// 	uint* d_idxShadingPts, float4* d_ioRadiance, float* d_ioReciDistAccu)
	///
	/// \brief	Performs final gather step by performing density estimation at gather ray hit points.
	///
	///			Opposed to Gather() this method does an indirect density estimation at the hits of
	///			a set of gather rays sent out from the actual shading points for which we seek the
	///			incoming, indirect radiance. To identify which accumulator index to use for an
	///			indirect density estimation, the \a d_idxShadingPts array stores the index of the
	///			original shading point.
	///
	///			It is assumed that all data was compacted before so that only valid hits, i.e. valid
	///			shading points are given.
	///
	/// \author	Mathias Neumann
	/// \date	June 2010
	///
	/// \param	sp							The shading points at which to perform density estimation. 
	///										Note that this are the gather ray hit points.
	/// \param	rayChunk					Corresponding gather rays.
	/// \param [in]		d_clrDiffHit		Diffuse hit colors at each shading point.
	/// \param [in]		d_idxShadingPts		Contains for each shading point the original shading point
	///										index for that we perform indirect density estimation. Hence
	///										\a d_idxShadingPts[i] gives the index for both \a d_ioRadiance
	///										and \a d_ioReciDistAccu.
	/// \param [in,out]	d_ioRadiance		Incoming radiance accumulator for original shading point
	///										for which we perform indirect density estimation.
	/// \param [in,out]	d_ioReciDistAccu	Reciprocal distance accumulator for original shading points.
	///										Used to find harmonic mean distances.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void GatherFor(const ShadingPoints& sp, const RayChunk& rayChunk,
				   float4* d_clrDiffHit, 
				   uint* d_idxShadingPts,
				   float4* d_ioRadiance, float* d_ioReciDistAccu);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Visualize(const ShadingPoints& sp, const RayChunk& rayChunk, float4* d_ioRadiance)
	///
	/// \brief	Visualizes this photon map.
	///
	///			Visualization is done using a very small query radius, so that the photons are
	///			represented as small discs.
	///
	/// \author	Mathias Neumann
	/// \date	June 2010
	///
	/// \param	sp						The shading points at which to perform the visualization.
	/// \param	rayChunk				The rays that were traced to the shading points \a sp.
	/// \param [in,out]	d_ioRadiance	Incoming radiance accumulator for each shading point.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Visualize(const ShadingPoints& sp, const RayChunk& rayChunk,
				   float4* d_ioRadiance);

// Accessors
public:
	/// Returns number of photons currently stored in this map.
	uint GetPhotonCount() const { return m_Photons.numPhotons; }
	/// Sets the k for kNN searches to use for this photon map.
	void SetKNNTargetCount(uint targetCount) { m_knnTargetCount = targetCount; }

// Implementation
public:
	void OnFinalNodeList(KDFinalNodeList* pNodesFinal);

private:
	void Destroy();
	void DestroyTree();

	// Estimates minimum estimated radiance for coarse tree cut nodes (E_min).
	float3 GetEminForCoarseCut(KDFinalNodeList* pNodesFinal);
	// Computes exact radiance at each work node.
	void ComputExactNodeRadiances(uint* d_workNodeIdx, uint numWorkNodes);
	// Generates final cut list.
	void GenerateFinalIllumCut();
};


#endif // __MN_PHOTONMAP_H__