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
/// \file	GPU\photon_knn.cu
///
/// \brief	Kernels for photon map density estimation using kNN search. 
/// \author	Mathias Neumann
/// \date	07.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"
#include "kd-tree/KDKernelDefs.h"
#include "RayPool.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"

// Constants for Gaussian filter for Caustics density estimation (Jensen, p. 82).

/// Gaussian filter constant alpha. See \ref lit_jensen "[Jensen 2001]", p. 82.
#define KNN_GAUSS_ALPHA			1.818f
/// Gaussian filter constant beta. See \ref lit_jensen "[Jensen 2001]", p. 82.
#define KNN_GAUSS_BETA			1.953f
/// \brief	Gaussian filter inverse denominator.
///
///			\code 1.f / (1.f - expf(-KNN_GAUSS_BETA)) \endcode
#define KNN_GAUSS_INVDENOM		1.165294576f


/// CUDA device properties used to get maximum grid size. See "kdtree.cu".
extern cudaDeviceProp f_DevProps;

/// kNN target count constant memory variable, i.e. the k in kNN.
__constant__ uint c_knnTargetCountPM = 50;
/// Photon map kd-tree data constant memory variable.
__constant__ KDTreeData c_PMap;


/// Photon map kd-tree texture for KDTreeData::d_preorderTree.
texture<uint, 1, cudaReadModeElementType> tex_pmap;
/// Photon map kd-tree node extents texture.
texture<float4, 1, cudaReadModeElementType> tex_pmapNodeExtent;
/// Photon data texture for PhotonData::d_positions.
texture<float4, 1, cudaReadModeElementType> tex_pmapPos;
/// Photon data texture for PhotonData::d_powers.
texture<float4, 1, cudaReadModeElementType> tex_pmapFlux;
/// Illumination cut interpolation data texture for estimated node center normals.
texture<float4, 1, cudaReadModeElementType> tex_icutNodeNormal;
/// Illumination cut interpolation data texture for "exact" irradiance in node centers.
texture<float4, 1, cudaReadModeElementType> tex_icutNodeIrrExact;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_RangeSearch(float3 ptQuery, float queryRadiusSqr, float3 nQuery,
/// 	uint& outNumFound)
///
/// \brief	Range search for density estimation (irradiance).
/// 		
/// 		Underlying data is the current photon map (see ::PMSetPhotonMap()). As irradiance is
/// 		computed, this function does not capture the general case of arbitrary BSDFs. It just
/// 		assumes a diffuse surface at the query point, so that the corresponding BRDF can be
/// 		separated from the computation. 
///
/// \author	Mathias Neumann
/// \date	08.04.2010
///
/// \todo	Implement a way to use the Gaussian filter for caustics only. Currently I just don't
///			use it at all.
///
/// \param	ptQuery				The query point. 
/// \param	queryRadiusSqr		The query radius (squared). 
/// \param	nQuery				Normal vector at query point. 
/// \param [out]	outNumFound	Will contain the number of used photons for estimation. 
///
/// \return	Estimated irradiance at query point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_RangeSearch(float3 ptQuery, float queryRadiusSqr, float3 nQuery, 
								  uint& outNumFound)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree depth first to look for photons q with ||p - q|| < r.
	int addrNode = 0;

	float3 result = make_float3(0.f, 0.f, 0.f);
	float maxDistSqr = 0.f;
	outNumFound = 0;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_PMap.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			// Texture fetching probably results in a lot of serialization due to cache misses.
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_PMap.d_preorderTree[addrNode+1], c_PMap.d_preorderTree[addrNode+2]);
			uint addrRight = parentInfo.x & 0x0FFFFFFF;
			uint splitAxis = parentInfo.x >> 30;
			float splitPos = *(float*)&parentInfo.y;

			// Compute squared distance on split axis from query point to splitting plane.
			float distSqr = (p[splitAxis] - splitPos) * (p[splitAxis] - splitPos);

			// Advance to next child node, possibly enqueue other child.
			uint addrOther;
			addrNode = addrLeft;
			addrOther = addrRight;
			if(p[splitAxis] > splitPos)
			{
				// Next: right node.
				addrNode = addrRight;
				addrOther = addrLeft;
			}

			// Enqueue other if required.
			if(distSqr < queryRadiusSqr)
			{
				// Enqueue second child in todo list.
				todoAddr[todoPos++] = addrOther;
			}

			// Read node index + leaf info (MSB) for new node.
			idxNode = c_PMap.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		// Now we have a leaf.
		uint numPhotons = c_PMap.d_preorderTree[addrNode+1];;

		for(uint i=0; i<numPhotons; i++)
		{
			// Get photon index.
			uint idxPhoton = c_PMap.d_preorderTree[addrNode+2+i];

			float4 pos4 = tex1Dfetch(tex_pmapPos, idxPhoton);
			float3 pos = make_float3(pos4);
			float azimuthal = pos4.w;

			float distSqr = dev_DistanceSquared(ptQuery, pos);
			// Ensure that the vector to the photon lies at least 60° away from the normal,
			// that is 0.5 = cos 60° > cos beta.
			// NOTE: I could not see much benefit from this technique in MNCaustics scene.
			//bool isWithinCompressed = fabsf(dot(normalize(pos - ptQuery), nQuery)) < 0.3f;
			if(distSqr < queryRadiusSqr/* && isWithinCompressed*/)
			{
				// Photon in range.
				float4 flux4 = tex1Dfetch(tex_pmapFlux, idxPhoton);
				float3 flux = make_float3(flux4);
				float polar = flux4.w;

				float3 inDir = dev_Spherical2Direction(azimuthal, polar);
				bool isFront = dot(nQuery, -inDir) > 0.f;

				// Gaussian filter, see Jensen p. 82.
				//float weight = KNN_GAUSS_ALPHA * 
				//	(1.f - (1.f - expf(-KNN_GAUSS_BETA*distSqr/queryRadiusSqr*0.5f))*KNN_GAUSS_INVDENOM);

				// WARNING: kd-tree node centers (illumination cut node centers) might lie within an
				//          object.
				float alpha = 1.f;
				if(!isFront)
					alpha = 0.f;
				result += alpha * flux;

				maxDistSqr = fmaxf(maxDistSqr, distSqr);
				outNumFound++;
			}
		}

		addrNode = -1;
		if(todoPos != 0)
		{
			// Pop next node from stack.
			todoPos--;
			addrNode = todoAddr[todoPos];
		}
	}

	float3 irr = make_float3(0.f, 0.f, 0.f);

	// Don't use maxDistSqr in case we couldn't find enough photons.
	if(outNumFound < c_knnTargetCountPM)
		maxDistSqr = queryRadiusSqr;

	// Scale the result by 1 / dA.
	if(maxDistSqr > 0.f)
		irr = result / (maxDistSqr * MN_PI);

	return irr;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_EstimateICUTRadius(float3 ptQuery)
///
/// \brief	Very simple illumination cut interpulation query radius estimation.
///
/// \author	Mathias Neumann
/// \date	October 2010
///
/// \param	ptQuery	The query point. 
///
/// \return	Radius of the leaf that contains \a ptQuery as estimated query radius. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_EstimateICUTRadius(float3 ptQuery)
{
	const float* p = (float*)&ptQuery;
	
	// Read node index + leaf info (MSB).
	int addrNode = 0; // root
	uint idxNode = c_PMap.d_preorderTree[addrNode];
	uint isLeaf = idxNode & 0x80000000;
	idxNode &= 0x7FFFFFFF;

	// Find the ICUT node for ptQuery.
	while(!isLeaf)
	{
		uint2 parentInfo = make_uint2(c_PMap.d_preorderTree[addrNode+1], c_PMap.d_preorderTree[addrNode+2]);
		uint isICutNode = parentInfo.x & 0x10000000; // Get bit 28
		if(isICutNode)
			break;

		uint addrLeft = addrNode + 1 + 2;
		uint addrRight = parentInfo.x & 0x0FFFFFFF;
		uint splitAxis = parentInfo.x >> 30;
		float splitPos = *(float*)&parentInfo.y;

		// Advance to next child node.
		if(p[splitAxis] > splitPos)
			addrNode = addrRight;
		else
			addrNode = addrLeft;

		// Read node index + leaf info (MSB) for new node.
		idxNode = c_PMap.d_preorderTree[addrNode];
		isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;
	}

	float4 e4 = tex1Dfetch(tex_pmapNodeExtent, idxNode);
	float radius = e4.w;

	return radius;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float dev_SpatialRBF(float r)
///
/// \brief	Spatial radial basis function (RBF) for illumination cut interpolation.
/// 		
/// 		The generated function vanishes for \a r > \c 1.f. Else the value \c 1.f - \a r * \a
/// 		r is returned. 
///
/// \author	Mathias Neumann
/// \date	24.07.2010
///
/// \param	r	The argument. 
///
/// \return	Function value at \a r. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_SpatialRBF(float r)
{
	if(r <= 1.f)
		return 1.f - r*r;
	else
		return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_InterpolateNodeIrradiance(float3 ptQuery, float queryRadiusSqr,
/// 	float3 nQuery)
///
/// \brief	Interpolates irradiance values stored at illumination cut nodes.
///
///			A weighted interpolation is performed, as \ref lit_wang "[Wang et al. 2009]" proposed. For the weights,
///			dev_SpatialRBF() is used to restrict the influence of illumination cut nodes according
///			to the corresponding node extent.
///
/// \warning This function is somewhat experimental, mainly due to the problems with illumination
///			 cut selection and node center normal estimation.
///
/// \author	Mathias Neumann
/// \date	24.07.2010
/// \see	dev_RangeSearch()
///
/// \param	ptQuery			The query point. 
/// \param	queryRadiusSqr	The query radius (squared). 
/// \param	nQuery			Normal vector at query point. 
///
/// \return	Estimated irradiance at query point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_InterpolateNodeIrradiance(float3 ptQuery, float queryRadiusSqr, float3 nQuery)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree depth first to look for illumination cut nodes within range.
	int addrNode = 0;

	float3 irrResult = make_float3(0.f, 0.f, 0.f);
	float weightSum = 0.f;
	//float minDist = MN_INFINITY;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_PMap.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next illumination cut node. This is not neccessary a leaf!
		while(!isLeaf)
		{
			uint2 parentInfo = make_uint2(c_PMap.d_preorderTree[addrNode+1], c_PMap.d_preorderTree[addrNode+2]);
			uint isICutNode = parentInfo.x & 0x10000000; // Get bit 28
			if(isICutNode)
				break;

			uint addrLeft = addrNode + 1 + 2;
			uint addrRight = parentInfo.x & 0x0FFFFFFF;
			uint splitAxis = parentInfo.x >> 30;
			float splitPos = *(float*)&parentInfo.y;

			// Compute squared distance on split axis from query point to splitting plane.
			float distSqr = (p[splitAxis] - splitPos) * (p[splitAxis] - splitPos);

			// Advance to next child node, possibly enqueue other child.
			uint addrOther;
			addrNode = addrLeft;
			addrOther = addrRight;
			if(p[splitAxis] > splitPos)
			{
				// Next: right node.
				addrNode = addrRight;
				addrOther = addrLeft;
			}

			// Enqueue other if required.
			if(distSqr < queryRadiusSqr)
			{
				// Enqueue second child in todo list.
				todoAddr[todoPos++] = addrOther;
			}

			// Read node index + leaf info (MSB) for new node.
			idxNode = c_PMap.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		// Now we have an illumination cut node. Add it's contribution to the result.
		// NOTE: Since pNodeC represents a *kd-node*-center, it might be somewhat offsetted.
		//       Therefore we cannot expect to have the same influence radius for each cut node,
		//       even if we fixed the side length.
		float4 n4 = tex1Dfetch(tex_icutNodeNormal, idxNode);
		float3 nNodeC = make_float3(n4);
		float maxSideLength = n4.w;

		// Avoid zero normal nodes (e.g. empty nodes).
		if(dot(nNodeC, nNodeC) != 0.f)
		{
			float3 pNodeC = make_float3(tex1Dfetch(tex_pmapNodeExtent, idxNode));
			float dist = length(pNodeC - ptQuery);
			float weight = dev_SpatialRBF(dist/maxSideLength) * fmaxf(0.f, dot(nNodeC, nQuery));

			float3 irrNode = make_float3(tex1Dfetch(tex_icutNodeIrrExact, idxNode));
			//if(dist < maxSideLength && weight > 0)
			//	irrResult = make_float3(weight);
			irrResult += weight * irrNode;
			weightSum += weight;
		}

		addrNode = -1;
		if(todoPos != 0)
		{
			// Pop next node from stack.
			todoPos--;
			addrNode = todoAddr[todoPos];
		}
	}

	if(weightSum != 0.f)
		irrResult /= weightSum;
	return irrResult;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_PhotonGather(ShadingPoints shadingPts, RayChunk srcRays,
/// 	float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint, float4* d_ioRadiance,
/// 	float* d_ioInvDistAccu)
///
/// \brief	Gathers photons at given shading points to perform radiance estimation.
/// 		
/// 		The device function dev_RangeSearch() is called for density estimation using the
/// 		current photon map. Should only be called for diffuse surfaces because
///			diffuse BRDFs are assumed.
///
/// \author	Mathias Neumann
/// \date	11.04.2010
/// \see	kernel_InterpolateIrradiance()
///
/// \param	shadingPts					The shading points for which we should perform density
/// 									estimation. For final gathering, they would be the hit
/// 									points of the gather rays, hence this would be an
/// 									indirect density estimation. 
/// \param	srcRays						Source ray chunk. These are the rays that led to the
/// 									shading point \a shadingPts. For direct density
/// 									estimation, it's just the a of primary/secondary rays.
/// 									For indirect density estimation, this would be a set of
/// 									gather rays. In either case, \a the i-th ray corresponds
/// 									to the i-th shading point. 
/// \param [in]		d_queryRadius		Query radius for each shading point. Can be estimated
/// 									using KDTreePoint::ComputeQueryRadii(). 
/// \param [in]		d_clrDiffuse		Diffuse material color at each shading point. Used for
/// 									diffuse BRDF evaluation. \c xyz contains color and \c w
/// 									transparency alpha. Pass \c NULL for this parameter to
///										signalize photon visualization.
/// \param [in]		d_idxShadingPoint	Used for final gathering, i.e. indirect density
/// 									estimation. For each shading point of \a shadingPts, it
/// 									contains the index of the actual shading point, from
/// 									which the gather ray was sent. Used to address the output
/// 									buffers for radiance and reciprocal distances. 
/// \param [in,out]	d_ioRadiance		Radiance accumulator for "direct" shading points. In case
/// 									of final gahtering, this would have to be addressed using
/// 									\a d_idxShadingPoint. 
/// \param [in,out]	d_ioInvDistAccu		Reciprocal distance accumulator for "direct" shading
/// 									points. In case of final gahtering, this would have to be
/// 									addressed using \a d_idxShadingPoint. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_PhotonGather(ShadingPoints shadingPts, RayChunk srcRays,
									float* d_queryRadius,
									float4* d_clrDiffuse, uint* d_idxShadingPoint,
									float4* d_ioRadiance,
									float* d_ioInvDistAccu)
{
	uint tid = blockDim.x * MNCUDA_GRID2DINDEX + threadIdx.x;

	// We compacted...
	if(tid < shadingPts.numPoints)
	{
		// Source ray direction.
		float3 inDir = make_float3(srcRays.d_dirs[tid]);
		// Get shading normal at gather sample.
		float3 nGatherSample = make_float3(shadingPts.d_normalsS[tid]);

		// Swap normal if source ray hit on other side.
		if(dot(-inDir, nGatherSample) < 0.f)
			nGatherSample *= -1.f;

		float3 ptSample = make_float3(shadingPts.d_ptInter[tid]);
	
		// Get query radius maximum for this sample.
		float queryRadius = d_queryRadius[tid];

		uint numFound;
		float3 irradiance = dev_RangeSearch(ptSample, queryRadius*queryRadius, nGatherSample, numFound);

		// I currently ignore gathers with less than 8 photons. The same thing is done by
		// Jensen2001 p. 160. However, in photon visualization mode, i.e. when d_clrDiffuse is
		// NULL, this is not employed.
		if(d_clrDiffuse && numFound < 8)
			irradiance = make_float3(0.f);

		// Get source shading point index.
		uint idxSourceSP;
		if(d_idxShadingPoint)
			idxSourceSP = d_idxShadingPoint[tid];
		else
			idxSourceSP = tid;

		// Compute the reciprocal distance sum if required.
		if(d_ioInvDistAccu)
		{
			float3 rayO = make_float3(srcRays.d_origins[tid]);
			float3 o2hit = ptSample - rayO;
			float reciDist = rsqrtf(dot(o2hit, o2hit));
			d_ioInvDistAccu[idxSourceSP] += reciDist;
		}

		// Get diffuse color of surface at the sample we gather at.
		float4 c4 = make_float4(1.f, 1.f, 1.f, 1.f);
		if(d_clrDiffuse)
			c4 = d_clrDiffuse[tid];
		float3 clrDiffuse = make_float3(c4.x, c4.y, c4.z);
		float transAlpha = c4.w;

		// Get radiance by multiplication with BSDF (diffuse). 
		// NOTE: The ray influence contains the PI / numGatherSamples term for final gathering.
		//		 In that case, the result would be an irradiance value.
		//		 See PBR p. 762 for details.
		float3 infl = make_float3(srcRays.d_influences[tid]);
		float3 rad = transAlpha * clrDiffuse * MN_INV_PI * irradiance * infl;
		d_ioRadiance[idxSourceSP] += make_float4(rad);
	}
}



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_InterpolateIrradiance( ShadingPoints shadingPts,
/// 	RayChunk srcRays, float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint,
/// 	float4* d_ioRadiance, float* d_ioInvDistAccu)
///
/// \brief	Perform radiance estimation by illumination cut node interpolation.
/// 		
/// 		The device function dev_InterpolateNodeIrradiance() is called for interpolation.
/// 		Should only be called for diffuse surfaces because diffuse BRDFs are assumed.
///			
///
/// \author	Mathias Neumann
/// \date	25.07.2010
/// \see	kernel_PhotonGather()
///
/// \param	shadingPts					The shading points for which we should perform density
/// 									estimation. For final gathering, they would be the hit
/// 									points of the gather rays, hence this would be an
/// 									indirect density estimation. 
/// \param	srcRays						Source ray chunk. These are the rays that led to the
/// 									shading point \a shadingPts. For direct density
/// 									estimation, it's just the a of primary/secondary rays.
/// 									For indirect density estimation, this would be a set of
/// 									gather rays. In either case, \a the i-th ray corresponds
/// 									to the i-th shading point. 
/// \param [in]		d_queryRadius		Query radius for each shading point. Can be estimated
/// 									using KDTreePoint::ComputeQueryRadii(). 
/// \param [in]		d_clrDiffuse		Diffuse material color at each shading point. Used for
/// 									diffuse BRDF evaluation. \c xyz contains color and \c w
/// 									transparency alpha. 
/// \param [in]		d_idxShadingPoint	Used for final gathering, i.e. indirect density
/// 									estimation. For each shading point of \a shadingPts, it
/// 									contains the index of the actual shading point, from
/// 									which the gather ray was sent. Used to address the output
/// 									buffers for radiance and reciprocal distances. 
/// \param [in,out]	d_ioRadiance		Radiance accumulator for "direct" shading points. In case
/// 									of final gahtering, this would have to be addressed using
/// 									\a d_idxShadingPoint. 
/// \param [in,out]	d_ioInvDistAccu		Reciprocal distance accumulator for "direct" shading
/// 									points. In case of final gahtering, this would have to be
/// 									addressed using \a d_idxShadingPoint. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_InterpolateIrradiance(
									ShadingPoints shadingPts, RayChunk srcRays,
									float* d_queryRadius, float4* d_clrDiffuse,
									uint* d_idxShadingPoint,
									float4* d_ioRadiance, float* d_ioInvDistAccu)
{
	uint tid = blockDim.x * MNCUDA_GRID2DINDEX + threadIdx.x;

	// We compacted...
	if(tid < shadingPts.numPoints)
	{
		// Source ray direction.
		float3 inDir = make_float3(srcRays.d_dirs[tid]);
		// Get shading normal at gather sample.
		float3 nGatherSample = make_float3(shadingPts.d_normalsS[tid]);

		// Swap normal if source ray hit on other side.
		// WARNING: DO NOT SWAP THE NORMAL HERE. IRRADIANCE INTERPOLATION SAMPLES ARE CALCULATED
		//          USING THE UNSWAPPED NORMALS!
		/*if(dot(-inDir, nGatherSample) < 0.f)
		{
			// Prints less here than in else!
			//printf("SWAPPING NORMAL.\n");
			nGatherSample *= -1.f;
		}*/

		float3 ptSample = make_float3(shadingPts.d_ptInter[tid]);
	
		// Get query radius for this sample.
		float queryRadius = d_queryRadius[tid];//dev_EstimateICUTRadius(ptSample);//

		// Get irradiance by interpolation.
		float3 irradiance = dev_InterpolateNodeIrradiance(ptSample, queryRadius*queryRadius, nGatherSample);

		// Get source shading point index.
		uint idxSourceSP = tid;
		if(d_idxShadingPoint)
			idxSourceSP = d_idxShadingPoint[tid];

		// Compute the reciprocal distance sum.
		// NOTE: Clamping required? Could not find any improvement.
		if(d_ioInvDistAccu)
		{
			float3 rayO = make_float3(srcRays.d_origins[tid]);
			float3 o2hit = ptSample - rayO;
			float reciDist = rsqrtf(dot(o2hit, o2hit));
			d_ioInvDistAccu[idxSourceSP] += reciDist;
		}

		// Get diffuse color of surface at the sample we gather at.
		float4 c4 = d_clrDiffuse[tid];
		float3 clrDiffuse = make_float3(c4.x, c4.y, c4.z);
		float transAlpha = c4.w;

		// Get radiance by multiplication with BSDF (diffuse). 
		// NOTE: The ray influence contains the PI / numGatherSamples term for final gathering.
		//		 In that case, the result would be an irradiance value.
		//		 See PBR p. 762 for details.
		float3 infl = make_float3(srcRays.d_influences[tid]);
		float3 rad = transAlpha * clrDiffuse * MN_INV_PI * irradiance * infl;
		d_ioRadiance[idxSourceSP] += make_float4(rad);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_IrradianceAtNodeCenters(float4* d_workQueryPoint,
/// 	float* d_queryRadius, uint* d_workNodeIdx, uint numWorkNodes, float4* d_nodeCenterNormals,
/// 	float4* d_ioIrrExact)
///
/// \brief	Computes irradiance at given kd-tree node centers using density estimation.
///
///			Calls dev_RangeSearch() to estimate the irradiance.
///
/// \author	Mathias Neumann
/// \date	24.07.2010
///
/// \param [in]		d_workQueryPoint	The query points. Corresponds to a compacted node center
///										array, compacted to remove nodes for which no density
///										estimation should be performed.
/// \param [in]		d_queryRadius		Query radius for each query point. Can be estimated
/// 									using KDTreePoint::ComputeQueryRadii(). 
/// \param [in]		d_workNodeIdx		Index of the actual kd-tree node for each work node.
/// \param	numWorkNodes				Number of work nodes. 
/// \param [in]		d_nodeCenterNormals	Estimated normals for each kd-tree node. Note that this
///										array is \em not compacted.
/// \param [in,out]	d_ioIrrExact		Will receive the computed irradiance values at the indices
///										given by the \a d_workNodeIdx array.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_IrradianceAtNodeCenters(float4* d_workQueryPoint, float* d_queryRadius, uint* d_workNodeIdx,
							   uint numWorkNodes, float4* d_nodeCenterNormals, float4* d_ioIrrExact)
{
	uint tid = blockDim.x * MNCUDA_GRID2DINDEX + threadIdx.x;

	if(tid < numWorkNodes)
	{
		// Get query point (node center).
		float3 pQuery = make_float3(d_workQueryPoint[tid]);
		// Get node index for this query point.
		uint idxQueryNode = d_workNodeIdx[tid];
		// Get normal at node center.
		float3 nQuery = make_float3(d_nodeCenterNormals[idxQueryNode]);
	
		// Get query radius for this sample.
		float queryRadius = d_queryRadius[tid];

		float3 result = make_float3(0.f);
		uint numFound;
		if(dot(nQuery, nQuery) > 0.f)
		{
			result = dev_RangeSearch(pQuery, queryRadius*queryRadius, nQuery, numFound);

			if(numFound < 8)
				result = make_float3(0.f);
		}

		d_ioIrrExact[idxQueryNode] = make_float4(result);
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


/// Sets cache configurations for kNN kernels.
extern "C"
void PMInitializeKernelData()
{
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_PhotonGather, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_InterpolateIrradiance, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_IrradianceAtNodeCenters, cudaFuncCachePreferL1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void PMSetPhotonMap(const KDTreeData& pmap, const PhotonData& photons,
/// 	uint knnTargetCount)
///
/// \brief	Binds textures for photon map kNN search.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param	pmap			kd-tree data for photon map.
/// \param	photons			Photons organized by the given kd-tree. 
/// \param	knnTargetCount	The k in kNN to use, i.e. the target photon count.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void PMSetPhotonMap(const KDTreeData& pmap, const PhotonData& photons, uint knnTargetCount)
{
	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdUint2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_knnTargetCountPM", &knnTargetCount, sizeof(uint)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_PMap", &pmap, sizeof(KDTreeData)));

	// Bind photon positions.
	tex_pmapPos.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmapPos, photons.d_positions, 
		cdFloat4, photons.numPhotons*sizeof(float4)));
	// Bind photon flux.
	tex_pmapFlux.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmapFlux, photons.d_powers, 
		cdFloat4, photons.numPhotons*sizeof(float4)));

	// Bind kd-tree stuff.
	tex_pmap.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmap, pmap.d_preorderTree, cdUint, 
		pmap.sizeTree*sizeof(uint)));
	tex_pmapNodeExtent.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmapNodeExtent, pmap.d_nodeExtent, cdFloat4, 
		pmap.numNodes*sizeof(float4)));
}

/// Unbinds textures for photon map kNN search.
extern "C"
void PMUnsetPhotonMap()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmapPos));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmapFlux));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmap));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmapNodeExtent));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void ICutSetInterpolData(float4* d_nodeCenterNormal, float4* d_nodeIrrExact,
/// 	uint numNodes)
///
/// \brief	Binds textures for illumination cut interpolation.
///
/// \author	Mathias Neumann
/// \date	July 2010
///
/// \param [in]	d_nodeCenterNormal	Estimated normal at each node center.
/// \param [in]	d_nodeIrrExact		"Exact" irradiance for node centers.
/// \param	numNodes				Number of kd-tree nodes. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void ICutSetInterpolData(float4* d_nodeCenterNormal, float4* d_nodeIrrExact, uint numNodes)
{
	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	tex_icutNodeNormal.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_icutNodeNormal, d_nodeCenterNormal, cdFloat4, numNodes*sizeof(float4)));
	tex_icutNodeIrrExact.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_icutNodeIrrExact, d_nodeIrrExact, cdFloat4, numNodes*sizeof(float4)));
}

/// Unbinds textures for illumination cut interpolation.
extern "C"
void ICutUnsetInterpolData()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_icutNodeNormal));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_icutNodeIrrExact));
}

/// Unbinds all textures. Should be called on exit of application.
extern "C"
void PMCleanupKernelData()
{
	PMUnsetPhotonMap();
	ICutUnsetInterpolData();
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_PhotonGather() kernel call.
extern "C"
void KernelPMPhotonGather(const ShadingPoints& shadingPts, const RayChunk& srcRays,
						  float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint,
						  float4* d_ioRadiance, float* d_ioInvDistAccu, bool bVisualizePhotons)
{
	dim3 blockSize = dim3(256, 1, 1);
	uint numBlocks = MNCUDA_DIVUP(shadingPts.numPoints, blockSize.x);
	dim3 gridSize = MNCUDA_MAKEGRID2D(numBlocks, f_DevProps.maxGridSize[0]);

	kernel_PhotonGather<<<gridSize, blockSize>>>( 
				shadingPts, srcRays, d_queryRadius, d_clrDiffuse, d_idxShadingPoint, d_ioRadiance, d_ioInvDistAccu);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_InterpolateIrradiance() kernel call.
extern "C"
void KernelICutGatherByInterpolation(const ShadingPoints& shadingPts, const RayChunk& srcRays,
								     float* d_queryRadius, float4* d_clrDiffuse, uint* d_idxShadingPoint, 
								     float4* d_ioRadiance, float* d_ioInvDistAccu)
{
	dim3 blockSize = dim3(256, 1, 1);
	uint numBlocks = MNCUDA_DIVUP(shadingPts.numPoints, blockSize.x);
	dim3 gridSize = MNCUDA_MAKEGRID2D(numBlocks, f_DevProps.maxGridSize[0]);

	kernel_InterpolateIrradiance<<<gridSize, blockSize>>>(shadingPts, srcRays,
		d_queryRadius, d_clrDiffuse, d_idxShadingPoint, d_ioRadiance, d_ioInvDistAccu);	
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_IrradianceAtNodeCenters() kernel call.
extern "C"
void KernelICutIrrAtNodeCenters(float4* d_workQueryPoint, float* d_queryRadius, uint* d_workNodeIdx,
							    uint numWorkNodes, float4* d_nodeCenterNormals, float4* d_ioIrrExact)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numWorkNodes, blockSize.x), 1, 1);

	kernel_IrradianceAtNodeCenters<<<gridSize, blockSize>>>(d_workQueryPoint, 
		d_queryRadius, d_workNodeIdx, numWorkNodes, d_nodeCenterNormals, d_ioIrrExact);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////