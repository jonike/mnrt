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
/// \file		GPU\illumcut.cu
///
/// \brief		CUDA kernels for illumination cut handling.
///
///				Illumination cuts are used to improve the performance of irradiance computation
///				for photon mapping. They were proposed by \ref lit_wang "[Wang et al. 2009]" and target to reduce the
///				data set. Without them, the full photon map with all its photons is used to perform
///				a density estimation. There are normally several 100,000 photons in a photon map.
///				
///				An illumination cut is basically a cut through the kd-tree that represents the
///				photon map. A \e cut through a tree is a set of nodes, so that there is exactly
///				one cut node on each path from a leaf to the root. Walter et al. used cuts to
///				generate clusters of many light sources. In our case we use a cut to create clusters
///				of the photon map's photons. A cut node then represents all photons that are
///				contained within its AABB. This is possible due to the cut property: There can
///				be no other node that can represent the photons of this cut node.
///
///				This file contains most of the kernels required to construct such an illumination 
///				cut. Some missing kernels are in photon_knn.cu, specifically the kernels to perform
///				kNN searches to compute "exact" radiances in tree node centers. Moving these
///				kernels out of this file was motivated by both coherency and CUDA restrictions.
///
/// \author		Mathias Neumann
/// \date		23.07.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"
#include "kd-tree/KDKernelDefs.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"

/// \brief	Irradiance estimation photon limit. Only nodes with less photons are evaluated.
/// \todo	Because irradiance estimation is very expensive for large photon map nodes, I introduced
///			this upper limit. It avoids estimation for these very large nodes. In most cases such an
///			estimation would result to useless irradiance values anyways. The actual value for this
///			limit still needs improvement.
#define ICUT_PHOTONLIMIT	5000


/// \brief	Photon position for each photon (texture memory).
///
///			To unburden the global memory cache, I move photon data to texture memory.
///			At least on my GTX 460 this was a bit faster than global memory reads only.
texture<float4, 1, cudaReadModeElementType> tex_pmapPos;
/// \brief	Photon flux (power) for each photon (texture memory).
///
///			To unburden the global memory cache, I move photon data to texture memory.
///			At least on my GTX 460 this was a bit faster than global memory reads only.
texture<float4, 1, cudaReadModeElementType> tex_pmapFlux;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_WriteNodeSideLength(KDFinalNodeList lstFinal,
/// 	float4* d_ioNodeNormals)
///
/// \brief	Enriches normal array with maximum node side lengths.
/// 		
/// 		For each node i the maximum side length of the AABB of node i is written into the
/// 		w-component \a d_ioNodeNormals[i].w. All other components are left unchanged. 
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param	lstFinal				The unordered final node list from kd-tree construction. 
/// \param [in,out]	d_ioNodeNormals	Node normal array. Contains estimated normals at node centers,
/// 								not necessarily for all node centers. This kernel enriches
/// 								the array by moving the maximum node AABB side length into
/// 								the w-coordinate of the normal entry. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_WriteNodeSideLength(KDFinalNodeList lstFinal, float4* d_ioNodeNormals)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		float3 aabbMin = make_float3(lstFinal.d_aabbMin[idxNode]);
		float3 aabbMax = make_float3(lstFinal.d_aabbMax[idxNode]);
		float maxSideLength = fmaxf(aabbMax.x - aabbMin.x, fmaxf(aabbMax.y - aabbMin.y, aabbMax.z - aabbMin.z));

		// Pack maximum side length into w-component.
		float4 old = d_ioNodeNormals[idxNode];
		old.w = maxSideLength;
		d_ioNodeNormals[idxNode] = old;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_EstimateNodeIrradiance(KDFinalNodeList lstFinal,
/// 	float4* d_nodeNormals, float4* d_outNodeIrrEstimate)
///
/// \brief	Estimates irradiance in the center of all qualified nodes of the given final
///			node list. 
///
///			A node is regarded as qualified if the following condition is \c true:
///
///			\code numPhotons < ::ICUT_PHOTONLIMIT && numPhotons > 0 && dot(nodeNormal, nodeNormal) > 0.f \endcode
///
///			where \c numPhotons is the number of photons of enclosed by the node and \c nodeNormal
///			is the estimated normal in the node center.  For unqualified nodes, the corresponding
///			component of \a d_outNodeIrrEstimate is set to zero.
///
///			The actual estimation process considers all photons enclosed in a given node and avoids
///			performing an accurate kNN query. Only photons that arrived from the upper hemisphere
///			(relative to the node normal) are included, however.
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param	lstFinal						The unordered final node list from kd-tree construction.
/// \param [in]		d_nodeNormals			Node normal array. Contains estimated normal (xyz) and
///											maximum node side length (w) for each node.
/// \param [out]	d_outNodeIrrEstimate	Returns estimated irradiance for each qualified node.
///											Only xyz components are used (RGB).
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_EstimateNodeIrradiance(KDFinalNodeList lstFinal, float4* d_nodeNormals,
											  float4* d_outNodeIrrEstimate)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		float4 n4 = d_nodeNormals[idxNode];
		float3 nodeNormal = make_float3(n4);
		float maxSideLength = n4.w;

		uint idxFirst = lstFinal.d_idxFirstElem[idxNode];
		uint numPhotons = lstFinal.d_numElems[idxNode];
		float3 result = make_float3(0.f, 0.f, 0.f);

		// Ignore evaluation for nodes with more photons than a given threshold or without any photons
		// or with zero normal.
		bool evaluate = true;
		evaluate &= numPhotons < ICUT_PHOTONLIMIT && numPhotons > 0 && dot(nodeNormal, nodeNormal) > 0.f;

		if(evaluate)
		{
			for(uint i=0; i<numPhotons; i++)
			{
				uint idxPhoton = lstFinal.d_elemNodeAssoc[idxFirst+i];

				// Read photon data using texture memory to unburden global memory cache.
				float4 pos4 = tex1Dfetch(tex_pmapPos, idxPhoton);
				float3 pos = make_float3(pos4);
				float azimuthal = pos4.w;

				float4 flux4 = tex1Dfetch(tex_pmapFlux, idxPhoton);
				float3 flux = make_float3(flux4);
				float polar = flux4.w;

				float3 inDir = dev_Spherical2Direction(azimuthal, polar);
				float NdotIn = dot(nodeNormal, -inDir);
				bool isFront = NdotIn > 0.f;

				float alpha = 1.f;
				if(!isFront)
					alpha = 0.f;
				result += alpha * flux;
			}

			// Divide by squared maximum side length (area).
			result /= (maxSideLength*maxSideLength);
		}

		// Write irradiance estimate.
		d_outNodeIrrEstimate[idxNode] = make_float4(result);
		// Zero normal to avoid further computation for empty nodes.
		if(numPhotons == 0)
			d_nodeNormals[idxNode] = make_float4(0.f, 0.f, 0.f, maxSideLength);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MaskIrrOnLevel(uint* d_nodeLevel, uint numNodes,
/// 	float4* d_nodeIrrEst, uint levelToMask, float4* d_outMaskedIrrEst,
/// 	uint* d_outIsUnmasked)
///
/// \brief	Masks all irradiances of nodes that do not belong to given node level. 
///
///			This kernel enables fast average irradiance calculation for a given node level using 
///			parallel reduction. All irradiance values of nodes on other levels are set to zero.
///			All irradiance values on level are left unchanged.
///
///			In addition to masking the radiance values, a binary 0/1 array \a d_outIsUnmasked is
///			constructed. It signalizes which nodes are preserved (1) and masked out (0).
///
/// \todo	Right now I set \a d_outIsUnmasked[i] = 0 for nodes with irradiances of zero, even if
///			they are on the requested node level. I got slightly better results doing so. This,
///			however, wasn't mentioned by \ref lit_wang "[Wang et al. 2009]".
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param [in]		d_nodeLevel			Contains the zero-based node level for each node.
/// \param	numNodes					Number of nodes. 
/// \param [in]		d_nodeIrrEst		Irradiance estimate for each node. This is not changed
///										to retain the actual values.
/// \param	levelToMask					The level to preserve. All other levels are masked. 
/// \param [out]	d_outMaskedIrrEst	Masked irradiance estimates. All values for nodes not at
///										level \a levelToMask are zeroed.
/// \param [out]	d_outIsUnmasked		Contains whether a node is masked (0) or not (1).
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MaskIrrOnLevel(uint* d_nodeLevel, uint numNodes, float4* d_nodeIrrEst,
								      uint levelToMask,
								      float4* d_outMaskedIrrEst, uint* d_outIsUnmasked)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < numNodes)
	{
		uint nodeLevel = d_nodeLevel[idxNode];
		
		uint onLevel = 1;
		float4 irr = d_nodeIrrEst[idxNode];
		if(nodeLevel != levelToMask || (irr.x == 0.f && irr.y == 0.f && irr.z == 0.f))
		{
			irr = make_float4(0.f);
			onLevel = 0;
		}
		d_outMaskedIrrEst[idxNode] = irr;
		d_outIsUnmasked[idxNode] = onLevel;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SplitIrradiance(float4* d_maskedIrrEst, uint numNodes,
/// 	float* d_outIrrR, float* d_outIrrG, float* d_outIrrB)
///
/// \brief	Splits radiance estimate into R, G and B arrays to enable reduction. 
///
///			This is required because \c float4 reduction won't work due to problems using \c float4 
///			and \c volatile together (see ::dev_ReduceFast()).
///
/// \author	Mathias Neumann
/// \date	14.08.2010
///
/// \param [in]		d_maskedIrrEst	Masked irradiance estimate for each node.
/// \param	numNodes				Number of nodes. 
/// \param [out]	d_outIrrR		All red (R) components of the estimates.
/// \param [out]	d_outIrrG		All green (G) components of the estimates.
/// \param [out]	d_outIrrB		All blue (B) components of the estimates.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SplitIrradiance(float4* d_maskedIrrEst, uint numNodes,
									   float* d_outIrrR, float* d_outIrrG, float* d_outIrrB)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < numNodes)
	{
		float4 irr = d_maskedIrrEst[idxNode];
		d_outIrrR[idxNode] = irr.x;
		d_outIrrG[idxNode] = irr.y;
		d_outIrrB[idxNode] = irr.z;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MarkNodesByEmin(float4* d_nodeIrrEst, uint numNodes, float3 Emin,
/// 	uint* d_outMarks)
///
/// \brief	Marks all nodes with irradiances larger than given minimum irradiance.
///
///			Currently I only mark a node when \em all irradiance components (R, G and B) are
///			at least as large as the corresponding component of \a Emin.
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param [in]		d_nodeIrrEst	Irradiance estimate for each node.
/// \param	numNodes				Number of nodes. 
/// \param	Emin					The minimum radiance estimate a node has to have to be
/// 								marked. 
/// \param [out]	d_outMarks		Node marks as binary 0/1 array. Contains 1 for marked nodes.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MarkNodesByEmin(float4* d_nodeIrrEst, uint numNodes, float3 Emin,
									   uint* d_outMarks)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < numNodes)
	{
		float3 Enode = make_float3(d_nodeIrrEst[idxNode]);

		// Mark a node when all irradiance components are at least as large as those of E_min.
		uint mark = 0;
		if(Enode.x >= Emin.x && Enode.y >= Emin.y && Enode.z >= Emin.z)
			mark = 1;

		d_outMarks[idxNode] = mark;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_RemoveParents(KDFinalNodeList lstFinal, uint* d_inIsMarkedOld,
/// 	uint* d_outIsMarkedNew)
///
/// \brief	Removes marks from parents that have both children marked. 
///
///			Additionally I remove marks from parents whose children are both unmarked. Note that
///			the kernel cannot be called inplace due to synchronization problems.
///
/// \author	Mathias Neumann
/// \date	24.07.2010
///
/// \param	lstFinal					The unordered final node list from kd-tree construction.
/// \param [in]		d_inIsMarkedOld		Old node marks. Binary 0/1 array.
/// \param [out]	d_outIsMarkedNew	New node marks. Binary 0/1 array.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_RemoveParents(KDFinalNodeList lstFinal, uint* d_inIsMarkedOld, 
									 uint* d_outIsMarkedNew)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		// Get child node information.
		uint left = lstFinal.d_childLeft[idxNode];
		uint right = lstFinal.d_childRight[idxNode];
		
		if(left != right) // Internal node?
		{
			uint markedL = d_inIsMarkedOld[left];
			uint markedR = d_inIsMarkedOld[right];

			// Force unmark of this node if both children marked.
			// TEST: Additionally I unmark nodes for which both children are unmarked.
			if(markedL == 1 && markedR == 1 || (markedL == 0 && markedR == 0))
				d_outIsMarkedNew[idxNode] = 0;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_UpdateCoveredState(KDFinalNodeList lstFinal, uint level,
/// 	uint* d_ioIsMarked, uint* d_ioIsCovered)
///
/// \brief	Updates the covered state for a given tree level and marks uncovered leafs on that
///			level.
///
///			Should be called top-down, beginning with the root level. Distinguished between
///			inner nodes an leafs:
///			\li Leafs: In case they are uncovered, a mark is added to ensure a correct cut.
///			\li Inner Nodes: When the node is covered, both children's cover states are set to
///				covered. Furthermore their marks are removed to ensure the cut property. Else,
///				when the node is uncovered, both children's cover states are set to their
///				marked state.
///
/// \note	I use another approach than \ref lit_wang "[Wang et al. 2009]" because I believe that their approach cannot
///			work. Or at least it seems to make assumptions on the underlying irradiance estimates
///			that do not hold for my estimates.
///
/// \author	Mathias Neumann
/// \date	24.07.2010
///
/// \param	lstFinal				The unordered final node list from kd-tree construction.
/// \param	level					The level to work on. 
/// \param [in,out]	d_ioIsMarked	Current node marks for the tree cut. Binary 0/1 array. 
///									Note that write conflicts aren't possible because we are
///									working on a single tree level only.
/// \param [in,out]	d_ioIsCovered	Keeps track of which nodes of the current level are covered.
///									Binary 0/1 array.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_UpdateCoveredState(KDFinalNodeList lstFinal, uint level, uint* d_ioIsMarked, 
										  uint* d_ioIsCovered)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		// Get child node information.
		uint left = lstFinal.d_childLeft[idxNode];
		uint right = lstFinal.d_childRight[idxNode];
		uint myLevel = lstFinal.d_nodeLevel[idxNode];
		uint myMark = d_ioIsMarked[idxNode];
		uint myCovered = d_ioIsCovered[idxNode];

		if(myLevel == level) // Node on correct level?
		{
			if(left == right) // Leaf?
			{
				// For leaf nodes we need to force a mark in case they are uncovered.
				if(myCovered == 0)
					myMark = 1;
			}
			else
			{
				// Consider the children of the current node. Propagate cover state.
				uint markL = d_ioIsMarked[left];
				uint markR = d_ioIsMarked[right];

				uint covL, covR;
				if(myCovered == 1)
				{
					// If parent covered, children covered anytime. Also be sure to remove marks.
					markL = 0;
					markR = 0;
					covL = 1;
					covR = 1;
				}
				else
				{
					// If parent uncovered, covering of children depends on their marks.
					covL = markL;
					covR = markR;
				}

				// Update children.
				d_ioIsMarked[left] = markL;
				d_ioIsMarked[right] = markR;
				d_ioIsCovered[left] = covL;
				d_ioIsCovered[right] = covR;
			}

			// Update parent node. Covering cannot change for non-leafs.
			d_ioIsMarked[idxNode] = myMark;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MarkLeafs(KDFinalNodeList lstFinal, uint* d_outIsMarked)
///
/// \brief	Marks all leafs of the given tree.
///
///			Simplified, worst possible algorithm to choose a cut. The leafs represent a trivial
///			cut through the tree. They are, however quite numerous. Hence performance benefit
///			of using such trivial cut instead of all photons of the photon map tree is not that
///			great.
///
/// \note	Was implemented for testing purposes.
///
/// \author	Mathias Neumann
/// \date	August 2010
///
/// \param	lstFinal				The unordered final node list from kd-tree construction.
/// \param [out]	d_outIsMarked	Node marks as binary 0/1 array. Only leafs will be marked.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MarkLeafs(KDFinalNodeList lstFinal, uint* d_outIsMarked)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		// Get child node information.
		uint left = lstFinal.d_childLeft[idxNode];
		uint right = lstFinal.d_childRight[idxNode];

		uint mark = 0;
		if(left == right)
			mark = 1;
		d_outIsMarked[idxNode] = mark;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_UpdateCut(KDTreeData pmap, float4* d_nodeIrrEst,
/// 	float4* d_nodeIrrExact, uint* d_workNodeIdx, uint numWorkNodes, float reqAccuracy,
/// 	uint* d_ioWorkNew, uint* d_ioFinalCutMarks)
///
/// \brief	Updates the current cut to improve accuracy.
/// 		
/// 		For each work node, the accuracy of it's irradiance estimate is checked by comparing
/// 		it to an "exact" irradiance value. The latter was computed using kNN search. Inner
/// 		work nodes with insufficient accuracy are replaced with their children. Else they are
/// 		added to the final cut. Leafs are always added to the final cut. 
/// \todo	Determine best accuracy criterion. 
///
/// \author	Mathias Neumann
/// \date	24.07.2010
///
/// \param	pmap						The photon map kd-tree data. Note that at this point we
/// 									no longer have access to the more complex
/// 									KDFinalNodeList. 
/// \param [in]		d_nodeIrrEst		Irradiance estimate for each node. 
/// \param [in]		d_nodeIrrExact		"Exact" irradiance for each node. Relevant are only the
/// 									values for the work nodes. This avoids computing exact
/// 									values for all nodes. 
/// \param [in]		d_workNodeIdx		Node index for each work node. That is, \a
/// 									d_workNodeIdx[i] contains the node index of the i-th work
/// 									node. 
/// \param	numWorkNodes				Number of work nodes. 
/// \param	reqAccuracy					The required accuracy for inner nodes to be inserted into
/// 									the final cut. See MNRTSettings::SetICutAccuracy(). The
/// 									comparison of estimated and exact irradiance values is
/// 									done for each component. 
/// \param [in,out]	d_ioWorkNew			Will contain the new work node marks. Binary 0/1 array.
/// 									This array should be initialized with the old work node
/// 									marks. They will be updated by this kernel. The work
/// 									marks are an array of as many elements as there are nodes
/// 									in the kd-tree. 
/// \param [in,out]	d_ioFinalCutMarks	The updated final cut node marks. Binary 0/1 array. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_UpdateCut(KDTreeData pmap, float4* d_nodeIrrEst, float4* d_nodeIrrExact,
								 uint* d_workNodeIdx, uint numWorkNodes, 
								 float reqAccuracy,
								 uint* d_ioWorkNew, uint* d_ioFinalCutMarks)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numWorkNodes)
	{
		// Get real node index.
		uint idxNode = d_workNodeIdx[tid];

		float3 radEstimate = make_float3(d_nodeIrrEst[idxNode]);
		float3 radExact = make_float3(d_nodeIrrExact[idxNode]);

		bool isOK = true;

		// Accuracy fine?
		float3 diff = radEstimate - radExact;
		isOK &= fabsf(diff.x) < reqAccuracy*radExact.x;
		isOK &= fabsf(diff.y) < reqAccuracy*radExact.y;
		isOK &= fabsf(diff.z) < reqAccuracy*radExact.z;

		// For leaf nodes, just take node anyways.
		uint left = pmap.d_childLeft[idxNode];
		uint right = pmap.d_childRight[idxNode];
		isOK |= (left == right);

		// Remove from work ...
		d_ioWorkNew[idxNode] = 0;
		if(isOK)
		{
			// ... and add to final cut.
			d_ioFinalCutMarks[idxNode] = 1;
		}
		else
		{
			// ... and add children to work.
			d_ioWorkNew[left] = 1;
			d_ioWorkNew[right] = 1;
		}
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_WriteNodeSideLength() kernel call.
extern "C"
void KernelICutWriteNodeSideLength(const KDFinalNodeList& lstFinal, float4* d_ioNodeNormals)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);

	kernel_WriteNodeSideLength<<<gridSize, blockSize>>>(lstFinal, d_ioNodeNormals);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_EstimateNodeIrradiance() kernel call.
extern "C"
void KernelICutEstimateNodeIrradiance(const KDFinalNodeList& lstFinal, float4* d_nodeNormals,
									  const PhotonData& photons,
							          float4* d_outNodeIrrEstimate)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);

	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	// Bind photon positions.
	tex_pmapPos.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmapPos, photons.d_positions, 
		cdFloat4, photons.numPhotons*sizeof(float4)));
	// Bind photon flux.
	tex_pmapFlux.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_pmapFlux, photons.d_powers, 
		cdFloat4, photons.numPhotons*sizeof(float4)));

	kernel_EstimateNodeIrradiance<<<gridSize, blockSize>>>(lstFinal, d_nodeNormals,
		d_outNodeIrrEstimate);
	MNCUDA_CHECKERROR;

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmapPos));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_pmapFlux));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" float3 KernelICutGetAverageIrradianceOnLevel(uint* d_nodeLevel,
/// 	float4* d_nodeIrrEst, uint numNodes, uint level)
///
/// \brief	Computes the average irradiance on a given node level.
/// 		
/// 		Uses kernel_MaskIrrOnLevel() and kernel_SplitIrradiance() to get masked components of
/// 		the estimated irradiances. Then reductions are performed to compute the average for
/// 		each component. 
///
/// \author	Mathias Neumann
/// \date	23.07.2010 
///	\see	::mncudaReduce()
///
/// \param [in]		d_nodeLevel		Zero-based node level for each node. 
/// \param [in]		d_nodeIrrEst	Irradiance estimate for each node. 
/// \param	numNodes				Number of nodes. 
/// \param	level					The level. 
///
/// \return	Average irradiane on given level. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
float3 KernelICutGetAverageIrradianceOnLevel(uint* d_nodeLevel, float4* d_nodeIrrEst, 
											 uint numNodes, uint level)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numNodes, blockSize.x), 1, 1);

	// First mask all other node radiances.
	MNCudaMemory<float4> d_maskedIrr(numNodes, "Temporary", 256);
	MNCudaMemory<uint> d_isUnmasked(numNodes);
	kernel_MaskIrrOnLevel<<<gridSize, blockSize>>>(d_nodeLevel, 
		numNodes, d_nodeIrrEst, level, d_maskedIrr, d_isUnmasked);
	MNCUDA_CHECKERROR;

	// Split irradiance to get components usable for reduction.
	MNCudaMemory<float> d_irrR(numNodes, "Temporary");
	MNCudaMemory<float> d_irrG(numNodes, "Temporary");
	MNCudaMemory<float> d_irrB(numNodes, "Temporary");
	kernel_SplitIrradiance<<<gridSize, blockSize>>>(d_maskedIrr, numNodes,
			d_irrR, d_irrG, d_irrB);
	MNCUDA_CHECKERROR;

	// Now use reduction to compute sum of all irradiances.
	float3 irrSum;
	mncudaReduce(irrSum.x, (float*)d_irrR, numNodes, MNCuda_ADD, 0.f);
	mncudaReduce(irrSum.y, (float*)d_irrG, numNodes, MNCuda_ADD, 0.f);
	mncudaReduce(irrSum.z, (float*)d_irrB, numNodes, MNCuda_ADD, 0.f);

	uint count;
	mncudaReduce(count, (uint*)d_isUnmasked, numNodes, MNCuda_ADD, (uint)0);

	//printf("Masked count: %d.\n", count);
	//printf("Rad sum: %.8f, %.8f, %.8f.\n", irrSum.x, irrSum.y, irrSum.z);

	return irrSum / float(count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelICutMarkCoarseCut(const KDFinalNodeList& lstFinal,
/// 	float4* d_nodeIrrEst, float3 Emin, uint* d_outMarks)
///
/// \brief	Computes a coarse illumination cut.
///
///			Calls kernel_MarkNodesByEmin() to mark initial nodes. As they do not represent a cut
///			in most cases, first kernel_RemoveParents() is called to eliminate some inner nodes
///			from the set of marked nodes. To ensure the cut property, hereafter the kernel
///			kernel_UpdateCoveredState() is used top-down to propagate a covered state through
///			the tree. A node is covered when there is at least one marked node on the path from
///			the node to the root. All uncovered leafs are marked to ensure the property.
///
/// \todo	The algorithm is quite suboptimal right now. For example, just adding leaf marks
///			to create a cut might not be the best solution. When there is a large subtree with
///			an uncovered root and without any marks, all leafs of that subtree will get marked.
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param	lstFinal				The unordered final node list from kd-tree construction.
/// \param [in]		d_nodeIrrEst	Irradiance estimate for each node.
/// \param	Emin					Minimum irradiance for nodes to be included in the initial
///									node set.
/// \param [out]	d_outMarks		Node marks for chosen coarse cut. Binary 0/1 array.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelICutMarkCoarseCut(const KDFinalNodeList& lstFinal, float4* d_nodeIrrEst, float3 Emin,
						     uint* d_outMarks)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);

	kernel_MarkNodesByEmin<<<gridSize, blockSize>>>(d_nodeIrrEst, lstFinal.numNodes, Emin,
		d_outMarks);
	MNCUDA_CHECKERROR;

	/*uint count;
	mncudaReduce(count, (uint*)d_outMarks, lstFinal.numNodes, MNCuda_ADD, (uint)0);
	printf("Emin nodes: %d.\n", count);*/

	// Now eliminate all nodes for which a child is marked.
	MNCudaMemory<uint> d_tempMarks(lstFinal.numNodes);
	mncudaSafeCallNoSync(cudaMemcpy(d_tempMarks, d_outMarks, lstFinal.numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	kernel_RemoveParents<<<gridSize, blockSize>>>(lstFinal, d_tempMarks, d_outMarks);
	MNCUDA_CHECKERROR;

	/*mncudaReduce(count, (uint*)d_outMarks, lstFinal.numNodes, MNCuda_ADD, (uint)0);
	printf("Non-parent nodes: %d.\n", count);*/

	// Get maximum tree level for next step.
	uint maxLevel;
	mncudaReduce(maxLevel, lstFinal.d_nodeLevel, lstFinal.numNodes, MNCuda_MAX, (uint)0);
	MNCUDA_CHECKERROR;

	// Cover all leaf to root paths with exactly one node.
	uint* d_isCovered = d_tempMarks;
	mncudaSafeCallNoSync(cudaMemset(d_isCovered, 0, lstFinal.numNodes*sizeof(uint)));
	for(uint lvl=0; lvl<=maxLevel; lvl++)
	{
		kernel_UpdateCoveredState<<<gridSize, blockSize>>>(lstFinal, lvl, d_outMarks, d_isCovered);
		MNCUDA_CHECKERROR;

		/*mncudaReduce(count, (uint*)d_outMarks, lstFinal.numNodes, MNCuda_ADD, (uint)0);
		printf("PASS nodes: %d.\n", count);*/
	}
}

/// Wraps kernel_MarkLeafs() kernel call.
extern "C"
void KernelICutMarkLeafs(const KDFinalNodeList& lstFinal, uint* d_outMarks)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);
	kernel_MarkLeafs<<<gridSize, blockSize>>>(lstFinal, d_outMarks);
}

/// Wraps kernel_UpdateCut() kernel call.
extern "C"
void KernelICutUpdateFinalCut(const KDTreeData& pmap, float4* d_nodeIrrEst, float4* d_nodeIrrExact,
							  uint* d_workNodeIdx, uint numWorkNodes, float reqAccuracy,
							  uint* d_ioWorkMarks, uint* d_ioFinalCutMarks)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numWorkNodes, blockSize.x), 1, 1);

	kernel_UpdateCut<<<gridSize, blockSize>>>(pmap, d_nodeIrrEst, d_nodeIrrExact,
		d_workNodeIdx, numWorkNodes, reqAccuracy, d_ioWorkMarks, d_ioFinalCutMarks);
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////