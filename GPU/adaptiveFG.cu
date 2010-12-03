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
/// \file	GPU\adaptiveFG.cu
///
/// \brief	CUDA kernels for adaptive final gathering.
/// \author	Mathias Neumann
/// \date	15.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"
#include "kd-tree/KDKernelDefs.h"

#include "mncudautil_dev.h"

/// \brief Keeps track of whether local data is initialized.
///
/// \see	::AFGSetClusterData()
bool f_dataInitialized = false;

/// \brief	Weighting factor alpha for geometric variation.
///
///			Should be scaled according to geometric extent of current scene.
/// \see	MNRTSettings::SetGeoVarAlpha()
__constant__ float c_alpha = 0.3f;
/// \brief	Cluster kd-tree constant memory variable.
///
///			Stored in constant memory for improved performance.
__constant__ KDTreeData c_kdCluster;

// Cluster kd-tree textures.

/// Cluster kd-tree texture for KDTreeData::d_preorderTree.
texture<uint, 1, cudaReadModeElementType> tex_kdCluster;
/// Cluster center positions texture.
texture<float4, 1, cudaReadModeElementType> tex_clusterP;
/// Cluster center normals texture.
texture<float4, 1, cudaReadModeElementType> tex_clusterN;

// Used for interpolation:

/// Reciprocal mean distances texture.
texture<float, 1, cudaReadModeElementType> tex_meanReciDists;
/// Cluster center irradiances texture.
texture<float4, 1, cudaReadModeElementType> tex_clusterIrr;
/// Cluster geometric variation texture.
texture<float, 1, cudaReadModeElementType> tex_clusterGeoVar;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_GeometricVariation(float alpha, float3 p_sp, float3 n_sp,
/// 	float3 p_cluster, float3 n_cluster)
///
/// \brief	Computes the geometric variation for a given point relative to another point. 
///
///			The geometric variation both considers position changes and normal changes. It is
/// 		defined as 
/// 		
/// 		\code eps = alpha * length(x_i - x_k) + sqrtf(2 - 2*(n_i dot n_k)) \endcode 
///
/// \author	Mathias Neumann
/// \date	17.04.2010
///
/// \param	alpha		Constant that determines the influence of position changes relative to
/// 					normal changes. A larger alpha leads to more influence for position
///						changes.
/// \param	p_sp		Point's position (shading point in most cases). 
/// \param	n_sp		Point's normal (shading point in most cases). 
/// \param	p_cluster	Other point's position (cluster center in most cases). 
/// \param	n_cluster	Other point's normal (cluster center in most cases). 
///
/// \return	Geometric variation (>= 0.f). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_GeometricVariation(float alpha, float3 p_sp, float3 n_sp, 
										float3 p_cluster, float3 n_cluster)
{
	// Small errors in calculation could lead to a negative argument for sqrt(), which
	// will lead to undefined results. Therefore use fmaxf().
	return alpha * length(p_sp - p_cluster) + sqrtf(fmaxf(0.f, 2.f - 2.f * dot(n_sp, n_cluster)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_GetInterpolationWeight(float3 p, float3 n, float3 p_cluster,
/// 	float3 n_cluster, float mrd_cluster)
///
/// \brief	Computes interpolation weight for sample interpolation. 
///
///			As I got problems when the point of interpolation was equal to a cluster (in this case
///			we'd get a division by zero), I just use a big weight when the corresponding
///			denominator is zero.
///
/// \author	Mathias Neumann
/// \date	22.04.2010
///
/// \param	p			Position of interpolation. 
/// \param	n			Normal at \a p. 
/// \param	p_cluster	Cluster center's position. 
/// \param	n_cluster	Cluster center's normal. 
/// \param	mrd_cluster	Mean reciprocal distance for cluster. Computed during final gathering.
/// 					Note that we avoid using the harmonic mean distance (reciprocal of mrd)
/// 					since we then would have to get its inverse. 
///
/// \return	Interpolation weight for given cluster (or sample). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_GetInterpolationWeight(float3 p, float3 n,	float3 p_cluster, float3 n_cluster,
											float mrd_cluster)
{
	float posComp = length(p - p_cluster) * mrd_cluster;
	float normComp = sqrtf(fmaxf(0.f, 2.f - 2.f * dot(n, n_cluster)));

	// Use an error of 1/100 for a cluster that matches the shading point.
	float result = 100.f;
	if(posComp + normComp != 0.f)
		result = 1.f / (posComp + normComp);
	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ int dev_RangeSearchBestFit(float3 ptQuery, float queryRadiusSqr,
/// 	float3 nQuery, float& outGeoVar)
///
/// \brief	Range search to find the best fitting cluster center.
///
///			The term "best fitting" is based on the geometric variation between cluster center and
///			query point. Due to the fact that only a small environment of the query point is
///			considered, the search might sometimes find a suboptimal solution in form of a
///			local minimum.
///
/// \todo	Find way to export range search device functions into a header file to avoid specifing 
///			nearly the same algorithm within multiple cu-files. I failed to do that due to problems
///			with the required texture references for performing the range search. As Fermi GPUs
///			have cached global memory, and as I determined that it is benefical to access the
///			kd-tree traversal buffer over this global memory, only range search operation dependend
///			data remains in texture memory. Moving them to global memory would not be useful because
///			then one cache, the texture cache, would not be employed. Maybe it is possible
///			to export the basic algorithm into a header file, as the global memory traversal data
///			could be passed as parameter. The operation dependend data could still remain in texture
///			memory, if the operation is specified in the actual cu-files. A template parameter could
///			inject the concrete operation into the basic algorithm.
///
/// \author	Mathias Neumann
/// \date	17.04.2010
///
/// \param	ptQuery				The query point. 
/// \param	queryRadiusSqr		The query radius (squared). 
/// \param	nQuery				The normal at the query point. 
/// \param [out]	outGeoVar	Will contain the geometric variation of query point to best cluster
///								center. If -1 is returned, this will be ::MN_INFINITY.
///
/// \return	The index of the best cluster center or -1, if none found. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ int dev_RangeSearchBestFit(float3 ptQuery, float queryRadiusSqr, float3 nQuery, float& outGeoVar)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	// Note that KD_MAX_HEIGHT might be somewhat overdimensioned for cluster kd-trees.
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree depth first to look for clusters q with ||p - q|| < r.
	int addrNode = 0;

	int idxNearest = -1;
	outGeoVar = MN_INFINITY;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_kdCluster.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			// Texture fetching probably results in a lot of serialization due to cache misses.
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_kdCluster.d_preorderTree[addrNode+1], c_kdCluster.d_preorderTree[addrNode+2]);
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
			idxNode = c_kdCluster.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		//__syncthreads();

		// Now we have a leaf.
		uint numClusters = c_kdCluster.d_preorderTree[addrNode+1];

		for(uint i=0; i<numClusters; i++)
		{		
			uint idxCluster = c_kdCluster.d_preorderTree[addrNode+2+i];

			float3 p_cluster = make_float3(tex1Dfetch(tex_clusterP, idxCluster));
			float3 n_cluster = make_float3(tex1Dfetch(tex_clusterN, idxCluster));

			// Evaluate error metric and test if new minimum found.
			float geoVar = dev_GeometricVariation(c_alpha, ptQuery, nQuery, p_cluster, n_cluster);
			if(geoVar < outGeoVar)
			{
				outGeoVar = geoVar;
				idxNearest = idxCluster;
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

	return idxNearest;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_RangeSearchInterpolate(float3 ptQuery, float queryRadiusSqr,
/// 	float3 nQuery, float4& outWeightedIrrSum)
///
/// \brief	Range search to find qualified samples and perform interpolation.
///
///			Qualified samples are samples in close range to the query point, i.e. the point
///			where the interpolation is performed.
///
/// \author	Mathias Neumann
/// \date	22.04.2010
///
/// \param	ptQuery						The query point, i.e. the point of interpolation. 
/// \param	queryRadiusSqr				The query radius (squared). 
/// \param	nQuery						The normal at the query point.
/// \param [out]	outWeightedIrrSum	Will contain the weighted irradiance sum that
///										has to be divided by the sum of interpolation weights.
///
/// \return	Sum of used interpolation weights. Might be zero.
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_RangeSearchInterpolate(float3 ptQuery, float queryRadiusSqr, float3 nQuery, 
											float4& outWeightedIrrSum)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	// Note that KD_MAX_HEIGHT might be somewhat overdimensioned for cluster kd-trees.
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree depth first to look for clusters q with ||p - q|| < r.
	int addrNode = 0;

	float weightSumGood = 0.f, weightSumBad = 0.f;
	float4 weightedIrrSumBad = make_float4(0.f);
	outWeightedIrrSum = make_float4(0.f);
	uint count = 0;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_kdCluster.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			// Texture fetching probably results in a lot of serialization due to cache misses.
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_kdCluster.d_preorderTree[addrNode+1], c_kdCluster.d_preorderTree[addrNode+2]);
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
			idxNode = c_kdCluster.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		//__syncthreads();

		// Now we have a leaf - here idxNode is the element count.
		uint numClusters = c_kdCluster.d_preorderTree[addrNode+1];

		for(uint i=0; i<numClusters; i++)
		{		
			uint idxCluster = c_kdCluster.d_preorderTree[addrNode+2+i];

			float3 p_cluster = make_float3(tex1Dfetch(tex_clusterP, idxCluster));
			float3 n_cluster = make_float3(tex1Dfetch(tex_clusterN, idxCluster));
			
			float mrd_cluster = tex1Dfetch(tex_meanReciDists, idxCluster);
			float4 irr_cluster = tex1Dfetch(tex_clusterIrr, idxCluster);

			float distSqr = dev_DistanceSquared(ptQuery, p_cluster);
			if(distSqr < queryRadiusSqr)
			{
				// Get interpolation weight.
				float w = dev_GetInterpolationWeight(ptQuery, nQuery,	
					p_cluster, n_cluster, mrd_cluster);

				// Only accept clusters with w > 1/a, where a is the expected error when
				// choosing the cluster. See Ward1988.
				// NOTE: Currently I avoid this and just use some standard tests. I believe the
				//       sample distribution is not well enough to use this.
				bool bSampleOK = true;//w > 4.f;
				// Normals too different?
				bSampleOK &= dot(nQuery, n_cluster) > 0.01f;
				// Cluster in front of the point being shaded?
				bSampleOK &= dot(ptQuery - p_cluster, nQuery + n_cluster) >= -.01f;
				if(bSampleOK)
				{
					outWeightedIrrSum += w * irr_cluster;
					weightSumGood += w;
				}

				// FIX:
				weightedIrrSumBad += w * irr_cluster;
				weightSumBad += w;
				count++;
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

	if(weightSumGood == 0.f && weightSumBad > 0.f)
	{
		// Use bad approximation.
		outWeightedIrrSum = weightedIrrSumBad;
		weightSumGood = weightSumBad;

		// WARNING: This can crash the application in case there are too many printfs.
		//printf("BAD APPROXIMATION: %d, %.3f %.3f %.3f (%.3f).\n", count, 
		//	weightedIrrSumBad.x, weightedIrrSumBad.y, weightedIrrSumBad.z, weightSumBad);
	}

	return weightSumGood;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ConstructQT2SPAssoc(ShadingPoints spHits, uint numQTLevels,
/// 	PairList outAssoc)
///
/// \brief	Constructs quad tree node to shading point association.
/// 		
/// 		The result is orgnaized so that sorting it after node index, i.e. after 
///			PairList::d_first is possible by sorting segments of ShadingPoints::numPoints
///			elements. Hence the pairs associated to a given quadtree level are stored
///			contiguously. The advantage is that we can avoid sorting all pairs in one run.
///
/// \author	Mathias Neumann
/// \date	16.04.2010
///
/// \param	spHits		The shading points (hits). 
/// \param	numQTLevels	Number of quadtree levels. 
/// \param	outAssoc	The association as a list of pairs. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ConstructQT2SPAssoc(ShadingPoints spHits, uint numQTLevels,
										   PairList outAssoc)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < spHits.numPoints)
	{
		// Get pixel coordinates.
		uint idxPixel = spHits.d_pixels[tid];
		uint logScreenSize = numQTLevels - 1;
		uint screenSize = 1 << logScreenSize;
		uint sp_x = idxPixel & (screenSize - 1);	// idxPixel % screenSize
		uint sp_y = idxPixel >> logScreenSize;		// idxPixel / screenSize
		
		uint idxLvlStart = 0;
		uint idxParentNode = 0;
		uint logChildNodeSize = logScreenSize;
		uint childNodeSize = screenSize;
		for(uint lvl=0; lvl<numQTLevels; lvl++)
		{
			// Get child node coordinate in current parent node (2x2 children).
			uint child_x = sp_x >> logChildNodeSize;
			uint child_y = sp_y >> logChildNodeSize;

			// Compute node index in final layout.
			uint nodeIdx = idxLvlStart + idxParentNode + 2*child_y + child_x;

			// First: Index of the l-th level parent quad tree node for shading point tid.
			outAssoc.d_first[lvl*spHits.numPoints + tid] = nodeIdx;

			// Second: Shading point index.
			outAssoc.d_second[lvl*spHits.numPoints + tid] = tid;

			// Reduce pixel coordinates to child node size.
			sp_x = sp_x & (childNodeSize - 1);
			sp_y = sp_y & (childNodeSize - 1);

			// Reduce child node size for next level.
			logChildNodeSize--;
			childNodeSize >>= 1;

			idxParentNode = 4*(idxParentNode + 2*child_y + child_x);
			idxLvlStart += (1 << lvl)*(1 << lvl); // += 4^lvl
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ComputeGeometricVariation(ShadingPoints spHits,
/// 	QuadTreeSP quadTree, float* d_outGeoVar)
///
/// \brief	Computes geometric variation for each shading point and each quadtree node.
/// 		
/// 		For each quadtree node, the node center is used to compute the geometric variation to
/// 		all contained shading points. The order of results is the same as for the
/// 		kernel_ConstructQT2SPAssoc() kernel to avoid uncoalesced access by taking advantage
/// 		of ::mncudaSetFromAddress() to reorder the geometric variations to sorted order
/// 		corresponding to the order of the PairList constructed by
/// 		kernel_ConstructQT2SPAssoc() and sorted by PairList::SortByFirst(). 
///
/// \author	Mathias Neumann
/// \date	16.04.2010 
/// \see	::dev_GeometricVariation(),
///
/// \param	spHits				The shading points (hits). 
/// \param	quadTree			The quad tree. It is assumed that both QuadTreeSP::d_positions
/// 							and QuadTreeSP::d_normals are initialized with the averaged
/// 							positions and normals respectively. 
/// \param [out]	d_outGeoVar	The geometric variations (same unsorted order as
/// 							kernel_ConstructQT2SPAssoc() produced). Contains zero for nodes
/// 							with zero normal. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ComputeGeometricVariation(ShadingPoints spHits, QuadTreeSP quadTree,
												 float* d_outGeoVar)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Use the same algorithm as in kernel_ConstructQT2SPAssoc to avoid different order.
	// I chose to generate the geometric variation in the same order as for the association
	// from tree node to shading points. We get the correct sorted order by copying the values
	// to their sorted destination.
	// The other way would be to use the d_second component of the pair list which has the
	// shading point index. However this would lead to significant uncoalesced access since
	// pair count is level-times higher than shading point count and for each pair we would
	// have to read both normal and position data.
	if(tid < spHits.numPoints)
	{
		// Get pixel coordinates.
		uint idxPixel = spHits.d_pixels[tid];
		uint logScreenSize = quadTree.numLevels - 1;
		uint screenSize = 1 << logScreenSize;
		uint sp_x = idxPixel & (screenSize - 1);	// idxPixel % screenSize
		uint sp_y = idxPixel >> logScreenSize;		// idxPixel / screenSize
		
		// Get our position and shading normal.
		float3 p_sp = make_float3(spHits.d_ptInter[tid]);
		float3 n_sp = make_float3(spHits.d_normalsG[tid]);
		
		uint idxLvlStart = 0;
		uint idxParentNode = 0;
		uint logChildNodeSize = logScreenSize;
		uint childNodeSize = screenSize;
		for(uint lvl=0; lvl<quadTree.numLevels; lvl++)
		{
			// Get child node coordinate in current parent node (2x2 children).
			uint child_x = sp_x >> logChildNodeSize;
			uint child_y = sp_y >> logChildNodeSize;

			// Compute node index in final layout.
			uint nodeIdx = idxLvlStart + idxParentNode + 2*child_y + child_x;

			// Get node average position and normal.
			float3 p_node = make_float3(quadTree.d_positions[nodeIdx]);
			float3 n_nodeTemp = make_float3(quadTree.d_normals[nodeIdx]);
			// Normalize quad tree normal. This is requried.
			float3 n_node = normalize(n_nodeTemp);

			// Compute geometric variation.
			float geomVar = dev_GeometricVariation(c_alpha, p_sp, n_sp, p_node, n_node);
			// Set geometric variation to zero in case the average normal is zero.
			if(n_nodeTemp.x == 0.f && n_nodeTemp.y == 0.f && n_nodeTemp.z == 0.f)
				geomVar = 0.f;
			d_outGeoVar[lvl*spHits.numPoints + tid] = geomVar;

			// Reduce pixel coordinates to child node size.
			sp_x = sp_x & (childNodeSize - 1);
			sp_y = sp_y & (childNodeSize - 1);

			// Reduce child node size for next level.
			logChildNodeSize--;
			childNodeSize >>= 1;

			idxParentNode = 4*(idxParentNode + 2*child_y + child_x);
			idxLvlStart += (1 << lvl)*(1 << lvl); // += 4^lvl
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_PropagateGeoVariation(float* d_ioGeoVars, uint idxParentStart,
/// 	uint numNodesParent, uint idxChildStart, uint numNodesChild, float fPropFactor)
///
/// \brief	Propagates geometric variation from child level to parent level.
///
///			For each parent node the geometric variation is updated the following way:
///
///			\code eps_node += fPropFactor * (eps_ch1 + eps_ch2 + eps_ch3 + eps_ch4) \endcode
///
///			when \c eps_node is the current geometric variation of the parent node and
///			\c eps_ch1 to \c eps_ch4 are the corresponding geometric variations of the
///			four children.
///
/// \author	Mathias Neumann
/// \date	20.08.2010
/// \see	MNRTSettings::SetGeoVarPropagation()
///
/// \param [in,out]	d_ioGeoVars	Geometric variations for each quadtree node. Stored in the same
///								order as it is used for QuadTreeSP.
/// \param	idxParentStart		The parent start node index. 
/// \param	numNodesParent		Number of parent nodes. 
/// \param	idxChildStart		The child start node index. 
/// \param	numNodesChild		Number of child nodes. 
/// \param	fPropFactor			The propagation factor. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_PropagateGeoVariation(float* d_ioGeoVars, uint idxParentStart, uint numNodesParent,
											 uint idxChildStart, uint numNodesChild, float fPropFactor)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numNodesParent)
	{
		uint idxNode = idxParentStart + tid;
		uint idxFirstChild = idxChildStart + 4*tid;

		float geoSumChildren = d_ioGeoVars[idxFirstChild] + d_ioGeoVars[idxFirstChild+1] + 
							   d_ioGeoVars[idxFirstChild+2] + d_ioGeoVars[idxFirstChild+3];

		d_ioGeoVars[idxNode] += fPropFactor*geoSumChildren;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_NormalizeGeoVariation(float* d_ioGeoVars, uint levelChildren,
/// 	uint idxChildStart, uint numNodesChild)
///
/// \brief	Normalize geometric variations for given child node level.
/// 		
/// 		This is done by reading the child node's geometric variations into shared memory and
/// 		summing up the four values child on each thread. In case the sum of geometric
/// 		variations for the child nodes is zero, I currently set the normalized geometric
/// 		variation to \c 0.25f for each of the four child nodes. 
///
/// \note	Required shared memory per thread block of size N: 8 * N bytes.
///
/// \author	Mathias Neumann
/// \date	17.04.2010
///
/// \param [in,out]	d_ioGeoVars	Geometric variations for each quadtree node. Stored in the same
/// 							order as it is used for QuadTreeSP. 
/// \param	levelChildren		The level of child nodes to work on. 
/// \param	idxChildStart		The child start node index. 
/// \param	numNodesChild		Number of child nodes. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_NormalizeGeoVariation(float* d_ioGeoVars, uint levelChildren, 
											 uint idxChildStart, uint numNodesChild)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numNodesChild)
	{
		__shared__ float s_geomVars[256];
		__shared__ float s_sums[256];

		uint idxNode = idxChildStart + tid;

		// We have the following order of nodes: The children of a given node are ordered
		// sequentially, so we just have to sum up all four children values and divide them
		// by four to get normalized geometric variations.

		// First read variations into shared memory. This works since the block size is multiple of 4.
		s_geomVars[threadIdx.x] = d_ioGeoVars[idxNode];
		s_sums[threadIdx.x] = s_geomVars[threadIdx.x];
		__syncthreads();

		// Sum up every four components of s_geomVars. To avoid sleeping threads, just build the
		// sum for each thread. This should also avoid bank conflicts.
		uint relIdx = threadIdx.x & 3; // % 4
		uint baseIdx = threadIdx.x - relIdx;
		float other1 = s_geomVars[baseIdx + ((relIdx+1) & 3)];
		float other2 = s_geomVars[baseIdx + ((relIdx+2) & 3)];
		float other3 = s_geomVars[baseIdx + ((relIdx+3) & 3)];
		s_sums[threadIdx.x] += other1 + other2 + other3;
		__syncthreads();

		// Now our normalized geometric variation is just:
		if(s_sums[threadIdx.x] > 0.f)
			d_ioGeoVars[idxNode] = s_geomVars[threadIdx.x] / s_sums[threadIdx.x];
		else
			d_ioGeoVars[idxNode] = 0.25f;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_DistributeSamplesToLevel(float* d_geoVars, float* d_qtNodeCounts,
/// 	float* d_randoms, uint lvlSrc, uint idxStartLvl, uint numNodesLvl, uint* d_numSamplesSrc,
/// 	uint* d_outNumSamplesDest)
///
/// \brief	Distributes samples relative to normalized geometric variation from parent nodes of
/// 		level \a lvlSrc to child nodes.
/// 		
/// 		Remaining samples that could not assigned in a ratio based way are moved into random
/// 		child nodes to avoid leaving out samples. 
///
/// \author	Mathias Neumann
/// \date	17.04.2010
///
/// \todo	Adding some randomness to the distribution might help to avoid clumping of the
/// 		samples in edges and so on. 
///
/// \param [in]		d_geoVars			The normalized geometric variations for each quadtree
/// 									node. Stored in the same order as it is used for
/// 									QuadTreeSP. 
/// \param [in]		d_qtNodeCounts		Shading point coutns for each quadtree node. Stored in the 
/// 									same order as it is used for QuadTreeSP. Used to assign
///										only as many samples to a node as there are shading points.
/// \param [in]		d_randoms			Uniform random numbers, one for each node, e.g. \a numNodesLvl
///										in total.
/// \param	lvlSrc						Source level. Samples from this level are distributed to
///										child level.
/// \param	idxStartLvl					Start index for source level in quadtree node. 
/// \param	numNodesLvl					Number of nodes on given source level. 
/// \param [in]		d_numSamplesSrc		Number of samples for source level nodes. 
/// \param [out]	d_outNumSamplesDest	Number of samples for child level nodes. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_DistributeSamplesToLevel(float* d_geoVars, float* d_qtNodeCounts, 
												float* d_randoms,
												uint lvlSrc, uint idxStartLvl, uint numNodesLvl,
												uint* d_numSamplesSrc, uint* d_outNumSamplesDest)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numNodesLvl)
	{
		uint idxStartLvlChild = idxStartLvl + numNodesLvl;
		uint idxFirstChild = idxStartLvlChild + 4*tid;

		// Read in normalized geometric variations of children. Also read shading point counts
		// as we can only distribute as many samples as there are shading points.
		float geoVarChild[4];
		uint spCountChild[4];
		for(uint i=0; i<4; i++)
		{
			geoVarChild[i] = d_geoVars[idxFirstChild + i];
			spCountChild[i] = (uint)d_qtNodeCounts[idxFirstChild + i];
		}

		// Read how many samples we have to distribute.
		uint toDistribute = d_numSamplesSrc[tid];

		// Distribute samples relative to normalized geometric variation.
		uint childSamples[4];
		for(uint i=0; i<4; i++)
			childSamples[i] = min(spCountChild[i], uint((float)toDistribute * geoVarChild[i]));

		// Compute how many samples we have to distribute randomly.
		int missing = toDistribute - childSamples[0] - childSamples[1] - childSamples[2] - childSamples[3];

		// The following technique might be somewhat suboptimal...
		float rnd = d_randoms[tid];
		for(int i=0; i<missing; i++)
		{
			// Generate next random using linear congruency RNG.
			rnd = dev_RandomLCGUniform(uint(rnd*4294967296.0f));

			// Determine child.
			uint child = 3;
			if(rnd < geoVarChild[0] && childSamples[0] < spCountChild[0])
				child = 0;
			else if(rnd < geoVarChild[0] + geoVarChild[1] && childSamples[1] < spCountChild[1])
				child = 1;
			else if(rnd < geoVarChild[0] + geoVarChild[1] + geoVarChild[2] && childSamples[2] < spCountChild[2])
				child = 2;

			if(childSamples[child] < spCountChild[child])
				childSamples[child]++;
			else
			{
				// Use another child to ensure the sample is distributed.
				if(childSamples[0] < spCountChild[0])
					childSamples[0]++;
				else if(childSamples[1] < spCountChild[1])
					childSamples[1]++;
				else if(childSamples[2] < spCountChild[2])
					childSamples[2]++;
				else
					childSamples[3]++;
			}
		}

		// Write child samples.
		for(uint i=0; i<4; i++)
			d_outNumSamplesDest[4*tid + i] = childSamples[i];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_CreateInitialClusterList(QuadTreeSP quadTree, uint idxStartLeafs,
/// 	uint numLeafs, uint* d_numSamplesLeaf, uint* d_clusterOffsets, ClusterList outClusters)
///
/// \brief	Create the initial sample list from given leaf sample distribution.
/// 		
/// 		As kernel_DistributeSamplesToLevel() provides a virutal distribution only, this
/// 		kernel selects the leaf's position and normal for each virtual sample assigned to the
/// 		leaf. 
///
/// \author	Mathias Neumann
/// \date	17.04.2010
///
/// \todo	Add multiple samples per leaf support. Currently I just use the quadtree leaf's position
///			and normal for all samples assigned to a leaf. This does not cover the problem of multiple
///			shading points for a single pixel (i.e. leaf).
///
/// \param	quadTree					The quad tree. 
/// \param	idxStartLeafs				The start index for leaf level. 
/// \param	numLeafs					Number of leafs. 
/// \param [in]		d_numSamplesLeaf	Number of samples for each leaf. One leaf might have gotten
///										multiple samples.
/// \param [in]		d_clusterOffsets	Cluster offsets for each leaf. Should contain scanned sample
///										counts and is used to write clusters at correct positions.
/// \param	outClusters					The clusters (i.e. samples) generated. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_CreateInitialClusterList(QuadTreeSP quadTree, uint idxStartLeafs, uint numLeafs,
												uint* d_numSamplesLeaf, uint* d_clusterOffsets, 
												ClusterList outClusters)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numLeafs)
	{
		uint idxNode = idxStartLeafs + tid;
		uint idxFirstCluster = d_clusterOffsets[tid];
		uint numSamples = d_numSamplesLeaf[tid];

		// Just get position and normal from leaf node.
		float3 p_node = make_float3(quadTree.d_positions[idxNode]);
		float3 n_nodeTemp = make_float3(quadTree.d_normals[idxNode]);
		// Normalize quad tree normal. This is requried.
		float3 n_node = normalize(n_nodeTemp);

		// Fill our samples with this information.
		for(uint i=0; i<numSamples; i++)
		{
			uint offset = idxFirstCluster + i;
			outClusters.d_positions[offset] = make_float4(p_node);
			outClusters.d_normals[offset] = make_float4(n_node);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ConstructCluster2SPAssoc(ShadingPoints spHits,
/// 	ClusterList clusters, float queryRadiusSqr, PairList outAssoc, float* d_outGeoVar)
///
/// \brief	Construct association between clusters and shading points. 
///
///			For each shading point, dev_RangeSearchBestFit() is called to find the best fitting
///			cluster center (according to geometric variation). A pair list of (cluster index,
///			shading point index) is constructed.
///			
///	\note	There might be clusters without any shading points. The other way around: There might
///			be shading points for which the range search could not find any cluster centers. Currently 
///			I assign a virtual cluster index to these shading points.
///
/// \author	Mathias Neumann
/// \date	18.04.2010
///
/// \param	spHits				The shading points.
/// \param	clusters			The cluster center list. 
/// \param	queryRadiusSqr		The query radius for range search (squared). 
/// \param	outAssoc			Will contain the constructed pairs.
/// \param [out]	d_outGeoVar	Will contain the geometric variation for each pair, i.e. the
///								geometric variation of the shading point to the corresponding
///								cluster center. Might be ::MN_INFINITY in case no cluster center
///								was found.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ConstructCluster2SPAssoc(ShadingPoints spHits, ClusterList clusters,
												float queryRadiusSqr,
												PairList outAssoc,
												float* d_outGeoVar)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < spHits.numPoints)
	{
		// Read out shading point data.
		float3 p_sp = make_float3(spHits.d_ptInter[tid]);
		float3 n_sp = make_float3(spHits.d_normalsS[tid]);

		int idxCluster = -1;
		float myGeoVar = MN_INFINITY;

		/*int idxOther = -1;
		float minError = MN_INFINITY;
		for(uint i=0; i<clusters.numClusters; i++)
		{
			float3 p_c = make_float3(clusters.d_positions[i]);
			float3 n_c = make_float3(clusters.d_normals[i]);
			
			float err = dev_GeometricVariation(c_alpha, p_sp, n_sp, p_c, n_c);
			if(err < minError)
			{
				idxOther = i;
				minError = err;
				myGeoVar = 0.f;
			}
		}*/

		// Ask kd-tree for best cluster center...
		idxCluster = dev_RangeSearchBestFit(p_sp, queryRadiusSqr, n_sp, myGeoVar);

		// Set cluster index to virtual index when nothing found.
		if(idxCluster == -1)
			idxCluster = clusters.numClusters + 1;

		// Add pair (cluster index, shading point index) to association list.
		outAssoc.d_first[tid] = idxCluster;
		outAssoc.d_second[tid] = tid;

		// Store geometric variation for shading point. This will be used to generate the
		// final cluster centers.
		d_outGeoVar[tid] = myGeoVar;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_CheckReclassifciation(ClusterList clustersOld, PairList cluster2sp,
/// 	float* d_geoVarsPairs, uint* d_outIsSPUnclassified, uint* d_outIsClusterNonEmpty)
///
/// \brief	Checks the provided reclassification of shading points to old cluster centers.
/// 		
/// 		A shading point's reclassification to an old cluster is accepted only if the
/// 		geometric variation shading point - cluster center is less than the maximunm
/// 		geometric variation stored for the old cluster center. 
///
/// \author	Mathias Neumann
/// \date	02.11.2010
///
/// \param	clustersOld						Old cluster center list. Has to contain old geometric
/// 										variation maxima. 
/// \param	cluster2sp						Pair list (cluster idx, shading point idx)
/// 										representing the reclassification. 
/// \param [in]		d_geoVarsPairs			Geometric variation correpsonding to each pair in the
/// 										pair list. 
/// \param [out]	d_outIsSPUnclassified	Binary 0/1 array. Will contain 1 for illegal
/// 										reclassifications, else 0. 
/// \param [in,out]	d_outIsClusterNonEmpty	Binary 0/1 array. Will contain 1 for non-empty clusters,
/// 										else 0. Has to be initialized with 0 before kernel
/// 										call. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_CheckReclassifciation(ClusterList clustersOld, PairList cluster2sp,
											 float* d_geoVarsPairs,
											 uint* d_outIsSPUnclassified, uint* d_outIsClusterNonEmpty)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < cluster2sp.numPairs)
	{
		uint idxCluster = cluster2sp.d_first[tid];
		uint idxSP = cluster2sp.d_second[tid];
		float geoVarNew = d_geoVarsPairs[tid];

		// Compare new geo var to old maximum. SPs assigned to the virtual cluster are
		// regarded as unclassified.
		bool isVirtualCluster = (idxCluster == clustersOld.numClusters + 1);
		float geoVarOldMax = 0.f;
		if(!isVirtualCluster)
			geoVarOldMax = tex1Dfetch(tex_clusterGeoVar, idxCluster);
		uint unclassified = 1;
		if(!isVirtualCluster && geoVarNew <= geoVarOldMax)
		{
			// Reclassification valid.
			unclassified = 0;
			// Write 1 to cluster non-empty array as the corresponding cluster is *not* empty.
			// Initialization with zeros assures that this works parallely, as write conflicts
			// are unproblematic.
			d_outIsClusterNonEmpty[idxCluster] = 1;
		}

		d_outIsSPUnclassified[idxSP] = unclassified;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_VisualizeClusters(ShadingPoints spHits, PairList cluster2sp,
/// 	uint numClusters, float4* d_outScreenBuffer)
///
/// \brief	Visualizes the clusters by coloring contained shading points.
/// 		
/// 		The kernel assumes a constructed (cluster index, shading point index) pair list. For
/// 		each shading point a greyscale color is chosen based on its cluster index. 
///
/// \author	Mathias Neumann
/// \date	18.04.2010 
/// \see	kernel_ConstructCluster2SPAssoc()
///
/// \param	spHits						The shading points. 
/// \param	cluster2sp					(cluster index, shading point index) pair list. 
/// \param	numClusters					Number of clusters. 
/// \param [in,out]	d_outScreenBuffer	Screen buffer that will contain the greyscale values
/// 									computed. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_VisualizeClusters(ShadingPoints spHits, PairList cluster2sp, uint numClusters,
										 float4* d_outScreenBuffer)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < cluster2sp.numPairs)
	{
		// Read cluster index and shading point index.
		uint idxCluster = cluster2sp.d_first[tid];
		uint idxSP = cluster2sp.d_second[tid];

		// Get pixel for shading point.
		uint idxPixel = spHits.d_pixels[idxSP];

		// Build a color from the cluster index.
		float4 clr = make_float4(float((8*idxCluster) & 255) / 256.f); // % 256
		if(idxCluster == numClusters + 1)
			clr = make_float4(0.f, 1.f, 0.f, 0.f);
		d_outScreenBuffer[idxPixel] = clr;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_MarkNonEmptyClusters(float* d_spCounts, uint numClusters,
/// 	uint* d_outClusterMarks)
///
/// \brief	Marks clusters containing shading points.
/// 		
/// 		A consequence of the averaging process is that there might be cluster centers without
/// 		any points. This kernel can be used to generate a binary 0/1 array to compact the
/// 		list of cluster centers to remove all empty clusters. 
///
/// \author	Mathias Neumann
/// \date	30.04.2010
///
/// \param [in]		d_spCounts			Shading point count for each cluster. Stored in \c float
/// 									array for convenience. 
/// \param	numClusters					Number of clusters. 
/// \param [out]	d_outClusterMarks	Binary 0/1 array marking all non empty nodes. Can be used
/// 									to eliminate empty clusters using compact primitive. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_MarkNonEmptyClusters(float* d_spCounts, uint numClusters, 
											uint* d_outClusterMarks)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numClusters)
	{
		uint isNonEmpty = ((d_spCounts[tid] >= 1.0f) ? 1 : 0);
		d_outClusterMarks[tid] = isNonEmpty;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GetFinalClusterIndices(float* d_geoVars, PairList cluster2sp,
/// 	uint numClusters, uint* d_outMinIndices)
///
/// \brief	Determines the indices of the shading points with minimal geometric variation within
/// 		each cluster.
/// 		
/// 		It is assumed that a segmented reduction was performed on the \a d_geoVars array and
/// 		that the resulting per-cluster geometric variations can be accessed using the
/// 		tex_clusterGeoVar texture. 
///
/// \author	Mathias Neumann
/// \date	19.04.2010
///
/// \param [in]		d_geoVars		The geometric variations of shading point to cluster in the
/// 								same order as the pairs in \a cluster2sp. 
/// \param	cluster2sp				(cluster index, shading point index) pair list. 
/// \param	numClusters				Number of clusters. 
/// \param [out]	d_outMinIndices	The i-th entry contain the index of the shading point with
/// 								minimum geometric variation to the i-th cluster. When there
/// 								is no such shading point, the i-th entry is undefined. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GetFinalClusterIndices(float* d_geoVars, PairList cluster2sp, 
											  uint numClusters,
											  uint* d_outMinIndices)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < cluster2sp.numPairs)
	{
		// Get cluster and shading point index.
		uint idxCluster = cluster2sp.d_first[tid];
		uint idxSP = cluster2sp.d_second[tid];
		bool isVirtualCluster = (idxCluster == numClusters+1);

		// Get cluster's minimum geometric variation from texture.
		float minGeoVar;
		if(!isVirtualCluster)
			minGeoVar = tex1Dfetch(tex_clusterGeoVar, idxCluster);

		// Check if we are the minimum. Note that we might have multiple minima within a cluster.
		// However this access conflict is no problem since we only write an index. It doesn't matter
		// which index is written as long it's geometric variation is minimal.
		if(!isVirtualCluster && d_geoVars[tid] == minGeoVar)
			d_outMinIndices[idxCluster] = idxSP;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_VisualizeClusterCenters(ShadingPoints clusters,
/// 	float4* d_outScreenBuffer)
///
/// \brief	Visualizes cluster centers by setting the corresponding pixel to white.
/// 		
/// 		Very simple visualization and probably not enough for printable images as pixels are
/// 		just too tiny. 
///
/// \author	Mathias Neumann
/// \date	19.04.2010
///
/// \param	clusters					The cluster list. 
/// \param [in,out]	d_outScreenBuffer	Screen buffer that will be enriched with cluster centers
/// 									as white pixels. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_VisualizeClusterCenters(ShadingPoints clusters, float4* d_outScreenBuffer)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < clusters.numPoints)
	{
		// Get pixel for shading point.
		uint idxPixel = clusters.d_pixels[tid];

		// Set pixel to white.
		d_outScreenBuffer[idxPixel] = make_float4(1.f);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_VisualizeInitialDistribution(ShadingPoints spHits,
/// 	uint* d_idxShadingPt, uint* d_numSamplesQTLeaf, uint numLeafs, float4* d_outScreenBuffer)
///
/// \brief	Visualizes the initial sample distribution over quadtree leafs.
/// 		
/// 		Takes as an input an array that connects quadtree leafs to their corresponding
///			shading point indices and an array that gives the number of samples for each leaf.
///			The corresponding pixels are colored with increasing brightness according to the
///			number of samples assigned to them.
///
/// \author	Mathias Neumann
/// \date	26.04.2010
///
/// \param	spHits						The shading points.
/// \param [in]		d_idxShadingPt		Shading point index for each quadtree leaf.
/// \param [in]		d_numSamplesQTLeaf	Number of samples contained in each quadtree leaf.
/// \param	numLeafs					Number of quadtree leafs. 
/// \param [in,out]	d_outScreenBuffer	Screen buffer that will contain a greyscale color
///										to represent the initial sample distribution.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_VisualizeInitialDistribution(ShadingPoints spHits, uint* d_idxShadingPt,
													uint* d_numSamplesQTLeaf, 
													uint numLeafs, 
													float4* d_outScreenBuffer)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numLeafs)
	{
		// Get pixel index.
		uint idxSP = d_idxShadingPt[tid];
		uint idxPixel = spHits.d_pixels[idxSP];

		// Set pixel to color relative to sample count.
		const uint maxCount = 2;
		uint myCount = d_numSamplesQTLeaf[tid];
		float myClr = min(1.f, float(myCount) / float(maxCount));
		d_outScreenBuffer[idxPixel] = make_float4(myClr);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_BestFitIrradianceDistrib(ShadingPoints spHits, PairList cluster2sp,
/// 	uint numClusters, float4* d_irrCluster, float4* d_ioIrrSP)
///
/// \brief	Assigns cluster irradiance values to all cluster members.
/// 		
/// 		Can be seen as an interpolation-less alternative to kernel_WangIrradianceDistrib(). 
///
/// \author	Mathias Neumann
/// \date	20.04.2010
///
/// \param	spHits					The shading points. 
/// \param	cluster2sp				(cluster index, shading point index) pair list. 
/// \param	numClusters				Number of clusters. 
/// \param [in]		d_irrCluster	Irradiance at each cluster center. 
/// \param [in,out]	d_ioIrrSP		Irradiance at each shading point, assigned as described
/// 								above. Note that this is a shading point related buffer and
/// 								does not represent a screen buffer of pixel values. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_BestFitIrradianceDistrib(ShadingPoints spHits, PairList cluster2sp,
												uint numClusters, float4* d_irrCluster,
											    float4* d_ioIrrSP)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < cluster2sp.numPairs)
	{
		// Read cluster index and shading point index.
		uint idxCluster = cluster2sp.d_first[tid];
		uint idxSP = cluster2sp.d_second[tid];

		float4 irrCluster = make_float4(0.f);

		// Use irradiance from cluster if cluster non-virtual.
		if(idxCluster != numClusters+1) // non-virtual cluster?
			irrCluster = d_irrCluster[idxCluster];
		d_ioIrrSP[idxSP] = irrCluster;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_WangIrradianceDistrib(ShadingPoints spHits, float* d_queryRadii,
/// 	float4* d_ioIrrSP)
///
/// \brief	Irradiance sample interpolation as proposed by \ref lit_wang "[Wang et al. 2009]".
///
///			This kernel basically calls dev_RangeSearchInterpolate() for each shading point,
///			using a shading point specific query radius. The irradiance assigned to the shading
///			point is computed by dividing the returned weighted irradiance sum by the weight sum.
///			If the latter is zero, the irradiance is set to zero, too.
///
/// \author	Mathias Neumann
/// \date	22.04.2010
///
/// \param	spHits					The shading points. 
/// \param [in]		d_queryRadii	Precomputed query radius for each shading point.
/// \param [in,out]	d_ioIrrSP		Irradiance at each shading point, assigned as described above.
///									Note that this is a shading point related buffer and does not
///									represent a screen buffer of pixel values.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_WangIrradianceDistrib(ShadingPoints spHits,
											 float* d_queryRadii,
											 float4* d_ioIrrSP)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < spHits.numPoints)
	{
		// Read sp position and normal.
		float3 p_sp = make_float3(spHits.d_ptInter[tid]);
		float3 n_sp = make_float3(spHits.d_normalsS[tid]);
		// Read query radius.
		float queryRadius = d_queryRadii[tid];

		// Perform interpolation using range search.
		float4 weightedIrrSum;
		float weightSum = dev_RangeSearchInterpolate(p_sp, queryRadius*queryRadius, n_sp, weightedIrrSum);

		// Write irradiance.
		float4 irr = make_float4(0.f, 0.f, 0.f, 0.f);
		if(weightSum > 0.f)
			irr = weightedIrrSum / weightSum;
		d_ioIrrSP[tid] = irr;
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void AFGSetGeoVarAlpha(float geoVarAlpha)
///
/// \brief	Sets the geometric variation alpha constant.
///
/// \author	Mathias Neumann
/// \date	October 2010
/// \see	dev_GeometricVariation(), c_alpha
///
/// \param	geoVarAlpha	Geometric variation alpha. Should be scaled to scene geometry.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void AFGSetGeoVarAlpha(float geoVarAlpha)
{
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_alpha", &geoVarAlpha, sizeof(float)));
}

/// Unbinds texture references used for cluster data for adaptive final gathering.
extern "C"
void AFGCleanupClusterData()
{
	if(!f_dataInitialized)
		return;

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_clusterP));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_clusterN));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_kdCluster));

	f_dataInitialized = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void AFGSetClusterData(const KDTreeData& kdCluster,
/// 	float4* d_posCluster, float4* d_normCluster, uint numClusters)
///
/// \brief	Updates cluster data for adaptive final gathering.
///
///			Binds data to textures.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param	kdCluster				The cluster kd-tree (has to be built). 
/// \param [in]		d_posCluster	Cluster center positions.
/// \param [in]		d_normCluster	Cluster center normals.
/// \param	numClusters				Number of clusters (cluster centers). 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void AFGSetClusterData(const KDTreeData& kdCluster, float4* d_posCluster, float4* d_normCluster,
					   uint numClusters)
{
	// Cleanup first (if required).
	if(f_dataInitialized)
		AFGCleanupClusterData();

	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_ConstructCluster2SPAssoc, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_WangIrradianceDistrib, cudaFuncCachePreferL1));

	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_kdCluster", &kdCluster, sizeof(KDTreeData)));

	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdUint2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);

	// Bind photon positions.
	tex_clusterP.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_clusterP, d_posCluster, cdFloat4, numClusters*sizeof(float4)));
	// Bind normals.
	tex_clusterN.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_clusterN, d_normCluster, cdFloat4, numClusters*sizeof(float4)));

	// Bind kd-tree stuff.
	tex_kdCluster.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_kdCluster, kdCluster.d_preorderTree, cdUint, 
		kdCluster.sizeTree*sizeof(uint)));

	f_dataInitialized = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void AFGSetInterpolationData(float* d_meanReciDists,
/// 	float4* d_irrCluster, uint numClusters)
///
/// \brief	Updates interpolation related cluster data. 
///
///			Binds data to textures.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param [in]		d_meanReciDists	Reciprocal mean distance at each cluster center.
/// \param [in]		d_irrCluster	Irradiance at each cluster center.
/// \param	numClusters				Number of cluster centers. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void AFGSetInterpolationData(float* d_meanReciDists, float4* d_irrCluster, uint numClusters)
{
	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	tex_meanReciDists.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_meanReciDists, d_meanReciDists, cdFloat, 
		numClusters*sizeof(float)));
	tex_clusterIrr.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_clusterIrr, d_irrCluster, cdFloat4, 
		numClusters*sizeof(float4)));
}

/// Unbinds texture references used for interpolation related cluster data.
extern "C"
void AFGCleanupInterpolationData()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_meanReciDists));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_clusterIrr));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_ConstructQT2SPAssoc() kernel call.
extern "C"
void KernelAFGConstructQT2SPAssoc(const ShadingPoints& spHits, uint numQTLevels,
							      PairList& outAssoc)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(spHits.numPoints, blockSize.x), 1, 1);

	kernel_ConstructQT2SPAssoc<<<gridSize, blockSize>>>(spHits, numQTLevels, outAssoc);
	MNCUDA_CHECKERROR;

	outAssoc.numPairs = numQTLevels * spHits.numPoints;
}

/// Wraps kernel_ComputeGeometricVariation() kernel call.
extern "C"
void KernelAFGComputeGeometricVariation(const ShadingPoints& spHits, const QuadTreeSP& quadTree,
									    float* d_outGeoVar)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(spHits.numPoints, blockSize.x), 1, 1);

	kernel_ComputeGeometricVariation<<<gridSize, blockSize>>>(spHits, quadTree, d_outGeoVar);
	MNCUDA_CHECKERROR;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelAFGNormalizeGeoVariation(const QuadTreeSP& quadTree,
/// 	float fPropagationFactor)
///
/// \brief	Normalizes geometric variations with optional propagation of geometric variation.
///
///			Wraps calls of kernel_PropagateGeoVariation() and kernel_NormalizeGeoVariation(). Takes
///			care that the kernels are correctly called to achieve bottom-up and top-down runs
///			respectively.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param	quadTree			The quadtree. It is assumed that all components of the structure
///								are correctly initialized.
/// \param	fPropagationFactor	The propagation factor. Pass \c 0.f to disable propagation.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelAFGNormalizeGeoVariation(const QuadTreeSP& quadTree, float fPropagationFactor)
{
	uint idxLvlStart;
	uint numLeafs = (1 << (quadTree.numLevels-1))*(1 << (quadTree.numLevels-1)); // 4^(numLevels-1)
	uint idxLeafStart = quadTree.numNodes - numLeafs;
	uint numPreLeafs = numLeafs / 4;

	// Propagate child geometric variation to parent geometric variation bottom-up.
	if(fPropagationFactor != 0.f)
	{
		idxLvlStart = idxLeafStart - numPreLeafs;
		for(uint lvl=quadTree.numLevels-2; lvl>0; lvl--) // No need to do this for root node.
		{
			uint numNodesLvl = (1 << lvl)*(1 << lvl); // 4^lvl
			uint idxChildStart = idxLvlStart + numNodesLvl;
			uint numNodesChild = numNodesLvl << 2;

			dim3 blockSize = dim3(256, 1, 1);
			dim3 gridSize = dim3(MNCUDA_DIVUP(numNodesLvl, blockSize.x), 1, 1);

			kernel_PropagateGeoVariation<<<gridSize, blockSize>>>(quadTree.d_geoVars, idxLvlStart, numNodesLvl, 
					idxChildStart, numNodesChild, fPropagationFactor);
			MNCUDA_CHECKERROR;

			idxLvlStart -= numNodesLvl / 4;
		}
	}

	// Normalize geometric variations for child nodes. This is done by checking the four
	// child nodes for each node on the current level. The variations for these child
	// nodes are normalized by dividing them through their sum.
	idxLvlStart = 0;
	for(uint lvl=0; lvl<quadTree.numLevels-1; lvl++)
	{
		uint numNodesLvl = (1 << lvl)*(1 << lvl); // 4^lvl
		uint idxChildStart = idxLvlStart + numNodesLvl;
		uint numNodesChild = numNodesLvl << 2;

		dim3 blockSize = dim3(256, 1, 1);
		dim3 gridSize = dim3(MNCUDA_DIVUP(numNodesChild, blockSize.x), 1, 1);

		kernel_NormalizeGeoVariation<<<gridSize, blockSize>>>(quadTree.d_geoVars, lvl+1, idxChildStart, numNodesChild);
		MNCUDA_CHECKERROR;

		idxLvlStart = idxChildStart;
	}
}

/// Wraps kernel_DistributeSamplesToLevel() kernel call.
extern "C"
void KernelAFGDistributeSamplesToLevel(float* d_geoVars, float* d_qtNodeCounts,
									   float* d_randoms,
									   uint lvlSrc, uint idxStartLvl, uint numNodesLvl,
									   uint* d_numSamplesSrc, uint* d_outNumSamplesDest)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numNodesLvl, blockSize.x), 1, 1);

	kernel_DistributeSamplesToLevel<<<gridSize, blockSize>>>(d_geoVars, d_qtNodeCounts, d_randoms,
		lvlSrc, idxStartLvl, numNodesLvl,
		d_numSamplesSrc, d_outNumSamplesDest);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_CreateInitialClusterList() kernel call.
extern "C"
void KernelAFGCreateInitialClusterList(const QuadTreeSP& quadTree, uint idxStartLeafs, uint numLeafs,
									   uint* d_numSamplesLeaf, uint* d_clusterOffsets, 
									   ClusterList& outClusters)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numLeafs, blockSize.x), 1, 1);

	kernel_CreateInitialClusterList<<<gridSize, blockSize>>>(quadTree, idxStartLeafs, numLeafs, d_numSamplesLeaf,
		d_clusterOffsets, outClusters);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ConstructCluster2SPAssoc() kernel call.
extern "C"
void KernelAFGConstructCluster2SPAssoc(const ShadingPoints& spHits, const ClusterList& clusters,
									   float queryRadius,
									   PairList& outAssoc,
									   float* d_outGeoVar)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(spHits.numPoints, blockSize.x), 1, 1);

	kernel_ConstructCluster2SPAssoc<<<gridSize, blockSize>>>(spHits, clusters, queryRadius*queryRadius, 
		outAssoc, d_outGeoVar);
	MNCUDA_CHECKERROR;

	outAssoc.numPairs = spHits.numPoints;
}

/// Wraps kernel_CheckReclassifciation() kernel call.
extern "C"
void KernelAFGCheckReclassifciation(const ClusterList& clustersOld, const PairList& cluster2sp,
									float* d_geoVarsPairs,
									uint* d_outIsSPUnclassified, uint* d_outIsClusterNonEmpty)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(cluster2sp.numPairs, blockSize.x), 1, 1);

	mncudaInitConstant(d_outIsClusterNonEmpty, clustersOld.numClusters, (uint)0);

	// Bind error maxima to texture to get caching.
	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_clusterGeoVar.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_clusterGeoVar, clustersOld.d_geoVarMax, 
		cdFloat, clustersOld.numClusters*sizeof(float)));

	kernel_CheckReclassifciation<<<gridSize, blockSize>>>(clustersOld, cluster2sp, d_geoVarsPairs, 
		d_outIsSPUnclassified, d_outIsClusterNonEmpty);
	MNCUDA_CHECKERROR;

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_clusterGeoVar));
}

/// Wraps kernel_VisualizeClusters() kernel call.
extern "C"
void KernelAFGVisualizeClusters(const ShadingPoints& spHits, const PairList& cluster2sp, uint numClusters,
							    float4* d_outScreenBuffer)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(cluster2sp.numPairs, blockSize.x), 1, 1);

	kernel_VisualizeClusters<<<gridSize, blockSize>>>(spHits, cluster2sp, numClusters, 
		d_outScreenBuffer);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_MarkNonEmptyClusters() kernel call.
extern "C"
void KernelAFGMarkNonEmptyClusters(float* d_spCounts, uint numClusters, 
							       uint* d_outClusterMarks)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numClusters, blockSize.x), 1, 1);

	kernel_MarkNonEmptyClusters<<<gridSize, blockSize>>>(d_spCounts, numClusters, d_outClusterMarks);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_GetFinalClusterIndices() kernel call.
extern "C"
void KernelAFGGetFinalClusterIndices(float* d_geoVars, float* d_geoVarMinima, uint numClusters,
								     const PairList& cluster2sp, uint* d_outMinIndices)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(cluster2sp.numPairs, blockSize.x), 1, 1);

	// Bind error minima to texture to get caching.
	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_clusterGeoVar.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_clusterGeoVar, d_geoVarMinima, 
		cdFloat, numClusters*sizeof(float)));

	kernel_GetFinalClusterIndices<<<gridSize, blockSize>>>(d_geoVars, cluster2sp, numClusters, 
		d_outMinIndices);
	MNCUDA_CHECKERROR;

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_clusterGeoVar));
}

/// Wraps kernel_VisualizeClusterCenters() kernel call.
extern "C"
void KernelAFGVisualizeClusterCenters(const ShadingPoints& clusters, float4* d_outScreenBuffer)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(clusters.numPoints, blockSize.x), 1, 1);

	kernel_VisualizeClusterCenters<<<gridSize, blockSize>>>(clusters, d_outScreenBuffer);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_VisualizeInitialDistribution() kernel call.
extern "C"
void KernelAFGVisualizeInitialDistribution(ShadingPoints spHits, uint* d_idxShadingPt,
										   uint* d_outNumSamplesQTLeaf, 
										   uint numLeafs, 
										   float4* d_outScreenBuffer)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numLeafs, blockSize.x), 1, 1);

	kernel_VisualizeInitialDistribution<<<gridSize, blockSize>>>(spHits, d_idxShadingPt, d_outNumSamplesQTLeaf,
		numLeafs, d_outScreenBuffer);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_BestFitIrradianceDistrib() kernel call.
extern "C"
void KernelAFGBestFitIrradianceDistrib(const ShadingPoints& spHits, const PairList& cluster2sp,
							 uint numClusters, float4* d_irrCluster, float4* d_ioIrrSP)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(cluster2sp.numPairs, blockSize.x), 1, 1);

	kernel_BestFitIrradianceDistrib<<<gridSize, blockSize>>>(spHits, cluster2sp, numClusters, 
		d_irrCluster, d_ioIrrSP);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_WangIrradianceDistrib() kernel call.
extern "C"
void KernelAFGWangIrradianceDistrib(const ShadingPoints& spHits, float* d_queryRadii, float4* d_ioIrrSP)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(spHits.numPoints, blockSize.x), 1, 1);

	kernel_WangIrradianceDistrib<<<gridSize, blockSize>>>(spHits, d_queryRadii, d_ioIrrSP);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////