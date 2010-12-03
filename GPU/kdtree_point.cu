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
/// \file	GPU\kdtree_point.cu
///
/// \brief	Kernels for kd-tree construction, specifically for point kd-trees.
/// \author	Mathias Neumann
/// \date	06.10.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "kd-tree/KDKernelDefs.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"

// Histogram parameters.

/// Histogram size, i.e. number of entries, for query radius refinement.
#define KNN_HISTOGRAM_SIZE			32
/// Histogram size in bytes.
#define KNN_HISTOGRAM_SIZE_BYTES	64
///	Inverse of KNN_HISTOGRAM_SIZE_INV.
#define KNN_HISTOGRAM_SIZE_INV		0.03125f
/// \brief	Macro to compute the histogram index to use on a given thread.
///
///			The shared memory histogram is stored as one array for all threads in the thread
///			block, so the threads have to use their share of that array. Instead of using
///			continguous segments with the resulting bank conflicts, the following pattern is
///			employed:
///
///			\code |0...|.0..|..0.|...0| \endcode
///
///			Here 0 denotes the first bin of a histogram. So the first thread finds his
///			first bin in the first component of his first segment, the second thread finds it in
///			the second component of his second segment and so on. A segment is a block of bins,
///			i.e. of size ::KNN_HISTOGRAM_SIZE. Here the bins are shifted as described above. Since
///			::KNN_HISTOGRAM_SIZE equals the current warp size, bank conflicts are avoided.
#define KNN_HISTOGRAM_IDX(i)  (threadIdx.x*KNN_HISTOGRAM_SIZE + ((i + threadIdx.x) & (KNN_HISTOGRAM_SIZE - 1)))


// Constant memory data.

/// \brief	kNN radius refinement iterations.
///
/// \see	MNRTSettings::SetKNNRefineIters()
__constant__ uint c_knnRefineIters = 2; 
/// \brief	kNN target count, i.e. the k in kNN.
///
/// \see	MNRTSettings::GetTargetCountGlobal(), MNRTSettings::GetTargetCountCaustics()
__constant__ uint c_knnTargetCount = 50;
/// Point kd-tree constant memory variable.
__constant__ KDTreeData c_kdTreePt;


/// Point kd-tree texture for KDTreeData::d_preorderTree.
texture<uint, 1, cudaReadModeElementType> tex_kdTreePt;
/// Point data texture. Holds coordinates of the points stored in the kd-tree.
texture<float4, 1, cudaReadModeElementType> tex_ptCoords;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_RangeSearchHist(float3 ptQuery, float queryRadiusSqr,
/// 	unsigned short* s_histogram = NULL, float r_min = 0.f, float deltaR = 0.f)
///
/// \brief	Range search to update histogram used for query radius refinement.
///
///			This is a basic range search that increments the histogram bin corresponding to
///			the distance of the found points. Points with distance smaller than \a r_min
///			are moved into the 0-th column of \a s_histogram. All other points are mapped
///			to equidistant segments, where each segment has its own bin.
///
/// \author	Mathias Neumann
/// \date	08.04.2010
///
/// \param	ptQuery				The query point. 
/// \param	queryRadiusSqr		The query radius (squared). 
/// \param [in,out]	s_histogram	The histogram to update. It is assumed that the histogram was
///								set to zero before calling this function.
/// \param	r_min				The minimum distance represented by the histogram.
/// \param	deltaR				Distance range the histogram represents.
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_RangeSearchHist(float3 ptQuery, float queryRadiusSqr,
									unsigned short* s_histogram = NULL, float r_min = 0.f, float deltaR = 0.f)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree depth first to look for points q with ||p - q|| < r.
	int addrNode = 0;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_kdTreePt.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_kdTreePt.d_preorderTree[addrNode+1], c_kdTreePt.d_preorderTree[addrNode+2]);
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
				todoAddr[todoPos++] = addrOther;

			// Read node index + leaf info (MSB) for new node.
			idxNode = c_kdTreePt.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		// Now we have a leaf. Update histogram.
		uint numPoints = c_kdTreePt.d_preorderTree[addrNode+1];;
		for(uint i=0; i<numPoints; i++)
		{
			uint idxPoint = c_kdTreePt.d_preorderTree[addrNode+2+i];

			float4 pos4 = tex1Dfetch(tex_ptCoords, idxPoint);
			float3 pos = make_float3(pos4);

			float distSqr = dev_DistanceSquared(ptQuery, pos);
			if(distSqr < queryRadiusSqr)
			{
				// Notice that all points that are found within a radius of less than
				// r_min are put into the zero-th column!
				float dist = sqrtf(distSqr);
				float fIdx = fmaxf(dist - r_min, 0.f) * float(KNN_HISTOGRAM_SIZE) / deltaR;
				uint idxHist = (uint)fIdx;
				unsigned short oldValue = s_histogram[KNN_HISTOGRAM_IDX(idxHist)];
				if(oldValue < 0xfffe)
					s_histogram[KNN_HISTOGRAM_IDX(idxHist)] = oldValue+1;
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_RefineQueryRadiusZhou(float3 ptQuery, float radiusMax)
///
/// \brief	Histogram-based query radius refinement.
///
///			Algorithm was proposed by \ref lit_zhou "[Zhou et al. 2008]". A histogram is stored 
///			in a shared memory block (block size N):
///
///			\code __shared__ unsigned short s_histogram[N*KNN_HISTOGRAM_SIZE]; \endcode
///
///			The access pattern used to avoid bank conflicts can be found in the description of
///			::KNN_HISTOGRAM_IDX(). To save shared memory, \c short is used as element type.
///			The histogram is built using dev_RangeSearchHist(). According to the structure of
///			the histogram the radius range is refined, starting with [0, \a radiusMax].
///
/// \note	Required shared memory per thread block of size N: 2 * ::KNN_HISTOGRAM_SIZE * N bytes.
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \todo	Evaluate multiple threads per query.
///
/// \param	ptQuery		The query point. 
/// \param	radiusMax	The query radius maximum.
///
/// \return	Returns the determined query radius for given query point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_RefineQueryRadiusZhou(float3 ptQuery, float radiusMax)
{
	extern __shared__ unsigned short s_histogram[];

	float r_min = 0.f;
	float r_max = radiusMax;
	for(uint i=0; i<c_knnRefineIters; i++)
	{
		float r = r_max;
		float deltaR = r_max - r_min;

		// Set histogram to zero.
		for(uint j=0; j<KNN_HISTOGRAM_SIZE; j++)
			s_histogram[KNN_HISTOGRAM_IDX(j)] = 0;

		// Increment histogram entries depending on range search.
		dev_RangeSearchHist(ptQuery, r*r, s_histogram, r_min, deltaR);

		// Find histogram entry m.
		uint m = KNN_HISTOGRAM_SIZE-1;
		uint localSum = 0;
		for(uint j=0; j<KNN_HISTOGRAM_SIZE; j++)
		{
			localSum += s_histogram[KNN_HISTOGRAM_IDX(j)];
			if(localSum >= c_knnTargetCount)
			{
				m = j;
				break;
			}
		}

		float r_minOld = r_min;
		r_min = r_minOld + float(m) * deltaR * KNN_HISTOGRAM_SIZE_INV;
		r_max = r_minOld + float(m+1) * deltaR * KNN_HISTOGRAM_SIZE_INV;
	}

	// Use r_max as query radius.
	return r_max;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_QREMarkQualifiedNodes(KDTreeData kdTree, float maxNodeRadius,
/// 	uint* d_outIsQualified)
///
/// \brief	Marks nodes of the given point kd-tree whose radius is less than a given radius.
/// 		
/// 		Note that the radius of a parent is always larger than that of a child. Therefore
/// 		this selects all children of a qualified node, too. 
///
/// \author	Mathias Neumann
/// \date	08.07.2010
///
/// \param	kdTree						The point kd-tree. 
/// \param	maxNodeRadius				Maximum node radius. All nodes with radius less or equal
///										this radius are marked.
/// \param [out]	d_outIsQualified	Binary 0/1 array. Contains 1 for marked nodes. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_QREMarkQualifiedNodes(KDTreeData kdTree, float maxNodeRadius,
											 uint* d_outIsQualified)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < kdTree.numNodes)
	{
		// Read node extent and extract node radius.
		float4 nodeExtent = kdTree.d_nodeExtent[tid];
		float nodeRadius = nodeExtent.w;

		d_outIsQualified[tid] = ((nodeRadius <= maxNodeRadius) ? 1 : 0);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_QREEliminateChildren(KDTreeData kdTree, uint* d_inIsQualifiedOld,
/// 	uint* d_outIsQualifiedNew)
///
/// \brief	Removes marks from children whose parent is marked, too.
/// 		
/// 		This reduces the marked nodes to a cut of the tree. Note that this cannot be done
/// 		inplace since we cannot synchronize over all threads. If we do this inplace, we would
/// 		remove marks from inner nodes and could leave more deeper nodes still marked if they
/// 		are considered later. 
///
/// \author	Mathias Neumann
/// \date	08.07.2010
///
/// \param	kdTree						The point kd-tree. 
/// \param [in]		d_inIsQualifiedOld	Old binary 0/1 marker array.
/// \param [out]	d_outIsQualifiedNew	New binary 0/1 marker array.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_QREEliminateChildren(KDTreeData kdTree, uint* d_inIsQualifiedOld, 
											uint* d_outIsQualifiedNew)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < kdTree.numNodes)
	{
		// Get child node information.
		uint left = kdTree.d_childLeft[tid];
		uint right = kdTree.d_childRight[tid];
		uint marked = d_inIsQualifiedOld[tid];
		if(marked == 1 && left != right) // Are we a marked internal node?
		{
			// Force unmark of children. This operation works since a child node is
			// only child to a single parent (tree!).
			d_outIsQualifiedNew[left] = 0;
			d_outIsQualifiedNew[right] = 0;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_QREMarkChildrenOfMarked(KDTreeData kdTree, uint* d_inMarkedOld,
/// 	uint* d_ioMarkedNew)
///
/// \brief	Marks the children of all marked nodes.
/// 		
/// 		This cannot work inplace as we have to we cannot consider all nodes in a sychronized
/// 		way. This kernel seems to reverse what kernel_QREEliminateChildren() had done. This
/// 		is only partly true as this kernel selects one child "level" only, where the
/// 		mentioned kernel eliminated all child "levels". 
///
/// \author	Mathias Neumann
/// \date	15.10.2010
///
/// \param	kdTree					The point kd tree. 
/// \param [in,out]	d_inMarkedOld	Old binary 0/1 marker array.
/// \param [in,out]	d_ioMarkedNew	New binary 0/1 marker array.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_QREMarkChildrenOfMarked(KDTreeData kdTree, uint* d_inMarkedOld, uint* d_ioMarkedNew)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < kdTree.numNodes)
	{
		// Get child node information.
		uint left = kdTree.d_childLeft[tid];
		uint right = kdTree.d_childRight[tid];
		uint marked = d_inMarkedOld[tid];
		if(marked == 1 && left != right) // Are we a marked internal node?
		{
			// Force marks for the children.
			d_ioMarkedNew[left] = 1;
			d_ioMarkedNew[right] = 1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_QREComputeNodeQR(uint* d_idxNode, float4* d_nodeExtent,
/// 	uint numNodes, float queryRadiusMax, float* d_ioNodeRadiusEstimate)
///
/// \brief	Estimates a query radius for all given nodes of the kd-tree.
///
///			Uses dev_RefineQueryRadiusZhou() to refine the query radii with the node centers
///			as query points. Not all kd-tree nodes are examined, but only a subset described
///			by \a d_idxNode.
///
/// \note	Required shared memory per thread block of size N: 2 * ::KNN_HISTOGRAM_SIZE * N bytes.
///
/// \author	Mathias Neumann
/// \date	October 2010
///
/// \param [in]		d_idxNode				Indices of the nodes to examine.
/// \param [in]		d_nodeExtent			Node extent for nodes to examine.
/// \param	numNodes						Number of indices and extents. 
/// \param	queryRadiusMax					The query radius maximum. 
/// \param [in,out]	d_ioNodeRadiusEstimate	Query radius estimate array (for \em all tree nodes).
///											Will contain the estimated values at indices
///											given by \a d_idxNode. Hence
///											\code d_ioNodeRadiusEstimate[d_idxNode[i]] \endcode
///											are the computed estimates, 0 <= \c i < \a numNodes.
///											All other elements, i.e. all other node radii are left
///											unchanged.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_QREComputeNodeQR(uint* d_idxNode, float4* d_nodeExtent, uint numNodes, float queryRadiusMax,
										float* d_ioNodeRadiusEstimate)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numNodes)
	{
		float3 nodeCenter = make_float3(d_nodeExtent[tid]);
		float estimate = dev_RefineQueryRadiusZhou(nodeCenter, queryRadiusMax);

		// Write estimate at correct index.
		uint idxNode = d_idxNode[tid];
		d_ioNodeRadiusEstimate[idxNode] = estimate;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_QREEstimateRadii(float4* d_queryPoints, uint numQueryPoints,
/// 	float* d_nodeRadiusEstimate, float4* d_nodeExtents, float globalQR,
/// 	float* d_outRadiusEstimate)
///
/// \brief	Estimates query radius at given query points using node estimates. 
///
///			Estimation is done by tree traversial: Each node containing the query point a thread
///			works on is considered. That gives a path from the root node to some leaf. The
///			estimate is initialized with \a globalQR and updated as
///
///			\code estimate = fminf(estimate, length(ptQuery - nodeCenter) + nodeEstimate); \endcode
///
/// \author	Mathias Neumann
/// \date	12.07.2010
///
/// \param [in]		d_queryPoints			The query points.
/// \param	numQueryPoints					Number of query points. 
/// \param [in]		d_nodeRadiusEstimate	Node radius estimates as computed using 
///											kernel_QREComputeNodeQR().
/// \param [in]		d_nodeExtents			The node extents for all kd-tree nodes.
/// \param	globalQR						The global query radius maximum.
/// \param [out]	d_outRadiusEstimate		The radius estimate for each query point.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_QREEstimateRadii(float4* d_queryPoints, uint numQueryPoints, float* d_nodeRadiusEstimate,
										float4* d_nodeExtents,
										float globalQR, float* d_outRadiusEstimate)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numQueryPoints)
	{
		float3 ptQuery = make_float3(d_queryPoints[tid]);
		const float* p = (float*)&ptQuery;

		// Start with global query radius.
		float estimate = globalQR;

		// Traverse the photon map kd-tree to find all nodes p lies within.
		int addrNode = 0; // root
		while(addrNode >= 0)
		{
			// Read node index + leaf info (MSB).
			uint idxNode = tex1Dfetch(tex_kdTreePt, addrNode);
			uint isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;

			// Check node's radius estimate.
			float nodeEstimate = d_nodeRadiusEstimate[idxNode];
			float3 nodeCenter = make_float3(d_nodeExtents[idxNode]);
			estimate = fminf(estimate, length(ptQuery - nodeCenter) + nodeEstimate);

			// Find next node.
			if(isLeaf)
				addrNode = -1;
			else
			{
				uint addrLeft = addrNode + 1 + 2;
				uint2 parentInfo = make_uint2(c_kdTreePt.d_preorderTree[addrNode+1], c_kdTreePt.d_preorderTree[addrNode+2]);
				uint addrRight = parentInfo.x & 0x0FFFFFFF;
				uint splitAxis = parentInfo.x >> 30;
				float splitPos = *(float*)&parentInfo.y;

				// Choose next child node depending on position of p.
				if(p[splitAxis] > splitPos)
					addrNode = addrRight;
				else
					addrNode = addrLeft;
			}
		}
		

		// Pass out result.
		d_outRadiusEstimate[tid] = estimate;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_RefineRadiusZhou(float4* q_queryPoints, uint numPoints,
/// 	float* d_ioQueryRadius)
///
/// \brief	Performs query radius refinement for given query points.
///
///			This kernel assumes that the maximum radius was estimated for each node and is given
///			in \a d_ioQueryRadius. This estimation can be done using kernel_QREEstimateRadii().
///			Given such an estimate for the maximum radius, all this kernel does is calling
///			dev_RefineQueryRadiusZhou().
///
/// \note	Required shared memory per thread block of size N: 2 * ::KNN_HISTOGRAM_SIZE * N bytes.
///
/// \author	Mathias Neumann
/// \date	06.10.2010
///
/// \param [in]		q_queryPoints	The query points. 
/// \param	numPoints				Number of query points. 
/// \param [in,out]	d_ioQueryRadius	Initially, this array should contain an estimate for the
///									maximum query radius for each query point. After execution,
///									this array will contain the refined query radii for each
///									query point. The radii are chosen so that approximately
///									::c_knnTargetCount points can be found using such a radius.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_RefineRadiusZhou(float4* q_queryPoints, uint numPoints, float* d_ioQueryRadius)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numPoints)
	{
		// Get query point.
		float3 ptQuery = make_float3(q_queryPoints[tid]);
		// Get query radius maximum for this sample.
		float queryRadiusMax = d_ioQueryRadius[tid];

		float rRefined = dev_RefineQueryRadiusZhou(ptQuery, queryRadiusMax);

		d_ioQueryRadius[tid] = rRefined;
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KDSetTreePoint(const KDTreeData& kdTree, float4* d_points, uint numPoints,
/// 	uint knnRefineIters, uint knnTargetCount)
///
/// \brief	Sets up data for query radius estimation. 
///
/// \author	Mathias Neumann
/// \date	24.10.2010
///
/// \param	kdTree				The point kd-tree. 
/// \param [in]		d_points	Point data of points stored in the kd-tree. 
/// \param	numPoints			Number of points. 
/// \param	knnRefineIters		Number of kNN refinement iterations to use. 
/// \param	knnTargetCount		The k in kNN to use, i.e. the target count. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KDSetTreePoint(const KDTreeData& kdTree, float4* d_points, uint numPoints, 
					uint knnRefineIters, uint knnTargetCount)
{
	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdUint2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_knnRefineIters", &knnRefineIters, sizeof(uint)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_knnTargetCount", &knnTargetCount, sizeof(uint)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_kdTreePt", &kdTree, sizeof(KDTreeData)));

	// Bind positions.
	tex_ptCoords.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_ptCoords, d_points, 
		cdFloat4, numPoints*sizeof(float4)));

	// Bind kd-tree stuff.
	tex_kdTreePt.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_kdTreePt, kdTree.d_preorderTree, cdUint, 
		kdTree.sizeTree*sizeof(uint)));
}

/// Unbinds textures set up by ::KDSetTreePoint().
extern "C"
void KDUnsetTreePoint()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_ptCoords));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_kdTreePt));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelKDMarkEstimationNodes(const KDTreeData& kdTree,
/// 	float maxNodeRadius, uint* d_outIsQualified)
///
/// \brief	Marks a set of nodes for which query radius estimation should be performed.
///
///			First nodes with a radius of at most \a maxNodeRadius are marked, using
///			kernel_QREMarkQualifiedNodes(). Then all children marks are eliminated using
///			kernel_QREEliminateChildren(). Finally, two more "levels" of nodes are added using
///			two calls of kernel_QREMarkChildrenOfMarked().
///
/// \author	Mathias Neumann
/// \date	24.10.2010
///
/// \param	kdTree						The point kd-tree. 
/// \param	maxNodeRadius				The maximum node radius for the initial node marks.
/// \param [out]	d_outIsQualified	Binary 0/1 array that contains the marks.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelKDMarkEstimationNodes(const KDTreeData& kdTree, float maxNodeRadius, uint* d_outIsQualified)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(kdTree.numNodes, blockSize.x), 1, 1);

	// First mark all qualified nodes with radius less than alpha*R.
	kernel_QREMarkQualifiedNodes<<<gridSize, blockSize>>>(kdTree, maxNodeRadius, d_outIsQualified);
	MNCUDA_CHECKERROR;

	// Now eliminate all nodes whose parent is marked.
	MNCudaMemory<uint> d_tempMarks(kdTree.numNodes);
	mncudaSafeCallNoSync(cudaMemcpy(d_tempMarks, d_outIsQualified, 
		kdTree.numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	kernel_QREEliminateChildren<<<gridSize, blockSize>>>(kdTree, d_tempMarks, d_outIsQualified);
	MNCUDA_CHECKERROR;

	// Add marks for two more "levels", that is, mark all children of the currently marked nodes and
	// thereafter the children of all marked nodes again. See Zhou et al.
	mncudaSafeCallNoSync(cudaMemcpy(d_tempMarks, d_outIsQualified, 
		kdTree.numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	kernel_QREMarkChildrenOfMarked<<<gridSize, blockSize>>>(kdTree, d_outIsQualified, d_tempMarks);
	mncudaSafeCallNoSync(cudaMemcpy(d_outIsQualified, d_tempMarks, 
		kdTree.numNodes*sizeof(uint), cudaMemcpyDeviceToDevice));
	kernel_QREMarkChildrenOfMarked<<<gridSize, blockSize>>>(kdTree, d_tempMarks, d_outIsQualified);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_QREComputeNodeQR() kernel call.
extern "C"
void KernelKDComputeNodeQR(uint* d_idxNode, float4* d_nodeExtent, uint numNodes, float queryRadiusMax,
						   float* d_ioNodeRadiusEstimate)
{
	// We perform radius refinement here, so get good block size.
	// WARNING: Assumes max register count to be 32.
	uint numBlocks = mncudaGetMaxBlockSize(KNN_HISTOGRAM_SIZE_BYTES, 32);
	numBlocks /= 2;
	//MNMessage("Threads per block: %d.", numBlocks);

	dim3 blockSize = dim3(numBlocks, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numNodes, blockSize.x), 1, 1);
	size_t sharedMem = numBlocks*KNN_HISTOGRAM_SIZE_BYTES;

	kernel_QREComputeNodeQR<<<gridSize, blockSize, sharedMem>>>(d_idxNode, d_nodeExtent, numNodes, queryRadiusMax,
		d_ioNodeRadiusEstimate);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_QREEstimateRadii() kernel call.
extern "C"
void KernelKDEstimateRadii(float4* d_queryPoints, uint numQueryPoints, float* d_nodeRadiusEstimate,
						   float4* d_nodeExtents,
						   float globalQR, float* d_outRadiusEstimate)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numQueryPoints, blockSize.x), 1, 1);

	kernel_QREEstimateRadii<<<gridSize, blockSize>>>(d_queryPoints, numQueryPoints, d_nodeRadiusEstimate, 
		d_nodeExtents, globalQR, d_outRadiusEstimate);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_RefineRadiusZhou() kernel call.
extern "C"
void KernelKDRefineRadiusZhou(float4* q_queryPoints, uint numQueryPoints, float* d_ioQueryRadius)
{
	// We perform radius refinement here, so get good block size.
	// WARNING: Assumes max register count to be 32.
	uint numBlocks = mncudaGetMaxBlockSize(KNN_HISTOGRAM_SIZE_BYTES, 32);
	numBlocks /= 2;
	//MNMessage("Threads per block: %d.", numBlocks);

	dim3 blockSize = dim3(numBlocks, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numQueryPoints, blockSize.x), 1, 1);
	size_t sharedMem = numBlocks*KNN_HISTOGRAM_SIZE_BYTES;

	kernel_RefineRadiusZhou<<<gridSize, blockSize, sharedMem>>>(q_queryPoints, numQueryPoints, d_ioQueryRadius);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////