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
/// \file	GPU\kdtree_tri.cu
///
/// \brief	Kernels for kd-tree construction, specifically for triangle kd-trees.
///
/// \author	Mathias Neumann
/// \date	01.04.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"		// for TriangleData
#include "MNUtilities.h"
#include "MNCudaUtil.h"
#include "kd-tree/KDKernelDefs.h"

// Defined in kdtree.cu
extern cudaDeviceProp f_DevProps;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDNodeListTri
///
/// \brief	Slim version of KDNodeList to avoid parameter space overflows.
/// 		
/// 		Also enables simpler access to triangle bounds by using explicit variable names. Used
/// 		for trianlge kd-tree construction kernels. 
///
/// \author	Mathias Neumann
/// \date	02.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDNodeListTri
{
#ifdef __cplusplus
public:
	/// Initializes helper struct from given node list.
	void Initialize(const KDNodeList& src)
	{
		numNodes = src.numNodes;
		d_idxFirstElem = src.d_idxFirstElem;
		d_numElems = src.d_numElems;
		d_aabbMinTight = src.d_aabbMinTight;
		d_aabbMinInherit = src.d_aabbMinInherit;
		d_aabbMaxTight = src.d_aabbMaxTight;
		d_aabbMaxInherit = src.d_aabbMaxInherit;
		d_elemNodeAssoc = src.d_elemNodeAssoc;
		d_aabbTriMin = src.d_elemPoint1;
		d_aabbTriMax = src.d_elemPoint2;
	}
#endif // __cplusplus

	/// See KDNodeList::numNodes.
	uint numNodes;

	/// See KDNodeList::d_idxFirstElem.
	uint* d_idxFirstElem;
	/// See KDNodeList::d_numElems.
	uint* d_numElems;
	/// See KDNodeList::d_aabbMinTight.
	float4* d_aabbMinTight;
	/// See KDNodeList::d_aabbMaxTight.
	float4* d_aabbMaxTight;
	/// See KDNodeList::d_aabbMinInherit.
	float4* d_aabbMinInherit;
	/// See KDNodeList::d_aabbMaxInherit.
	float4* d_aabbMaxInherit;

	/// See KDNodeList::d_elemNodeAssoc.
	uint* d_elemNodeAssoc;
	/// See KDNodeList::d_elemPoint1.
	float4* d_aabbTriMin;
	/// See KDNodeList::d_elemPoint2.
	float4* d_aabbTriMax;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	TDVertexData
///
/// \brief	Slim version of TriangleData to avoid parameter space overflows.
///
///			This structure stores the vertices only. Used for trianlge kd-tree construction kernels.
///
/// \author	Mathias Neumann
/// \date	01.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct TDVertexData
{
#ifdef __cplusplus
public:
	/// Initializes helper struct from given triangle data.
	void Initialize(const TriangleData& td)
	{
		for(uint i=0; i<3; i++)
			d_verts[i] = td.d_verts[i];
		numTris = td.numTris;
	}
#endif // __cplusplus

	/// See TriangleData::d_verts.
	float4* d_verts[3];
	/// See TriangleData::numTris.
	uint numTris;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	PointList
///
/// \brief	Simple point list for split clipping.
/// 		
/// 		Keeps track of the clipped triangle points during kd-tree construction's splitt
/// 		clipping. As all segments of the triangle are clipped against all sides of the node's
/// 		AABB, the number of segments, i.e. the number of points might increase.
///
/// \note	This list will be stored in local memory.
///
/// \author	Mathias Neumann
/// \date	01.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct PointList
{
	/// Current point count.
	uint count;
	/// \brief	Point array.
	///
	///			Currently I allow up to 8 clipped points. This was enough for my test scenes. Using
	///			less points I got overflows. Note that there have to be at least 6 points, as there can
	///			be clipped triangles with that many points.
	float3 pts[8];
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_ClipSegment(float3 pStart, float3 pEnd, uint axis, float axisValue,
/// 	float sense, PointList& ioClipped)
///
/// \brief	Clips a segment at a given bounding box side.
/// 		
/// 		Checks on which side the segment end points are. If not both points are in the inside,
/// 		a clipped point is generated and added to the clipped point list. Furthermore, when
/// 		the end point is inside, it is also added right after the clipped point. 
///
/// \author	Mathias Neumann
/// \date	25.03.2010
///
/// \param	pStart				Segment start point. 
/// \param	pEnd				Segment end point. 
/// \param	axis				The axis (x = 0, y = 1, z = 2) of the AABB side. 
/// \param	axisValue			Axis value giving the position of the AABB side. 
/// \param	sense				Sense of the side. If if \c 1.f is passed, it's a minimum side,
/// 							else, if \c -1.f is passed, it's a maximum side. All other values
/// 							are illegal. 
/// \param [in,out]	ioClipped	The clipped point list to update. Points are added at the end. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_ClipSegment(float3 pStart, float3 pEnd, uint axis, float axisValue, float sense,
								PointList& ioClipped)
{
	float* fStart = (float*)&pStart;
	float* fEnd = (float*)&pEnd;

	bool isInsideStart = (fStart[axis] - axisValue) * sense >= 0;
	bool isInsideEnd = (fEnd[axis] - axisValue) * sense >= 0;

	if(isInsideStart != isInsideEnd)
	{
		float t = (axisValue - fStart[axis]) / (fEnd[axis] - fStart[axis]);
		float3 pHit = pStart + t*(pEnd-pStart);

		// Ensure the hit is exactly on bounds.
		((float*)&pHit)[axis] = axisValue;

		ioClipped.pts[ioClipped.count] = pHit;
		ioClipped.count++;
	}

	// Do not forget the end point if it lies inside.
	if(isInsideEnd)
	{
		ioClipped.pts[ioClipped.count] = pEnd;
		ioClipped.count++;
	}

	/*if(ioClipped.count == 9)
		printf("KD - ERROR: CLIP LIST OVERFLOW!\n");*/
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_ClipSegments(const PointList& lstPoints, uint axis, float axisValue,
/// 	float sense, PointList& outClipped)
///
/// \brief	Clips all segments at a given bounding box side.
///
///			Clipping of a given segment is performed by dev_ClipSegment(). All segments are
///			handled in an iterative way.
///
/// \author	Mathias Neumann
/// \date	25.03.2010
///
/// \param	lstPoints			The point list to clip.
/// \param	axis				The axis (x = 0, y = 1, z = 2) of the AABB side. 
/// \param	axisValue			Axis value giving the position of the AABB side. 
/// \param	sense				Sense of the side. If if \c 1.f is passed, it's a minimum side,
/// 							else, if \c -1.f is passed, it's a maximum side. All other values
/// 							are illegal.
/// \param [out]	outClipped	Here the clipped list of points will be returned.
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_ClipSegments(const PointList& lstPoints, uint axis, float axisValue, float sense,
								 PointList& outClipped)
{
	outClipped.count = 0;

	for(uint i=0; i<lstPoints.count; i++)
	{
		const float3& pt1 = lstPoints.pts[i];
		uint ip1 = i + 1;
		if(ip1 == lstPoints.count)
			ip1 = 0;
		const float3& pt2 = lstPoints.pts[ip1];

		dev_ClipSegment(pt1, pt2, axis, axisValue, sense, outClipped);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_ClipTriToBounds(float3 verts[3], float3 aabbMin, float3 aabbMax,
/// 	float* aabbTriMin, float* aabbTriMax, PointList& outList)
///
/// \brief	Clips triangle to the given bounding box.
/// 		
/// 		Basically, each triangle segment is clipped against each side of the bounding box.
/// 		Hence, this function calls dev_ClipSegments() multiple times, i.e. once for each
/// 		side. 
///
///			Basic algorithm idea from http://graphics.stanford.edu/papers/i3dkdtree/.
///
/// \author	Mathias Neumann
/// \date	25.03.2010
///
/// \param	verts				The triangle vertices. 
/// \param	aabbMin				The AABB minimum. 
/// \param	aabbMax				The AABB maximum. 
/// \param [in]	aabbTriMin		Triangle AABB minimum. Used to avoid clipping on sides where the
///								triangle has no extent.
/// \param [in]	aabbTriMax		Triangle AABB maximum. Used to avoid clipping on sides where the
///								triangle has no extent.
/// \param [out]	outList		The clipped point list. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_ClipTriToBounds(float3 verts[3], float3 aabbMin, float3 aabbMax,
									float* aabbTriMin, float* aabbTriMax,
									PointList& outList)
{
	PointList lstOther;

	// Construct initial lists.
	outList.count = 3;
	#pragma unroll
	for(uint i=0; i<3; i++)
		outList.pts[i] = verts[i];

	// Clip against axes.
	// Minimum side: Sense == 1.f
	// Maximum side: Sense == -1.f

	// No need to clip in case the current triangle bounds have no extent in current axis.
	// The neccessary clipping is performed by the caller then.
	if(aabbTriMin[0] != aabbTriMax[0])
	{
		dev_ClipSegments(outList, 0, aabbMin.x, 1.f, lstOther);
		dev_ClipSegments(lstOther, 0, aabbMax.x, -1.f, outList);
	}
	if(aabbTriMin[1] != aabbTriMax[1])
	{
		dev_ClipSegments(outList, 1, aabbMin.y, 1.f, lstOther);
		dev_ClipSegments(lstOther, 1, aabbMax.y, -1.f, outList);
	}
	if(aabbTriMin[2] != aabbTriMax[2])
	{
		dev_ClipSegments(outList, 2, aabbMin.z, 1.f, lstOther);
		dev_ClipSegments(lstOther, 2, aabbMax.z, -1.f, outList);
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GenTriAABBs(KDNodeListTri lstRoot, TDVertexData td)
///
/// \brief	Computes the axis alligned bounding boxes for all triangles in the given node list. 
///
///			It is assumed that the ENA is initialized with identity values. Therefore no ENA access
///			is required to find the triangles corresponding to node's elements.
///
/// \author	Mathias Neumann
/// \date	14.02.2010
/// \note	Input structures were reduced to avoid parameter space overflows.
///
/// \param	lstRoot		The node list containing the root node only.
/// \param	td			Triangle data. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GenTriAABBs(KDNodeListTri lstRoot, TDVertexData td)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < td.numTris)
	{
		float4 v0 = td.d_verts[0][idx];
		float4 v1 = td.d_verts[1][idx];
		float4 v2 = td.d_verts[2][idx];
		
		lstRoot.d_aabbTriMin[idx] = 
			make_float4(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)),
						fminf(v0.z, fminf(v1.z, v2.z)), 0.f);
		lstRoot.d_aabbTriMax[idx] = 
			make_float4(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)),
						fmaxf(v0.z, fmaxf(v1.z, v2.z)), 0.f);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_PerformSplitClipping(KDNodeListTri lstNext, KDChunkList lstChunks,
/// 	TDVertexData td, uint* d_splitAxis, float* d_splitPos)
///
/// \brief	Perform split clipping for given node list.
///
///			Split clipping is performed to reduce the triangle AABBs according to the node AABBs.
///			It was proposed by Havran, "Heuristic Ray Shooting Algorithms", 2000. Each triangle is
///			clipped by dev_ClipTriToBounds(). The kernel works on all triangles in parallel by
///			using a chunk list constructed for the considered node list.
///
/// \note	Input structures were reduced to avoid parameter space overflows.
///
/// \author	Mathias Neumann
/// \date	25.03.2010
///
/// \param	lstNext				The list of nodes for which split clipping shall be performed.
///								This is usually the next list in large node stage.
/// \param	lstChunks			A list of chunks for the node list.
/// \param	td					Vertex data for triangles.
/// \param [in]		d_splitAxis	The split axes used to generate the child nodes in \a lstNext.
///								This is the corresponding array KDNodeList::d_splitAxis of the
///								parent node list, i.e. the active list.
/// \param [in]		d_splitPos	The split positions used to generate the child nodes in \a lstNext.
///								This is the corresponding array KDNodeList::d_splitPos of the
///								parent node list, i.e. the active list.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_PerformSplitClipping(KDNodeListTri lstNext, KDChunkList lstChunks,
											TDVertexData td, uint* d_splitAxis, float* d_splitPos)
{
	uint chk = MNCUDA_GRID2DINDEX;
	uint idx = threadIdx.x;

	__shared__ uint s_numTrisChunk;
	__shared__ uint s_idxFirstTri;
	__shared__ float3 s_aabbNodeMin;
	__shared__ float3 s_aabbNodeMax;
	__shared__ uint s_splitAxis;
	__shared__ float s_splitPos;
	if(threadIdx.x == 0)
	{
		s_numTrisChunk = lstChunks.d_numElems[chk];
		uint idxNode = lstChunks.d_idxNode[chk];
		s_idxFirstTri = lstChunks.d_idxFirstElem[chk];

		// Get split information.
		uint idxNodeParent = idxNode;
		uint numNodesParent = lstNext.numNodes >> 1;
		if(idxNode >= numNodesParent)
			idxNodeParent -= numNodesParent;
		s_splitAxis = d_splitAxis[idxNodeParent];
		s_splitPos = d_splitPos[idxNodeParent];

		// Get node's inherited bounds. Tight bounds are not yet available at this point.
		s_aabbNodeMin = make_float3(lstNext.d_aabbMinInherit[idxNode]);
		s_aabbNodeMax = make_float3(lstNext.d_aabbMaxInherit[idxNode]);
	}
	__syncthreads();

	uint splitAxis = s_splitAxis;
	float splitPos = s_splitPos;
	if(idx < s_numTrisChunk)
	{
		uint idxTNA = s_idxFirstTri + idx;

		// Read triangle AABB.
		float4 temp = lstNext.d_aabbTriMin[idxTNA];
		float aabbTriMin[3] = { temp.x, temp.y, temp.z };
		temp = lstNext.d_aabbTriMax[idxTNA];
		float aabbTriMax[3] = { temp.x, temp.y, temp.z };

		// Preread for coalesced access...
		uint idxTri = lstNext.d_elemNodeAssoc[idxTNA];

		// We use < here since triangles on the split plane do not lead to clipping.
		bool isLeft = aabbTriMin[splitAxis] < splitPos;
		bool isRight = splitPos < aabbTriMax[splitAxis];

		if(isLeft && isRight)
		{
			// Read triangle vertices.
			// WARNING: We CANNOT reduce clipping to the non-split axes. The clipping on the
			//          split axis is CRITICAL, e.g. for triangles lying fully in a parent node.
			float3 verts[3];
			for(uint i=0; i<3; i++)
				verts[i] = make_float3(td.d_verts[i][idxTri]);

			// Clip the triangle to our node bounds.
			PointList lstClipped;
			dev_ClipTriToBounds(verts, s_aabbNodeMin, s_aabbNodeMax, aabbTriMin, aabbTriMax, lstClipped);

			/*if(lstClipped.count == 0)
			{
				printf("KD - WARNING: Illegal clip (axis: %d, pos %.3f). TriMin: %.8f, %.3f, %.3f TriMax: %.8f, %.3f, %.3f, ChildMin: %.3f, %.3f, %.3f, ChildMax: %.8f, %.3f, %.3f\n", 
					splitAxis, splitPos, aabbTriMin[0], aabbTriMin[1], aabbTriMin[2], aabbTriMax[0], aabbTriMax[1], aabbTriMax[2],
					s_aabbNodeMin.x, s_aabbNodeMin.y, s_aabbNodeMin.z, s_aabbNodeMax.x, s_aabbNodeMax.y, s_aabbNodeMax.z);
			}*/

			// Now we have the points. Rebuild the triangle bounds.
			for(uint a=0; a<3; a++)
			{
				aabbTriMin[a] = MN_INFINITY;
				aabbTriMax[a] = -MN_INFINITY;
				for(uint i=0; i<lstClipped.count; i++)
				{
					aabbTriMin[a] = fminf(((float*)&lstClipped.pts[i])[a], aabbTriMin[a]);
					aabbTriMax[a] = fmaxf(((float*)&lstClipped.pts[i])[a], aabbTriMax[a]);
				}
			}
		}

#ifndef __DEVICE_EMULATION__
		__syncthreads();
#endif
		/*if( s_aabbNodeMin.x > aabbTriMax[0] || s_aabbNodeMax.x < aabbTriMin[0] ||
			s_aabbNodeMin.y > aabbTriMax[1] || s_aabbNodeMax.y < aabbTriMin[1] ||
			s_aabbNodeMin.z > aabbTriMax[2] || s_aabbNodeMax.z < aabbTriMin[2])
			printf("KD - ERROR: Illegal clippped triangle bounds detected.\n");*/

		// Ensure the bounds are within the node's bounds on split axis.
		aabbTriMin[splitAxis] = fmaxf(((float*)&s_aabbNodeMin)[splitAxis], aabbTriMin[splitAxis]);
		aabbTriMax[splitAxis] = fminf(((float*)&s_aabbNodeMax)[splitAxis], aabbTriMax[splitAxis]);

		// Write clipped data.
		lstNext.d_aabbTriMin[idxTNA] = make_float4(aabbTriMin[0], aabbTriMin[1], aabbTriMin[2], 0.f);
		lstNext.d_aabbTriMax[idxTNA] = make_float4(aabbTriMax[0], aabbTriMax[1], aabbTriMax[2], 0.f);
	}
}


//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{



/// Wraps kernel_GenTriAABBs() kernel call.
extern "C"
void KernelKDGenerateTriAABBs(const KDNodeList& lstRoot, const TriangleData& td)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(td.numTris, blockSize.x), 1, 1);

	TDVertexData vtd;
	vtd.Initialize(td);

	KDNodeListTri lstTri;
	lstTri.Initialize(lstRoot);

	// Generate AABBs using kernel.
	kernel_GenTriAABBs<<<gridSize, blockSize>>>(lstTri, vtd);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_PerformSplitClipping() kernel call.
extern "C"
void KernelKDPerformSplitClipping(const KDNodeList& lstActive, const KDNodeList& lstNext,
								  const KDChunkList& lstChunks, const TriangleData& td)
{
	dim3 blockSize = dim3(KD_CHUNKSIZE, 1, 1);
	dim3 gridSize = MNCUDA_MAKEGRID2D(lstChunks.numChunks, f_DevProps.maxGridSize[0]);

	TDVertexData vtd;
	vtd.Initialize(td);

	KDNodeListTri lstNextTri;
	lstNextTri.Initialize(lstNext);

	kernel_PerformSplitClipping<<<gridSize, blockSize>>>(lstNextTri, lstChunks, vtd, 
		lstActive.d_splitAxis, lstActive.d_splitPos);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////