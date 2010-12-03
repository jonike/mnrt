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
/// \file	kd-tree\KDTreePoint.h
///
/// \brief	Declares the KDTreePoint class. 
/// \author	Mathias Neumann
/// \date	02.04.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_KDTREEPOINT_H__
#define __MN_KDTREEPOINT_H__

#pragma once

#include "KDTreeGPU.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	KDTreePoint
///
/// \brief	Subclass of KDTreeGPU for point kd-trees.
/// 		
/// 		This subclass realizes the GPU-based kd-tree construction of point kd-trees by
/// 		extending the KDTreeGPU class. This is done by implementing the AddRootNode() method.
/// 		In addition, it supports query radius estimation as suggested by
/// 		\ref lit_zhou "[Zhou et al. 2008]".
/// 		
/// 		For simple points, a single element point suffices. There is no need to compute any
///			bounding boxes. Furthermore split clipping is not required.
///
/// \author	Mathias Neumann
/// \date	02.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class KDTreePoint : public KDTreeGPU
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	KDTreePoint(float4* _d_points, uint numPoints, float3 sceneAABBMin, float3 sceneAABBMax,
	/// 	float maxQueryRadius)
	///
	/// \brief	Initializes this object. To build the tree, call BuildTree(). 
	///
	/// \author	Mathias Neumann
	/// \date	02.04.2010
	///
	/// \param [in]		_d_points	The three-dimensional points to use as elements. 
	/// \param	numPoints			Number of points. 
	/// \param	sceneAABBMin		The scene AABB minimum. 
	/// \param	sceneAABBMax		The scene AABB maximum. 
	/// \param	maxQueryRadius		The maximum query radius used for point queries. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	KDTreePoint(float4* _d_points, uint numPoints, float3 sceneAABBMin, float3 sceneAABBMax, 
		float maxQueryRadius);
	virtual ~KDTreePoint(void);

// Data
private:
	// Input data.
	float4* d_points;
	uint m_numPoints;

	// KNN radius refinement iterations.
	uint m_knnRefineIters;
	// KNN target count.
	uint m_knnTargetCount;
	// Precomputed radius estimate for each kd-tree node.
	float* d_nodeRadiusEstimate;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ComputeQueryRadii(float4* d_queryPoints, uint numQueryPoints, float* d_outRadii)
	///
	/// \brief	Estimates the query radius for given array of query points. 
	///
	///			For estimation, precomputed query radii for a set of kd-tree nodes are used to find
	///			an initial radius. After that, histogram-based query radius refinement is performed
	///			to find a query radius that allows to find approximately the required number of
	///			points within the resulting range.
	///
	/// \author	Mathias Neumann
	/// \date	September 2010
	/// \see	SetKNNRefineIters(), SetKNNTargetCount()
	///
	/// \param [in]	d_queryPoints		Device array of three-dimensional query points.
	/// \param	numQueryPoints			Number of query points. 
	/// \param [out]	d_outRadii		Device array of query radii. The memory has to be provided
	///									by the calller. The i-th component of this array will hold
	///									the estimated query radius for the i-th query point.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ComputeQueryRadii(float4* d_queryPoints, uint numQueryPoints, float* d_outRadii);

	virtual void Destroy();

// Accessors
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetKNNRefineIters(uint iters)
	///
	/// \brief	Sets the number of query radius refinement iterations.
	///
	///			For search radius estimation, a histogram-based refinement iteration is performed
	///			to improve an initial radius estimate. This parameter controls the number of iterations
	///			and therefore the quality of resulting search radii.
	///
	/// \author	Mathias Neumann
	/// \date	September 2010
	///
	/// \param	iters	New number of iterations to perform.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetKNNRefineIters(uint iters) { m_knnRefineIters = iters; }
	/// Returns current number of query radius refinement iterations.
	uint GetKNNRefineIters() const { return m_knnRefineIters; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetKNNTargetCount(uint targetCount)
	///
	/// \brief	Sets kNN search target count, that is the parameter k.
	///
	///			Required for query radius refinement.
	///
	/// \author	Mathias Neumann
	/// \date	August 2010
	///
	/// \param	targetCount	The k used for kNN searches.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetKNNTargetCount(uint targetCount) { m_knnTargetCount = targetCount; }
	/// Returns the current k for kNN searches.
	uint GetKNNTargetCount() const { return m_knnTargetCount; }

// Implementation:
protected:
	virtual void AddRootNode(KDNodeList* pList);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void PostBuild()
	///
	/// \brief	Cleans up auxiliary data after building.
	/// 		
	/// 		Additionally, this method estimates the query radii for a number of kd-tree node
	/// 		centers. 
	///
	/// \author	Mathias Neumann
	/// \date	September 2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void PostBuild();

private:
	// Precomputes the query radius for all kd-tree nodes using Zhou's algorithm.
	void PrecomputeQueryRadii();
};

#endif // __MN_KDTREEPOINT_H__