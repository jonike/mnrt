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
/// \file	kd-tree\KDTreeTri.h
///
/// \brief	Declares the KDTreeTri class. 
/// \author	Mathias Neumann
/// \date	02.04.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_KDTREETRI_H__
#define __MN_KDTREETRI_H__

#pragma once

#include "KDTreeGPU.h"

struct TriangleData;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	KDTreeTri
///
/// \brief	Subclass of KDTreeGPU for triangle kd-trees.
///
///			This subclass realizes the GPU-based kd-tree construction of triangle kd-trees by
///			extending the KDTreeGPU class. This is done by taking a TriangleData object as input
///			and overwriting the AddRootNode() and PerformSplitClipping() methods.
///
///			As the bounding boxes of the triangle suffice for the construction process, two
///			element points are used. These are minimum and maximum vertex of the triangle AABB.
///
/// \author	Mathias Neumann
/// \date	02.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class KDTreeTri : public KDTreeGPU
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	KDTreeTri(const TriangleData& td)
	///
	/// \brief	Initializes this object. To build the tree, call BuildTree().
	///
	/// \author	Mathias Neumann
	/// \date	02.04.2010
	///
	/// \param	td	Triangle information used for the elements the kd-tree should organize.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	KDTreeTri(const TriangleData& td);
	virtual ~KDTreeTri(void);

// Data
private:
	// Input data.
	const TriangleData* m_pTD;

// Implementation:
protected:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void AddRootNode(KDNodeList* pList)
	///
	/// \brief	Adds the root node to the given node list.
	///
	///			This method uses the provided TriangleData object (see KDTreeTri()) to copy the required
	///			information for the root node from host to device memory. Furthermore the element points
	///			are set by computing AABBs for all triangles.
	///
	/// \author	Mathias Neumann
	/// \date	02.04.2010
	///
	/// \param [in,out]	pList	Node list where the root node should be added. It is assumed that this
	///							list is \em empty.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void AddRootNode(KDNodeList* pList);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void PerformSplitClipping(KDNodeList* pListParent, KDNodeList* pListChild)
	///
	/// \brief	Performs split clipping for large nodes.
	/// 		
	/// 		For triangles, split clipping can be used to reduce the actual triangle AABB within
	/// 		child nodes after node splitting. This was suggested by Havran, "Heuristic Ray
	/// 		Shooting Algorithms", 2000. To parallelize the process, this method computes a chunk
	/// 		list for child node list. 
	///
	/// \author	Mathias Neumann
	/// \date	April 2010
	///
	/// \param [in]		pListParent	The list of parent nodes. This is usually the active list.
	/// \param [in,out]	pListChild	The list of child nodes that should be updated. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void PerformSplitClipping(KDNodeList* pListParent, KDNodeList* pListChild);
};

#endif // __MN_KDTREETRI_H__