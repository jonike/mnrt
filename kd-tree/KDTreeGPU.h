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
/// \file	kd-tree\KDTreeGPU.h
///
/// \brief	Declares the KDTreeGPU class. 
/// \author	Mathias Neumann
/// \date	06.02.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	kdtreeCon	kd-Tree Construction
/// 
/// \brief	Components of MNRT used for kd-tree construction (both CPU- and GPU-based).
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_KDTREEGPU_H__
#define __MN_KDTREEGPU_H__

#pragma once

#include "KDKernelDefs.h"
#include <vector>
#include "../MNCudaMemPool.h"

class KDTreeListener;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	KDTreeGPU
///
/// \brief	GPU-based kd-tree implementation abstract base class.
/// 		
/// 		This class provides a GPU-based kd-tree construction algorithm that can be adapted
/// 		for different types of kd-trees (e.g. points or triangles as objects) within
/// 		subclasses. However, this extension possibility is somewhat limited due to problems
/// 		with the parallel implementation and might need some reworking to allow more types of
/// 		primitives.
/// 		
/// 		The implementation is based on the work of \ref lit_zhou "[Zhou et al. 2008]".
/// 		
/// 		This class uses pointers to the concrete elements to store in the kd-tree. These
/// 		pointers are indices for the array of elements. Within the construction algorithm,
/// 		the elements are in most cases hidden behind their AABBs. But at some places, e.g.
/// 		for splitting the nodes, there has to be special treatment depending on the primitive
/// 		type. Opposed to \ref lit_zhou "[Zhou et al. 2008]" I limit this special treatment
/// 		quite a bit to allow both point kd-trees and triangle kd-trees to be a subclass of
/// 		this class. This was done by introducing the number of \em element \em points, which
/// 		can be 1 or 2, depending on the primitive type. It is used for construction puropses
/// 		only.
/// 		
/// 		If the number of element points is 1, only one at most four-dimensional point can be
/// 		used per element. This is enough for a point kd-tree, where each element is just a
/// 		three-dimensional point. Similarly two four-dimensional points can be stored per
/// 		element, when the number of element points is set to 2. This is enough for triangle
/// 		kd-trees as the construction process mainly requires the AABB of the triangles.
/// 		Special handling like split clipping can be sourced out to the corresponding
/// 		subclass. This might be extended to other primtive objects as long as they can be
/// 		represented with at most two element points. 
///
/// \author	Mathias Neumann
/// \date	06.02.2010 
/// \see	KDTreePoint, KDTreeTri
////////////////////////////////////////////////////////////////////////////////////////////////////
class KDTreeGPU
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	KDTreeGPU(size_t numInputElems, uint numElementPoints, float3 rootAABBMin,
	/// 	float3 rootAABBMax)
	///
	/// \brief	Initializes this object. To build the tree, call BuildTree().
	/// 		
	/// 		Root AABB vertices are in most cases available before calling this method. To avoid
	/// 		internal computation, they have to be passed as parameters. The corresponding AABB
	/// 		may be less tight than possible. This can however decrease the performance.
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \param	numInputElems		Number of elements to store within the kd-tree. Note that this
	/// 							removes or at least restricts the ability to dynamically update the
	/// 							tree. 
	/// \param	numElementPoints	Number of element points that can be stored for each element of
	/// 							the kd-tree. Only relevant for construction of the tree. Has to
	/// 							be either 1 or 2. 
	/// \param	rootAABBMin			Root node AABB minimum vertex. 
	/// \param	rootAABBMax			Root node AABB maximum vertex. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	KDTreeGPU(size_t numInputElems, uint numElementPoints, float3 rootAABBMin, float3 rootAABBMax);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual ~KDTreeGPU(void)
	///
	/// \brief	Destructs this object. Calls Destroy().
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual ~KDTreeGPU(void);

// Data
protected:
	/// Currently active nodes list.
	KDNodeList* m_pListActive;
	/// Node list for next pass of algorithm.
	KDNodeList* m_pListNext;

	/// Chunk list used for node AABB calculation and element counting.
	KDChunkList* m_pChunkList;

	/// Root node bounding box minimum supplied by constructor. Not neccessarily tight.
	float3 m_rootAABBMin;
	/// Root node bounding box maximum supplied by constructor. Not neccessarily tight.
	float3 m_rootAABBMax;

private:
	// The final node list.
	KDTreeData* m_pKDData;

	// Input element count.
	size_t m_numInputElements;
	size_t m_numElementPoints;

	// Final internal node list.
	KDFinalNodeList* m_pListFinal;
	// Small nodes list.
	KDNodeList* m_pListSmall;

	// List the chunk list was build for. Avoids useless rebuilding.
	KDNodeList* m_pCurrentChunkListSource;
	// Split candidate list. Stores split candidates for root small nodes.
	KDSplitList* m_pSplitList;

	// Small root parents stored here as m_pListNode indices. This is used to
	// update the parent indices *when* the small roots are added to m_pListNode.
	// Without this there is no connection between the small roots and their
	// parents. The MSB of each value is used to distinguish between left (0) and
	// right (1) child.
	uint* d_smallRootParents;
	
	// Single value buffer used for return values and other stuff.
	MNCudaMemory<uint> d_tempVal;

	// Settings
	float m_fEmptySpaceRatio;
	uint m_nSmallNodeMax;
	// Conservative global estimate for KNN query radius. All queries will have at most this query radius.
	float m_fMaxQueryRadius;
	std::vector<KDTreeListener*> m_vecListeners;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool BuildTree()
	///
	/// \brief	Constructs the kd-tree for the elements supplied by subclasses.
	/// 		
	/// 		The process starts by requesting auxiliary structures (in most cases lists of nodes).
	/// 		After that, the root node is created using AddRootNode(). This step has to be defined
	/// 		by subclasses. Hence the concrete elements have to be supplied by the subclasses.
	/// 		Subsequently, large nodes are processed within the large node stage until no more
	/// 		large nodes are available. Then, within the small node stage, all small nodes are
	/// 		processed.
	/// 		
	/// 		In both stages, all final nodes are added to a final node list that represents the
	/// 		final kd-tree information. As node ordering in this final node list is somewhat
	/// 		chaotic, a final step is added to layout the tree in a new, more cache friendly way.
	/// 		This is done using a two tree traversals. To avoid high memory consumption, all
	/// 		auxiliary structures used for construction, e.g. lists of intermediate nodes or the
	/// 		chunk list, are destroyed before this method returns. 
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool BuildTree();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void Destroy()
	///
	/// \brief	Destroys the kd-tree by releasing both auxiliary structures and final kd-tree data.
	///
	///			Subclasses should overwrite this to destroy their own data. In this case, do not
	///			forget to call this base class method.
	///
	/// \author	Mathias Neumann
	/// \date	06.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void Destroy();

// Accessors
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	KDTreeData* GetData()
	///
	/// \brief	Gets final kd-tree data after construction.
	///
	/// \author	Mathias Neumann
	/// \date	February 2010
	///
	/// \return	Returns final kd-tree data.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	KDTreeData* GetData() { return m_pKDData; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetCustomBits(uint bitNo, uint* d_values)
	///
	/// \brief	Sets custom bits within final kd-tree data.
	/// 		
	/// 		The final kd-tree's parent information in KDTreeData::d_preorderTree contains two free
	/// 		bits that can be used for custom purposes. I introduced this to mark \e inner kd-tree
	/// 		nodes without adding a new array. This avoids adding another memory read or texture
	/// 		fetch, e.g. within GPU based traversal algorithms.
	/// 		
	/// 		\warning As this method requires the final kd-tree layout, only call it \em after
	/// 		 building the kd-tree with BuildTree(). 
	///
	/// \author	Mathias Neumann. 
	/// \date	September 2010. 
	///
	/// \param	bitNo				Number of the custom bit to manipulate. This can be either 0 or 1,
	/// 							but nothing else. 
	/// \param [in]		d_values	Values for the given custom bit. Has to be an binary array of
	/// 							KDTreeData::numNodes elements, that is only 0 or 1 are allowed as
	/// 							values. Note that the bits for leafs are ignored as leafs have
	///								no custom bits!
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetCustomBits(uint bitNo, uint* d_values);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetEmptySpaceRatio(float ratio)
	///
	/// \brief	Sets the empty space ratio \a ratio used for construction.
	/// 		
	/// 		If at one side of a node's bounding box are at least \a ratio times the AABB extent
	/// 		at the corresponding axis of free space, the space is cut off in form of an empty
	/// 		node. 
	///
	/// \param	ratio	The ratio. Has to be a value within [\c 0.0f, \c 1.0f]. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetEmptySpaceRatio(float ratio) { m_fEmptySpaceRatio = ratio; }
	/// Returns the current empty space ratio.
	float GetEmptySpaceRatio() const { return m_fEmptySpaceRatio; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetSmallNodeMax(uint maximum)
	///
	/// \brief	Sets the maximum of elements for small nodes.
	/// 		
	/// 		For performance reasons, larger kd-tree nodes are handled in a different way compared
	/// 		to small nodes, hence there is a large node stage and a small node stage. This
	/// 		property draws the line between small and large nodes. 
	///
	/// \param	maximum	The maximum number of elements a small node can contain. All larger nodes are
	/// 				considered as large nodes. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetSmallNodeMax(uint maximum);
	/// Returns the current small node element maximum.
	uint GetSmallNodeMax() const { return m_nSmallNodeMax; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetMaxQueryRadius(float radius)
	///
	/// \brief	Sets the maximum query radius.
	///
	///			The maximum query radius is required for split cost evaluation, e.g. for the VVH cost
	///			evaluation. It's somewhat misplaced here and would better fit into the corresponding
	///			subclass.
	///
	/// \param	radius	The maximum query radius. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetMaxQueryRadius(float radius) { m_fMaxQueryRadius = radius; }
	/// Returns the current maximum query radius.
	float GetMaxQueryRadius() const { return m_fMaxQueryRadius; }

	/// Registers kd-tree listener.
	void AddListener(KDTreeListener* pListener);
	/// Removes given kd-tree listener from the list of registered listeners.
	void RemoveListener(KDTreeListener* pListener);

// Implementation
protected:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void AddRootNode(KDNodeList* pList) = 0
	///
	/// \brief	Adds the root node to the given node list.
	///
	///			This method has to be specified by subclasses. It's responsible for creating and
	///			adding the kd-tree root node to the given node list. The concrete implementation
	///			has to copy all required data (see KDNodeList) for the root node into the first
	///			node list entry. This is usually a host to device copy operation.
	///
	///			Regarding AABBs: It is assumed that the subclass will only initialize the inherited
	///			root node bounds. Furthermore the subclass is responsible for calculating and
	///			initializing the element points for all root node elements.
	///			
	///
	/// \author	Mathias Neumann
	/// \date	02.04.2010
	///
	/// \param [in,out]	pList	Node list where the root node should be added. It is assumed that this
	///							list is \em empty.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void AddRootNode(KDNodeList* pList) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void PerformSplitClipping(KDNodeList* pListParent, KDNodeList* pListChild)
	///
	/// \brief	Performs split clipping for large nodes.
	/// 		
	/// 		The default implementation does nothing. This is enough for point kd-trees as points
	/// 		have a degenerated AABB that cannot be reduced by clipping the point to a child nodes
	/// 		AABB. But for other element types, e.g. triangles, split clipping can help improving
	/// 		kd-tree quality. Here the element is clipped to the child node bounds and a new
	/// 		element AABB is generated.
	///
	/// \author	Mathias Neumann. 
	/// \date	April 2010. 
	///
	/// \param [in]		pListParent	The list of parent nodes. This is usually the active list.
	/// \param [in,out]	pListChild	The list of child nodes that should be updated.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void PerformSplitClipping(KDNodeList* pListParent, KDNodeList* pListChild) {};

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void CreateChunkList(KDNodeList* pList)
	///
	/// \brief	Fills chunk list #m_pChunkList with chunks for the given node list.
	///
	///			The creation is only performed when the current chunk list #m_pChunkList is not
	///			created for the given node list \a pList. Structurally it might be better to source
	///			this out into the KDChunkList type and to allow multiple chunk lists. However, I opted
	///			to use a single chunk list as I wanted to reduce memory and time requirements.
	///
	/// \author	Mathias Neumann
	/// \date	February 2010
	///
	/// \param [in]	pList	The node list for which the current chunk list should be created.
	///						May not be \c NULL.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void CreateChunkList(KDNodeList* pList);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void PreBuild()
	///
	/// \brief	Initializes auxiliary structures and data.
	///
	///			This method is called right before BuildTree() performs the actual construction. It's
	///			default implementation is important and subclasses should not forget to call it.
	///
	/// \author	Mathias Neumann
	/// \date	July 2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void PreBuild();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void PostBuild()
	///
	/// \brief	Cleans up auxiliary data after building.
	/// 		
	/// 		This method is called right before BuildTree() returns. It's default implementation
	/// 		is important and subclasses should not forget to call it. 
	///
	/// \author	Mathias Neumann. 
	/// \date	July 2010. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void PostBuild();

private:
	// Implements the large node stage as described in the paper.
	void LargeNodeStage();
	// Processes large nodes in the current active list.
	void ProcessLargeNodes(uint* d_finalListIndexActive);
	// Computes per node bounding boxes.
	void ComputePerNodeAABBs();
	// Splits large nodes using empty space cutting and spatial median splitting
	// along the longest axis.
	void SplitLargeNodes(uint* d_finalListIndexActive);
	// Sorts and clips all triangles of the active TNA into the next TNA.
	void SortAndClipToChildNodes();
	// Compacts the element data from source to destination.
	uint CompactElementData(KDNodeList* pListDest, uint destOffset, uint nodeOffset, 
							KDNodeList* pListSrc, uint srcOffset, uint numSourceNodes,
							uint* d_validMarks, uint* d_countsUnaligned, uint numSegments = 1);
	// Updates the small list by removing all small nodes from the next list.
	void UpdateSmallList(uint* d_finalListIndexActive);

	// Implements the small node stage as described in the paper.
	void SmallNodeStage();
	// Preprocesses small nodes in the current small list.
	void PreProcessSmallNodes();
	// Processes small nodes in the current active list.
	void ProcessSmallNodes();

	// Performs preorder traversal of the final tree to generate a final node list.
	void PreorderTraversal();

// Helpers
private:
	// Tests given node AABB. Returns true if node AABB fine.
	bool TestNodeAABB(float4 aabbMinI, float4 aabbMaxI, float4 aabbMinT, float4 aabbMaxT,
							   bool bHasInheritedBounds, bool bHasTightBounds);
	// Tests the given node list.
	void TestNodeList(KDNodeList* pList, char* strTest,
					bool bHasInheritedBounds, bool bHasTightBoundsbool, bool bCheckElements);
	// Prints out the resulting tree.
	void PrintTree();
};

#endif // __MN_KDTREEGPU_H__