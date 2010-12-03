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
/// \file	kd-tree\KDKernelDefs.h
///
/// \brief	Definitions and types used for GPU-based kd-tree construction.
/// \author	Mathias Neumann
/// \date	16.02.2010
/// \ingroup	kdtreeCon
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_KDKERNELDEFS_H__
#define __MN_KDKERNELDEFS_H__

#pragma once

#include "../MNMath.h"		// for uint
#include <vector_types.h>	// for float4, ...

// Forward declarations
struct KDNodeList;
struct KDFinalNodeList;

/// Element mask type used for element bit masks in small node stage. Currently up to 64 bit.
typedef unsigned long long ElementMask;

/// Maximum chunk size used for chunk lists. Should be power of 2.
#define KD_CHUNKSIZE		256

/// Maximum kd-tree height. Used to limit traversal stacks to fixed size.
#define KD_MAX_HEIGHT		50

/// Maximum number of elements for small nodes given by ::ElementMask.
#define KD_SMALLNODEMAX		64

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDTreeData
///
/// \brief	Final representation of kd-tree after construction.
///
///			The KDFinalNodeList representation that is generated during the construction process
///			is not sufficient for the needs of fast traversal algorithms. It contains too much
///			information without any real compression. Furthermore it does nothing
///			to improve cache performance.
///
///			\par Reorganizing Traversal Information
///			This structure by contrast avoids storing useless data and compresses traversal
///			related information to reduce memory accesses. To allow better cache performance,
///			traversal related information is reorganized in #d_preorderTree using the
///			following format:
///
///			\code root_idx | root_info | left_idx | left_info | left_subtree | right_idx | right_info | right_subtree \endcode
///
///			where
///			\li \c root_idx:		Root node index.
///			\li \c root_info:		Parent information for root, see below. Takes two \c uint.
///			\li \c left_idx:		Index of the left child node.
///			\li	\c left_info:		Parent information for left child node, see below. Takes two \c uint.
///			\li \c left_subtree:	All further nodes in the left subtree.
///
///			It is important to note that the whole left subtree is stored before the right child 
///			and its subtree. This helps improving cache performance. Indices are relative to the
///			order of the other arrays of this structure, e.g. #d_nodeExtent. They are enriched
///			with leaf information: The MSB is set, if the node is a leaf. Else the MSB is not set.
///			This improves leaf detection performance because it avoids reading child information.
///
///			The above format applies only for inner nodes. For leafs, instead of the parent
///			information, element count and element indices are stored. The element indices are
///			relative to the underlying data, e.g. triangle data or point data. Hence we have
///			the subsequent format for leafs:
///
///			\code leaf_idx | leaf_count | leaf_elmIndices \endcode
///
///			\par Parent information
///			As noted above, inner node representation contains a \em parent \em information. This is
///			a compressed form of what is needed during traversal. It takes two \c uint, hence two
///			elements of #d_preorderTree. It is organized the following way:
///
///			\li First Entry: [MSB] Split axis (2 bits) + Custom bits (2 bits) + Child above address (28 bits) [LSB]
///			\li Second Entry: Split position (\c float stored as \c uint).
///
///			Child above address is 0 for leaf nodes. Else it is the address of the right child
///			node in #d_preorderTree. The left child node address is not stored explicitly as it 
///			can be computed. Split axis takes 2 most significant bits. 0 stands for x-axis, 
///			1 for y-axis and 2 for z-axis. The custom bits are used to mark nodes. See
///			KDTreeGPU::SetCustomBits().
///
///
/// \note	I tried to remove the element indices from the layout to an external array, but this
///			did not improve traversal times.		
///
/// \author	Mathias Neumann
/// \date	19.03.2010
/// \see	KDFinalNodeList, KDNodeList
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDTreeData
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(KDFinalNodeList* pList, float3 aabbMin, float3 aabbMax)
	///
	/// \brief	Initializes device memory and reads out some information from given node list.
	///
	///			Currently, only element counts (#d_numElems) and left/right child indices (#d_childLeft, 
	///			#d_childRight) are transfered. Root node AABB information is required to avoid
	///			reading it back from GPU memory.
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	///
	/// \param [in]		pList	Unordered final node list from kd-tree construction.
	/// \param	aabbMin			Root node AABB minimum.
	/// \param	aabbMax			Root node AABB maximum. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(KDFinalNodeList* pList, float3 aabbMin, float3 aabbMax);
	/// Returns whether this list is empty.
	bool IsEmpty() { return numNodes == 0; }
	/// Frees CUDA memory.
	void Free();

#endif // __cplusplus

	/// Number of nodes in this list.
	uint numNodes;
	/// Root node bounding box minimum.
	float3 aabbRootMin;
	/// Root node bounding box maximum.
	float3 aabbRootMax;

	/// Number of elements (e.g. triangles) in each node (device memory).
	uint* d_numElems;
	/// \brief	Address of each nodes in #d_preorderTree.
	///
	///			Stored in the same order as #d_numElems or #d_childLeft. Associates these "unordered"
	///			arrays with the "ordered" array #d_preorderTree, so that we can modify the latter when
	///			only having data in "unordered" array, e.g. when setting custom bits.
	uint* d_nodeAddresses;
	/// \brief	Node extents (device memory).
	///
	///			\li \c xyz: Node center.
	///			\li \c w:	Node "radius", that is half the diagonal of the node's bounding box.
	float4* d_nodeExtent;
	/// \brief	Left child node indices (device memory). 
	/// 
	///			Stored due to the fact that we cannot get child information when iterating
	///			over the nodes instead of traversal using #d_preorderTree (e.g. for query radius estimation).
	///			Using #d_nodeAddresses, it might be possible to remove it. I still use it for convenience.
	uint* d_childLeft;
	/// \brief	Right child node indices (device memory). 
	/// 
	///			Stored due to the fact that we cannot get child information when iterating
	///			over the nodes instead of traversal using #d_preorderTree (e.g. for query radius estimation).
	///			Using #d_nodeAddresses, it might be possible to remove it. I still use it for convenience.
	uint* d_childRight;

	/// Size of the preorder tree representation #d_preorderTree in bytes.
	uint sizeTree;
	/// \brief	Preorder tree representation. 
	///
	///			The format description can be found in the detailed description of this structure.
	uint* d_preorderTree;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDFinalNodeList
///
/// \brief	Structure to hold all final nodes generated during kd-tree construction.
///
///			Separated from the default node list structure KDNodeList to save memory since only a 
///			subset of the node list data is required for the generation of the final kd-tree
///			layout. Note that this is a structure of arrays and not an array of structures.
///
/// \author	Mathias Neumann
/// \date	20.03.2010
/// \see	KDTreeData, KDNodeList
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDFinalNodeList
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(uint _maxNodes, uint _maxElems)
	///
	/// \brief	Initializes device memory. 
	///
	///			Provided maximum numbers should not be too low to avoid multiple resizes of
	///			the corresponding buffers.
	///
	/// \author	Mathias Neumann
	/// \date	20.03.2010
	///
	/// \param	_maxNodes	The initial maximum number of nodes. 
	/// \param	_maxElems	The initial maximum number of elements. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(uint _maxNodes, uint _maxElems);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void AppendList(KDNodeList* pList, bool appendENA, bool hasInheritedBounds)
	///
	/// \brief	Appends a node list to this final node list.
	///
	///			Copies data from given node list and resizes buffers if required.
	///
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \todo	Implement a way to avoid storing non-leaf ENA information. I suspended this since
	///			the conversion step would need to mark all elements as leaf elements. This would require
	///			either a chunk list (small chunks <= 64 !) or a lot of uncoalesced access... Anyways this
	///			would be a way to reduce memory requirements.
	///
	/// \param [in]	pList			The node list to append.
	/// \param	appendENA			Controls whether element node association (ENA) data should be
	///								appended.
	/// \param	hasInheritedBounds	Pass \c true to use inherited bounds as node bounds for this list.
	///								Else tight bounds are used. This was added as there are different
	///								bounds available at different stages of the construction.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void AppendList(KDNodeList* pList, bool appendENA, bool hasInheritedBounds);

	/// Returns whether this list is empty.
	bool IsEmpty() { return numNodes == 0; }
	/// Clears this list, but keeps requested memory.
	void Clear();
	/// Releases device memory.
	void Free();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ResizeNodeData(uint required)
	///
	/// \brief	Resize node related device memory.
	///
	///			To prevent frequently resizes, the new maximum #maxNodes is chosen to be at least
	///			twice as large as the previous #maxNodes.
	///	
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \see	::mncudaResizeMNCudaMem()
	///
	/// \param	required	The required number of node entries.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ResizeNodeData(uint required);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ResizeElementData(uint required)
	///
	/// \brief	Resize element related device memory.
	/// 		
	/// 		To prevent frequently resizes, the new maximum #maxElems is chosen to be at least
	/// 		twice as large as the previous #maxElems. 
	///
	/// \author	Mathias Neumann. 
	/// \date	April 2010. 
	/// \see	::mncudaResizeMNCudaMem()
	///
	/// \param	required	The required number of element entries. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ResizeElementData(uint required);

public:

#endif // __cplusplus

	/// Current number of nodes.
	uint numNodes;
	/// Maximum number of nodes that can be stored.
	uint maxNodes;
	/// Next free element position. Aligned to allow coalesced access.
	uint nextFreePos;
	/// Maximum number of elements that can be stored.
	uint maxElems;

	/// First element index address in ENA for each node (device memory).
	uint* d_idxFirstElem;
	/// Number of elements for each node (device memory).
	uint* d_numElems;
	/// Node bounds minimum for radius/center calculation (device memory). Might not be tight.
	float4 *d_aabbMin;
	/// Node bounds minimum for radius/center calculation (device memory). Might not be tight.
	float4 *d_aabbMax;
	/// Node levels (device memory). Starting with 0 for root.
	uint* d_nodeLevel;

	/// Split axis for each node (device memory).  Can be 0, 1 or 2. However, this is only valid for inner nodes.
	uint* d_splitAxis;
	/// Split position for each node (device memory). However, this is only valid for inner nodes.
	float* d_splitPos;
	/// Left (below) child node index for each node (device memory). Only valid for inner nodes.
	uint* d_childLeft;
	/// Right (above) child node index for each node (device memory). Only valid for inner nodes.
	uint* d_childRight;

	/// \brief	Element node association (ENA) list.
	///			
	///			Keeps track of which elements are assigned to each node. Element indices are stored
	///			contiguously for each node. The first element index address for node i is given by
	///			#d_idxFirstElem[i]. There can be holes between adjacent node element indices
	///			as the first element index address is aligned to improve performance.
	uint* d_elemNodeAssoc;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDNodeList
///
/// \brief	kd-tree node list data structure for intermediate node lists.
/// 		
/// 		This structure is used during kd-tree construction and is quite memory consuming. It
/// 		stores all required information all steps of the construction process. For node
/// 		bounds we distinguish tight and inherited bounds. Tight bounds result from computing
/// 		the exact AABB for all elements contained. Inherited bounds are generated when
/// 		splitting into child nodes. They are usually less tight. This distinction enables
/// 		empty space cutting. 
///
/// \author	Mathias Neumann
/// \date	16.02.2010 
///	\see	KDTreeData, KDFinalNodeList
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDNodeList
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(uint _maxNodes, uint _maxElems, uint _numElementPoints = 2)
	///
	/// \brief	Initializes device memory.
	/// 		
	/// 		Provided maximum numbers should not be too low to avoid multiple resizes of the
	/// 		corresponding buffers. 
	///
	/// \author	Mathias Neumann. 
	/// \date	16.02.2010. 
	///
	/// \param	_maxNodes			The initial maximum number of nodes. 
	/// \param	_maxElems			The initial maximum number of elements. 
	/// \param	_numElementPoints	Number of element points. See KDTreeGPU for a description of
	///								this parameter.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(uint _maxNodes, uint _maxElems, uint _numElementPoints = 2);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void AppendList(KDNodeList* pList, bool appendENA)
	///
	/// \brief	Appends other node list to this node list.
	/// 		
	/// 		Copies data from given node list and resizes buffers if required. 
	///
	/// \author	Mathias Neumann. 
	/// \date	April 2010.
	///
	/// \param [in]		pList	The node list to append. 
	/// \param	appendENA		Controls whether element node association (ENA) data should be
	/// 						appended. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void AppendList(KDNodeList* pList, bool appendENA);

	/// Returns whether this list is empty.
	bool IsEmpty() { return numNodes == 0; }
	/// Clears this list, but keeps requested memory.
	void Clear();
	/// Releases device memory.
	void Free();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ResizeNodeData(uint required)
	///
	/// \brief	Resize node related device memory.
	///
	///			To prevent frequently resizes, the new maximum #maxNodes is chosen to be at least
	///			twice as large as the previous #maxNodes.
	///	
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \see	::mncudaResizeMNCudaMem()
	///
	/// \param	required	The required number of node entries.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ResizeNodeData(uint required);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void ResizeElementData(uint required)
	///
	/// \brief	Resize element related device memory.
	/// 		
	/// 		To prevent frequently resizes, the new maximum #maxElems is chosen to be at least
	/// 		twice as large as the previous #maxElems. 
	///
	/// \author	Mathias Neumann. 
	/// \date	April 2010. 
	/// \see	::mncudaResizeMNCudaMem()
	///
	/// \param	required	The required number of element entries. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void ResizeElementData(uint required);

public:

#endif // __cplusplus

	/// Number of nodes in this list.
	uint numNodes;
	/// Maximum number of nodes that can be stored.
	uint maxNodes;
	/// Next free spot for new element data in ENA list. Aligned to allow coalesced access.
	uint nextFreePos;
	/// Maximum number of elements that can be stored.
	uint maxElems;
	/// Number of element specific points in this list. Can be 1 or 2. See KDTreeGPU. 
	uint numElementPoints;

	/// First element index address in ENA for each node (device memory).
	uint* d_idxFirstElem;
	/// Number of elements for each node (device memory).
	uint* d_numElems;
	/// Tight AABB minimum for each node (device memory).
	float4* d_aabbMinTight;
	/// Tight AABB maximum for each node (device memory).
	float4* d_aabbMaxTight;
	/// Inherited AABB minimum for each node (device memory).
	float4* d_aabbMinInherit;
	/// Inherited AABB maximum for each node (device memory).
	float4* d_aabbMaxInherit;
	/// Node levels (device memory). Starting with 0 for root.
	uint* d_nodeLevel;

	// Split information

	/// Split axis for each node (device memory). Can be 0, 1 or 2. Only valid after splitting.
	uint* d_splitAxis;
	/// Split position for each node (device memory). Only valid after splitting.
	float* d_splitPos;
	/// Left child node index for each node (device memory). Only valid after splitting.
	uint* d_childLeft;
	/// Right child node index for each node (device memory). Only valid after splitting.
	uint* d_childRight;

	// Element information

	/// \brief	Element node association (ENA) list.
	///			
	///			Keeps track of which elements are assigned to each node. Element indices are stored
	///			contiguously for each node. The first element index address for node i is given by
	///			#d_idxFirstElem[i]. There can be holes between adjacent node element indices
	///			as the first element index address is aligned to improve performance.
	///
	/// \note	This list usually contains a single element (e.g. triangle) multiple times.
	uint* d_elemNodeAssoc;
	/// \brief	Element specific points 1.  Same order as ENA.
	///
	///			Allowed are at most two points per element. E.g. for triangle kd-trees, this would
	///			be the minimum of the triangle's bounding box.
	/// \see	#numElementPoints, KDTreeGPU
	float4* d_elemPoint1;
	/// \brief	Element specific points 2. Same order as ENA.
	///
	///			Allowed are at most two points per element. E.g. for triangle kd-trees, this would
	///			be the maximum of the triangle's bounding box.
	/// \see	#numElementPoints, KDTreeGPU
	float4* d_elemPoint2;


	// Small node only information. This is *not* appended to final node list and
	// only valid in small node stage.

	/// \brief	Small root node index in small node list for each node.
	///
	///			Used to remember the corresponding small root parent for a given
	///			small node. The small root parent is the original small node list
	///			node. For each such node we precompute element masks and possible
	///			splits. Using this array we can combine that information with
	///			node specific data. See #d_elemMask.
	uint* d_idxSmallRoot;
	/// \brief	Element set stored as bit mask for each node (relative to small root node).
	///
	///			For each small root node, that is for each initial small node, this
	///			element mask is precomputed by mapping all node elements to the bits of
	///			the bit mask. As the number of elements per small node is restricted by
	///			::KD_SMALLNODEMAX, this mapping is possible. So initially, all #d_numElems[i]
	///			bits starting from bit 0 are set for the i-th small node.
	///
	///			For subsequent small nodes, that is for each direct or indirect child j of an initial
	///			small node i, #d_elemMask[j] is relative to the corresponding #d_elemMask[i] of the
	///			initial small node list. So if any of the bits set in #d_elemMask[i] is unset in
	///			#d_elemMask[j], the corresponding element is not contained in the child j.
	ElementMask* d_elemMask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDChunkList
///
/// \brief	Chunk list representation used for kd-tree construction.
/// 		
/// 		This list is used for node bounding box computation and node element counting.
///			The chunks partition the elements associated to the nodes of a given node list.
///			A chunk can only be assigned to a single node. Each node might be represented
///			by multiple chunks, depending on the element count of the node. That's because
///			each chunk can contain at most ::KD_CHUNKSIZE elements.
///
///			The chunks are mapped to thread blocks of CUDA threads. At the beginning,
///			chunk information like node index and first element index address are read
///			into shared memory. Then all threads can read this data and use it to
///			process it's element, e.g. for element AABB computation.
///
/// \todo	Add resize mechanism that resizes the buffers when there are too many chunks. Also
///			improve this class by adding a better relationship to the node list represented.
///
/// \author	Mathias Neumann
/// \date	16.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDChunkList
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(uint _maxChunks)
	///
	/// \brief	Initializes device memory. 
	///
	/// \author	Mathias Neumann.
	/// \date	16.02.2010.
	///
	/// \param	_maxChunks	The maximum number of chunks. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(uint _maxChunks);
	/// Returns whether this list is empty.
	bool IsEmpty() { return numChunks == 0; }
	/// Clears this list, but keeps requested memory.
	void Clear();
	/// Releases device memory.
	void Free();

#endif // __cplusplus

	/// Number of chunks in this list.
	uint numChunks;
	/// Maximum number of chunks that can be stored.
	uint maxChunks;

	/// Index of the associated node in the corresponding node list (device memory).
	uint* d_idxNode;
	/// Address of the first element index in the corresponding list's ENA list (device memory).
	uint* d_idxFirstElem;
	/// Number of elements in each chunk (device memory).
	uint* d_numElems;
	/// Chunk AABB minimum coordinates (device memory).
	float4* d_aabbMin;
	/// Chunk AABB maximum coordinates (device memory).
	float4* d_aabbMax;
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	KDSplitList
///
/// \brief	Stores split information used in small node stage.
/// 		
/// 		Specifically all split candidates for the small root nodes are stored. Split axis is
/// 		not stored. Instead it is calculated from the split index. This is possible as we
/// 		know the order the splits are stored. Note that the number of small root nodes is not
/// 		stored explicitly as it is equivalent to the number of nodes in the small node list. 
///
///			For each split i we precalculate element masks #d_maskLeft[i] and #d_maskRight[i] that
///			show which elements of the corresponding small root node would get into the left and
///			right child respectively. These element masks can be combined with the element mask
///			of the small root node (or more general: a small parent node) using a Booealn AND
///			operation to get the element masks for the child nodes.
///
/// \author	Mathias Neumann
/// \date	19.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct KDSplitList
{
#ifdef __cplusplus
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Initialize(KDNodeList* pSmallRoots)
	///
	/// \brief	Initializes device memory. 
	///
	/// \author	Mathias Neumann.
	/// \date	19.02.2010.
	///
	/// \param [in]		pSmallRoots	List of small root nodes. These is the list of small node just
	/// 							after finishing the large node stage. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(KDNodeList* pSmallRoots);
	/// Releases device memory.
	void Free();

#endif // __cplusplus

	/// First index of split data in split list each small root node (device memory).
	uint* d_idxFirstSplit;
	/// Number of split positions for each small root node (device memory).
	uint* d_numSplits;

	// These arrays store splits contiguously for all small nodes.

	/// Split position for each split (device memory).
	float* d_splitPos;
	/// \brief	Split information for each split (device memory).
	///
	///			The tree least significant bits are used. Bits 0 and 1 store the split
	///			axis and bit 3 stores whether this is a minimum or maximum split. The
	///			distinction is important for non-degenerating element AABBs, e.g. for
	///			triangle kd-trees. It determines on which side the split triangle is
	///			placed.
	/// \see	::kernel_InitSplitMasks()
	uint* d_splitInfo;
	/// \brief	Element set on the left of the splitting plane for each split (device memory).
	///
	///			A set bit signalizes that the corresponding element would get into the left
	///			child node when splitting according to this split.
	ElementMask* d_maskLeft;
	/// \brief	Element set on the right of the splitting plane for each split (device memory).
	///
	///			A set bit signalizes that the corresponding element would get into the right
	///			child node when splitting according to this split.
	ElementMask* d_maskRight;
};


#endif // __MN_KDKERNELDEFS_H__