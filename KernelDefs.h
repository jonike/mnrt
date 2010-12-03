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
/// \file	MNRT\KernelDefs.h
///
/// \brief	Declares appropriate structures to pass data to CUDA kernels.
/// \author	Mathias Neumann
/// \date	31.01.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	globalillum	GPU-based Global Illumination
/// 
/// \brief	GPU-based components of MNRT for global illumination.
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_KERNELDEFS_H__
#define __MN_KERNELDEFS_H__

#pragma once

#include <cuda_runtime.h> // for cudaArray
#include <vector_types.h>
#include <vector>
#include <string>

class BasicScene;

/// \brief	Unsigned 32-bit integer type.
///
///			It is important that this is 32-bit wide as some operations, e.g. CUDPP primitives, do
///			not support wider types.
typedef unsigned __int32 uint;
/// Unsigned char type.
typedef unsigned char uchar;


/// Maximum number of materials allowed. The number is restricted to ensure constant structure size for
/// GPU's constant memory.
#define MAX_MATERIALS			64
/// Number of texture types. It is restricted by material flag array. See MaterialProperties.
#define NUM_TEX_TYPES			2


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \enum	LightType
///
/// \brief	Light types supported by MNRT.
///
///			As MNRT uses a GPU-based implementation, support for area light sources with custom
///			shapes to define the area is not available. This is basically a simplification to
///			avoid searching for parallel techniques to support custom shapes.
///
/// \author	Mathias Neumann
/// \date	31.01.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
enum LightType
{
	/// Point light source. Emitting uniformly in all directions of the surrounding sphere.
	Light_Point = 0,
	/// Directional light source. Placed infinitely far away from the scene and emits in a single direction.
	Light_Directional = 1,
	/// Area light source with disc as area. The disc is defined using the light's position as center,
	/// the light's direction and a radius.
	Light_AreaDisc = 2,
	/// Area light source with rectangle as area. The rectangle is defined using the light's position
	/// as one rectangle vertex and two edge vectors for the adjacent edges.
	Light_AreaRect = 3
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \enum	TextureType
///
/// \brief	Texture types supported by MNRT.
///
/// \author	Mathias Neumann
/// \date	03.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
enum TextureType
{
	/// Diffuse material texture. Can be used instead of the material's diffuse color.
	Tex_Diffuse = 0,
	/// Bump map (height map) to provide more geometric detail by adapting the surface normal
	/// according to the bump map. Currently this texture type support is not that sophisticated
	/// as ray differentials are not tracked in MNRT. These would be required to determine correct
	/// offsets for fetching adjacent texels from the bump map.
	Tex_Bump = 1
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	TextureHost
///
/// \brief	Temporary structure for texture information read into host memory.
///
///			Used to transfer textures into device memory.
///
/// \author	Mathias Neumann
/// \date	03.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct TextureHost
{
	/// Host memory texture data. \c float array for bump maps and \c uchar4 array (RGBA) for diffuse textures.
	void* h_texture;
	/// Two-dimensional size (width, height).
	uint2 size;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	ShadingPoints
///
/// \brief	Structure to hold the shading points determined during a ray tracing pass.
///
///			The ray \em tracing kernel fills this structure with intersection information that
///			describes the found hits for rays traced. Not all rays have to hit something. Therefore
///			the #d_idxTris contains -1 for rays that hit nothing.
///
///			All data is stored in global GPU memory since we generate the rays on the GPU. A
///			structure of arrays (SoA) is used instead of an array of structures (AoS) to allow
///			coalesced memory access. Each thread handles a single ray. The global thread index
///			\c tid is the index of the shading point to write. Hence the shading point arrays
///			have to be at least as large as the RayChunk that is traced.
///
/// \note	The member functions are only for the C++ CPU side. Device code cannot use them.
///
/// \author	Mathias Neumann
/// \date	31.01.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct ShadingPoints
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint _maxPoints)
	///
	/// \brief	Initializes the shading point structure.
	///
	///			Requests device memory of the given maximum number of shading points.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	///
	/// \param	_maxPoints	The maximum number of shading points to store. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint _maxPoints);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Clear()
	/// 
	/// \brief	Sets the number of shading points to zero.
	/// 
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Clear() { numPoints = 0; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	/// 
	/// \brief	Frees allocated memory.
	/// 
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void CompactSrcAddr(uint* d_srcAddr, uint countNew)
	///
	/// \brief	Compacts this shading point structure using the given source address array.
	///
	///			This operation assumes that the source addresses were generated before, e.g. using
	///			::mncudaGenCompactAddresses(). The latter also returns the required new number of
	///			shading points. Basically, this was done to allow compacting multiple structures
	///			using the same source addresses.
	/// 
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \see	::mncudaGenCompactAddresses(), PhotonData::CompactSrcAddr()
	///
	/// \param [in]	d_srcAddr		The source addresses (device memory). \a d_srcAddr[i] defines at
	///								which original index the new value for the i-th shading point 
	///								can be found.
	/// \param	countNew			The new number of shading points. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompactSrcAddr(uint* d_srcAddr, uint countNew);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Add(const ShadingPoints& other)
	///
	/// \brief	Adds other shading point list to this list.
	///
	/// \author	Mathias Neumann
	/// \date	October 2010
	///
	/// \param	other	Shading points to add.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Add(const ShadingPoints& other);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint Merge(const ShadingPoints& other, uint* d_isValid)
	///
	/// \brief	Merges given shading point data into this shading point data. 
	///
	///			The merge process is controlled using the given \a d_isValid "binary" array. If
	///			\a d_isValid[i] = 1, then the i-th element of \a other is merged into this object.
	///			Else, if \a d_isValid[i] = 0, the i-th element of \a other is ignored.
	/// 
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \see	::mncudaGenCompactAddresses(), ::mncudaSetFromAddress()
	///
	/// \param	other				Shading points to merge into this object.
	/// \param [in]		d_isValid	"Binary" device array that contains as many elements as \a other
	///								has shading points. Element values have to be 0 and 1.
	///
	/// \return	Returns number of merged points.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint Merge(const ShadingPoints& other, uint* d_isValid);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetFrom(const ShadingPoints& other, uint* d_srcAddr, uint numSrcAddr)
	///
	/// \brief	Sets this shading points based on other shading points.
	///
	///			The process of copying the \a other shading points into this object is guided by
	///			the source addresses \a d_srcAddr in the following way:
	///
	///			\code d_pixels[i] = other.d_pixels[d_srcAddr[i]]; // Same for other members \endcode
	/// 
	/// \author	Mathias Neumann
	/// \date	July 2010
	/// \see	::mncudaSetFromAddress(), CompactSrcAddr()
	///
	/// \param	other				Shading points to set this object from.
	/// \param [in]	d_srcAddr		The source addresses (device memory). \a d_srcAddr[i] defines at
	///								which original index the new value for the i-th shading point 
	///								can be found.
	/// \param	numSrcAddr			Number of source address. This is also the new number of shading
	///								points in this object.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetFrom(const ShadingPoints& other, uint* d_srcAddr, uint numSrcAddr);

public:
#endif // __cplusplus

	/// Number of shading points stored.
	uint numPoints;
	/// Maximum number of shading points that can be stored.
	uint maxPoints;
	/// Index of the corresponding pixel (device array).
	uint* d_pixels;
	/// Index of the intersected triangle or -1, if no intersection (device array).
	int* d_idxTris;
	/// Point of intersection coordinates (device array).
	float4* d_ptInter;
	/// Geometric normal at intersection point (device array).
	float4* d_normalsG;
	/// Shading normal at intersection point (device array).
	float4* d_normalsS;
	/// Barycentric hit coordinates (device array).
	float2* d_baryHit;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	LightData
///
/// \brief	Holds information about the primary light source for use in kernels.
///
///			This structure is loaded into GPU's constant memory. Therefore no host/device memory 
///			pointers are allowed. Currently I only allow one single light source for simplicity.
///			Extension to more light sources would be possible by allowing a maximum number of
///			light sources, just as with the MaterialProperties structure.
///
/// \author	Mathias Neumann
/// \date	31.01.2010
/// \see	LightType, BasicLight
////////////////////////////////////////////////////////////////////////////////////////////////////
struct LightData
{
	/// Type of light source.
	LightType type;
	/// Light source position. Invalid for directional light sources.
	float3 position;
	/// Light direction. Invalid for point light sources.
	float3 direction;
	/// Emitted radiance of light source. This is the intensity for point lights.
	float3 L_emit;
	/// Vector that spans first side of the rectangle for rectangle area lights.
	float3 areaV1;
	/// Vector that spans second side of the rectangle for rectangle area lights.
	float3 areaV2;
	/// Area light radius for disc area lights.
	float areaRadius;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	MaterialProperties
///
/// \brief	Material properties structure for use in kernels.
///
///			This structure represents the material data for CUDA kernels. It is copied into
///			GPU's constant memory and therefore has to have a constant size. Hence there is an
///			limit (::MAX_MATERIALS) for the total number of materials allowed.
///
/// \author	Mathias Neumann
/// \date	31.01.2010
/// \see	BasicMaterial
///
/// \todo	Check if area light material flag is correcly used.
///
////////////////////////////////////////////////////////////////////////////////////////////////////
struct MaterialProperties
{
	/// \brief	Texture flags. 
	///
	///			Bits as follows:
	///			\li \c x: Diffuse texture index. -1 if no texture.
	///			\li \c y: Bump map texture index. -1 if no texture.
	///			\li \c w0: Area light flag (bit 0). Set to 1 for an area light material.
	char4 flags[MAX_MATERIALS];
	/// Diffuse material color.
	float3 clrDiff[MAX_MATERIALS];
	/// Specular material color.
	float3 clrSpec[MAX_MATERIALS];
	/// Specular exponent (shininess).
	float specExp[MAX_MATERIALS];
	/// Transparency alpha (opaque = 1, completely transparent = 0).
	float transAlpha[MAX_MATERIALS];
	/// Index of refraction (1 = vacuum; 1.333 = water; 1.5-1.6 = glass;...).
	float indexRefrac[MAX_MATERIALS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MaterialData
///
/// \brief	Manages material data and loads texture images.
///
///			Creates the MaterialProperties structure from a given BasicScene and loads all
///			diffuse and bump map textures from specified files. Textures are internally stored
///			as \c cudaArray objects to benefit from improved cache performance for 2D locality.
///
/// \author	Mathias Neumann
/// \date	March 2010
/// \see	MaterialProperties
///
/// \todo	Hide public members as this is no longer a struct for kernels.
////////////////////////////////////////////////////////////////////////////////////////////////////
class MaterialData
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(BasicScene* pScene)
	///
	/// \brief	Initialization from BasicScene object.
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	///
	/// \param [in]	pScene	Scene to load the materials for.
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(BasicScene* pScene);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Releases requested memory.
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static void* LoadImageFromFile(const std::string& strImage, bool isBumpMap,
	/// 	uint2* outImgSize)
	///
	/// \brief	Loads image from file using DevIL.
	/// 		
	/// 		This method uses the Developer's Image Library (DevIL) to load and convert images
	/// 		from lots of file formats. Check http://openil.sourceforge.net/ for more information
	/// 		about DevIL. 
	///
	/// \author	Mathias Neumann
	/// \date	03.04.2010
	///
	/// \param	strImage			The image file path. 
	/// \param	isBumpMap			\c true if is a bump map texture. 
	/// \param [out]	outImgSize	In this parameter the image size is stored. 
	///
	/// \return	Returns the loaded image in host memory. This is an \c float array for bump maps and
	/// 		a \c uchar4 array (RGBA) for diffuse textures. Might return \c NULL in case loading
	/// 		failed. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void* LoadImageFromFile(const std::string& strImage, bool isBumpMap, uint2* outImgSize);

private:
	void LoadTextures(BasicScene* pScene, uint texType, std::vector<TextureHost>* outHostTextures);

public:
	/// Number of materials. Maximum is ::MAX_MATERIALS.
	uint numMaterials;
	/// Material properties for GPU's constant memory.
	MaterialProperties matProps;
	/// Textures as \c cudaArray objects. One vector for each texture type.
	std::vector<cudaArray*> vecTexArrays[NUM_TEX_TYPES];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	TriangleData
///
/// \brief	Triangle data structure for kernel use.
/// 		
/// 		Holds triangle data from a BasicScene in form of a structure of arrays for
/// 		coalescing. \c float4 is used instead of \c float3 to ensure alignment and improve
/// 		performance.
/// 		
/// \note	It seems that the texture fetcher can only fetch from a texture once per thread
/// 		without using too many registers (observed on a GTS 250, CUDA 2.3). Also there seems
/// 		to be a limit of fetches that can be done without using too many registers and having
/// 		the performance dropping. Also note that we use linear device memory instead of cuda
/// 		arrays since cuda arrays can only hold up to 8k elements when one dimensional, while
/// 		linear device memory can have up to 2^27 elements (observed on a GTS 250). 
///
/// \todo	Add way to handle dynamic geometry. 
///
/// \author	Mathias Neumann
/// \date	31.01.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct TriangleData
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(BasicScene* pScene)
	///
	/// \brief	Initializes this object from a given scene.
	///
	///			Requests all required memory so that all triangle information can be stored in
	///			global memory.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	///
	/// \param [in]	pScene	The scene.
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(BasicScene* pScene);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Releases requested memory.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

private:
	// Allocates auxiliary memory (e.g. for bounding boxes).
	bool InitAuxillary();

public:
#endif // __cplusplus

	/// Number of triangles.
	uint numTris;
	/// Scene AABB minimum. Stored for kd-tree initial node.
	float3 aabbMin;
	/// Scene AABB maximum. Stored for kd-tree initial node.
	float3 aabbMax;
	/// Vertices, 3 per triangle (device memory).
	float4 *d_verts[3];
	/// Normals, 3 per triangle (device memory).
	float4* d_normals[3];
	/// Material indices, 1 per triangle (device memory).
	uint* d_idxMaterial;
	/// Texture coordinates (UV), 3 per triangle (device memory).
	float2* d_texCoords[3];
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	PhotonData
///
/// \brief	GPU-based photon array representation.
///
///			Used to represent photons on the GPU-side, e.g. for photon spawning. Furthermore this
///			structure is used for the final photon array. The photons are stored as structure of
///			arrays to improve memory performance. \c float4 is used for both position and power
///			to achieve some compression and to reduce the number of texture fetches within kernels.
///			
///			The structure has a maximum for storable photons, so that we can only work on a limited
///			number of photons within one kernel. This is unproblematic in most cases: Usually one
///			photon is assigned to one thread, and the thread number cannot change during kernel
///			execution.
///
/// \author	Mathias Neumann
/// \date	07.04.2010
/// \see	PhotonMap
///
/// \todo	Increase compression, see e.g. \ref lit_jensen "[Jensen 2001]".
////////////////////////////////////////////////////////////////////////////////////////////////////
struct PhotonData
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint _maxPhotons)
	///
	/// \brief	Initializes this object by requesting device memory.
	///
	/// \author	Mathias Neumann
	/// \date	07.04.2010
	///
	/// \param	_maxPhotons	The maximum number of photons that should be stored. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint _maxPhotons);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Clear()
	///
	/// \author	Mathias Neumann
	/// \date	07.04.2010
	///
	/// \brief	Clears this data structure by resetting the photon count to zero.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Clear();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \author	Mathias Neumann
	/// \date	07.04.2010
	///
	/// \brief	Releases device memory.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void CompactSrcAddr(uint* d_srcAddr, uint countNew)
	///
	/// \brief	Compacts this photon structure using the given source address array.
	///
	///			See ShadingPoints::CompactSrcAddr() for more information.
	///
	/// \author	Mathias Neumann
	/// \date	07.04.2010
	///	\see	ShadingPoints::CompactSrcAddr(), ::mncudaGenCompactAddresses()
	///
	/// \param [in]	d_srcAddr		The source addresses (device memory). \a d_srcAddr[i] defines at
	///								which original index the new value for the i-th element
	///								can be found.
	/// \param	countNew			The new number of photons.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompactSrcAddr(uint* d_srcAddr, uint countNew); 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint Merge(const PhotonData& other, uint* d_isValid)
	///
	/// \brief	Merges given photon data into this photon data. 
	///
	///			The merge process is controlled using the given \a d_isValid "binary" array. If
	///			\a d_isValid[i] = 1, then the i-th element of \a other is merged into this object.
	///			Else, if \a d_isValid[i] = 0, the i-th element of \a other is ignored.
	///
	/// \author	Mathias Neumann
	/// \date	July 2010
	///	\see	ShadingPoints::Merge(), ::mncudaGenCompactAddresses()
	///
	/// \param	other				Photons to merge into this object.
	/// \param [in]		d_isValid	"Binary" device array that contains as many elements as \a other
	///								has photons. Element values have to be 0 and 1. 
	///
	/// \return	Returns number of merged photons.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint Merge(const PhotonData& other, uint* d_isValid);

public:
#endif // __cplusplus

	/// Number of photons.
	uint numPhotons;
	/// Maximum number of photons.
	uint maxPhotons;

	/// \brief	3D positions of the stored photons and azimuthal angle.
	///
	///			\li \c xyz: Position of photon.
	///			\li \c w: Azimuthal angle phi (spherical coordinates first angle) of indicent direction (radians).
	float4* d_positions;
	/// \brief	Transported power (flux) for each photon and polar angle. 
	///
	///			\li \c xyz: Power, stored for R, G and B color bands.
	///			\li \c w: Polar angle theta (spherical coordinates second angle) of indicent direction (radians).
	float4* d_powers;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	PairList
///
/// \brief	Stores pairs of unsigned 32-bit integers and allows sorting them by the first pair
/// 		component.
/// 		
/// 		The sorting by first component can be usefull to organize the pairs in a segmented
/// 		reduction compatible way. Once done, pairs with identical first pair element lie
/// 		contiguously. Hence on all elements with identical first element a reduction can be
/// 		performed by calling ::mncudaSegmentedReduce with first elements as owner array and
/// 		second elements as data array. 
///
/// \author	Mathias Neumann
/// \date	15.04.2010
/// \see	::mncudaSegmentedReduce
////////////////////////////////////////////////////////////////////////////////////////////////////
struct PairList
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint _maxPairs)
	///
	/// \brief	Initializes this object by requesting device memory. 
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	///
	/// \param	_maxPairs	The maximum number of pairs to store. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint _maxPairs);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Clear()
	///
	/// \brief	Resets the pair number to zero.
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Clear() { numPairs = 0; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Releases device memory.
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SortByFirst(uint firstValueMax, uint sortSegmentSize = 0, uint* d_outSrcAddr = NULL)
	///
	/// \brief	Sorts the pair list by first element values.
	/// 		
	/// 		Pass maximum for first value so we can compute the number of least significant bits
	/// 		we use for radix sort.
	///
	/// \todo	Check why \c cudppSort has problems with sorting arrays starting from
	///			unaligned addresses. Currently I avoid this by copying the input data into
	///			temporary buffers before sorting.
	///
	/// \author	Mathias Neumann. 
	/// \date	15.04.2010. 
	///
	/// \param	firstValueMax			The first value maximum. Can be used to restrict number of
	/// 								significant bits for radix sort. 
	/// \param	sortSegmentSize			Size of a sort segment. If non-zero, this parameter divides
	/// 								sorting into chunks of size \a sortSegmentSize that are
	/// 								sorted individually. 
	/// \param [out]	d_outSrcAddr	If not \c NULL, the source addresses to sort further arrays
	/// 								(exaclty as the second element values were sorted) is
	/// 								returned. In this case, the array should have #numPairs
	/// 								elements. Further sorting can then be done using the
	///									::mncudaSetFromAddress() method.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SortByFirst(uint firstValueMax, uint sortSegmentSize = 0, uint* d_outSrcAddr = NULL);
public:
#endif // __cplusplus

	/// Number of pairs.
	uint numPairs;
	/// Maximum number of pairs.
	uint maxPairs;

	/// First pair values.
	uint* d_first;
	/// Second pair values.
	uint* d_second;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	QuadTreeSP
///
/// \brief	Quadtree used to organize shading points in screen space. For use within kernels.
///
///			This is used for adaptive sample seeding using the geometric variation of the quadtree
///			nodes. A shading point can be classified using it's pixel coordinate in screen space.
///			So we can, on each tree level, find exactly one quadtree node for the shading point.
///			Using segmented reduction we can compute the average position, normal and geometric
///			variation of a quadtree node center. The geometric variation is given as sum of
///			the variations between the averaged node center and all quadtree node shading points.
///
///			The quadtree is constructed down to the pixel level where each quadtree node
///			represents exactly one screen pixel. Note that this structure does \em not record
///			the association between shading points and quadtree nodes. It only provides the frame
///			for the tree.
///
///			\par Order of Nodes
///			Kernels constructing the quad tree should create the following order of nodes: Nodes
///			of one tree level are stored contiguously, and all nodes of level i appear before
///			all deeper nodes of level i+1. Hence the root node is in element 0 and its children are
///			in elements 1-4. The child in 1 is the top-left child, 2 is the top-right child, 3
///			the bottom-left and 4 the bottom-right child. The order of the children of these 
///			children stays the same, i.e 5-8 will contain the children of element 1, 9-12 those
///			of element 2 and so forth. Keeping this order is important as many kernels rely on it.
///
/// \author	Mathias Neumann
/// \date	16.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct QuadTreeSP
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint sizeScreen)
	///
	/// \brief	Initialization from given screen size.
	/// 		
	/// 		Requests required memory to store a quadtree for the given screen size. 
	///
	/// \author	Mathias Neumann
	/// \date	16.04.2010
	///
	/// \param	sizeScreen	The size of the screen. Has to be power of 2. It is assumed that the
	/// 					screen is quadratic, e.g. 512 times 512 pixels. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint sizeScreen);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Releases all device memory and therefore destroys the quadtree.
	///
	/// \author	Mathias Neumann
	/// \date	16.04.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint GetLevel(uint idxNode) const
	///
	/// \brief	Gets the level for the given node index.
	///
	///			Avoids storing the level for each node.
	///
	/// \author	Mathias Neumann
	/// \date	16.04.2010
	///
	/// \param	idxNode	The node index. 
	///
	/// \return	The level. Level starts from 0 for root and goes to log(sizeScreen) for leafs.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint GetLevel(uint idxNode) const;

public:
#endif // __cplusplus

	/// Number of quadtree nodes.
	uint numNodes;
	/// Maximum number of nodes.
	uint maxNodes;
	/// Number of levels of the constructed tree.
	uint numLevels;

	/// Average position for nodes.
	float4* d_positions;
	/// Average normal for nodes. Might not always be normalized.
	float4* d_normals;
	/// Geometric variation for nodes. Defined as the sum of the geometric variation to all contained shading points.
	float* d_geoVars;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	ClusterList
///
/// \brief	List of clusters for use within kernels.
/// 		
/// 		Keeps track of cluster information, mainly position and normal. Used during adaptive
/// 		sample seeding. For convenience, the arrays have one additional entry that can be
///			used for a virtual cluster, e.g. a cluster of unclassified points. It is \e not
///			taken into account by all methods of this structure besides Initialize(), which
///			allocates one more entry than #maxClusters.
///
/// \author	Mathias Neumann
/// \date	17.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct ClusterList
{
#ifdef __cplusplus

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint _maxClusters)
	///
	/// \brief	Initializes the list by requesting device memory.
	///
	/// \author	Mathias Neumann
	/// \date	17.04.2010
	///
	/// \param	_maxClusters	The maximum number of clusters to store. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint _maxClusters);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Clear()
	///
	/// \brief	Resets the clusters number to zero.
	///
	/// \author	Mathias Neumann
	/// \date	17.04.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Clear() { numClusters = 0; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void CompactSrcAddr(uint* d_srcAddr, uint countNew)
	///
	/// \brief	Compacts this cluster list using the given source address array.
	///
	///			This operation assumes that the source addresses were generated before, e.g. using
	///			::mncudaGenCompactAddresses(). The latter also returns the required new number of
	///			new clusters. Basically, this was done to allow compacting multiple structures
	///			using the same source addresses.
	/// 
	/// \author	Mathias Neumann
	/// \date	April 2010
	/// \see	::mncudaGenCompactAddresses(), PhotonData::CompactSrcAddr()
	///
	/// \param [in]	d_srcAddr		The source addresses (device memory). \a d_srcAddr[i] defines at
	///								which original index the new value for the i-th cluster
	///								can be found.
	/// \param	countNew			The new number of clusters. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompactSrcAddr(uint* d_srcAddr, uint countNew);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Add(const ClusterList& other)
	///
	/// \brief	Adds other cluster list to this list. 
	///
	/// \author	Mathias Neumann. 
	/// \date	October 2010. 
	///
	/// \param	other	Clusters to add. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Add(const ClusterList& other);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint Merge(const ClusterList& other, uint* d_isValid)
	///
	/// \brief	Merges given clusters into this cluster list.
	/// 		
	/// 		The merge process is controlled using the given \a d_isValid "binary" array. If \a
	/// 		d_isValid[i] = 1, then the i-th element of \a other is merged into this object. Else,
	/// 		if \a d_isValid[i] = 0, the i-th element of \a other is ignored. 
	///
	/// \author	Mathias Neumann. 
	/// \date	April 2010 \see	::mncudaGenCompactAddresses(), ::mncudaSetFromAddress() 
	///
	/// \param	other				Clusters to merge into this object. 
	/// \param [in]		d_isValid	"Binary" device array that contains as many elements as \a other
	/// 							has clusters. Element values have to be 0 and 1. 
	///
	/// \return	Returns number of merged clusters. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint Merge(const ClusterList& other, uint* d_isValid);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Destroy()
	///
	/// \brief	Releases all device memory and therefore destroys the list.
	///
	/// \author	Mathias Neumann
	/// \date	17.04.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Destroy();

public:
#endif // __cplusplus

	/// \brief	Number of clusters.
	///
	///			This does not include the virtual cluster at the end of the cluster list.
	uint numClusters;
	/// \brief	Maximum number of clusters.
	///
	///			This does not include the virtual cluster at the end of the cluster list.
	uint maxClusters;

	/// Cluster center positions.
	float4* d_positions;
	/// Cluster center normals.
	float4* d_normals;
	/// \brief	Stores which shading point is used as cluster center (index). 
	///
	///			Only valid after final irradiance samples are generated.
	uint* d_idxShadingPt;
	/// \brief	Maximum geometric variation of all shading points assigned to a cluster (for each cluster).
	///
	///			Stored from generation of last frame. Will be used to classify new frame's shading points
	///			according to old clusters to retain some of the latter for temporal coherence. Only valid
	///			after final irradiance samples are generated.
	float* d_geoVarMax;
};


#endif // __MN_KERNELDEFS_H__