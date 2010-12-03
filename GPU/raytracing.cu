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
/// \file	GPU\raytracing.cu
///
/// \brief	Kernels for ray tracing.
///
/// \todo	Evaluate shared memory usage and compability with older GPUs.
///
/// \author	Mathias Neumann
/// \date	30.01.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"
#include "kd-tree/KDKernelDefs.h"
#include "RayPool.h"
#include "MNCudaUtil.h"
#include "MNCudaMemPool.h"
#include "MNCudaMT.h"
#include "MNStatContainer.h"

#include "intersect_dev.h"
#include "photon_dev.h"
#include "sample_dev.h"

/// \brief	Thread block size used for intersection search.
///	
///			For kernels calling dev_FindNextIntersectionKDWhileWhile().
#define INTERSECT_BLOCKSIZE	128

/// \brief	Maximum number of diffuse textures supported.
///
///			The number of diffuse textures has to be limited to a small amount due to the problem that
///			arrays of textures are not possible, yet.
///
/// \todo	Right now, arrays of texture references are not possible in CUDA. Hence I currently
///			use many single texture references to reach diffuse or bump map texture support in
///			MNRT. This led to quite a lot of redundant code. I tried to reduce this using macros,
///			but that only worked to some extent. Accordingly it would be relieving to find
///			a better solution for texture memory variables for real textures (e.g. bump maps).
///			As discussed for ::tex_diffTex0, the solution of moving the real textures into
///			slices of a 3D CUDA array worked, but also had several major drawbacks.
#define MAX_DIFF_TEX_COUNT	20

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	GENERATETEXVAR(idx) tex_diffTex ## idx
///
/// \brief	Macro to generate diffuse texture reference names.
/// 		
/// 		This macro was introduced for convenience.  
///
/// \param	idx	Zero-based index the texture reference. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define GENERATETEXVAR(idx) tex_diffTex ## idx
////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	BINDTEX2ARRAY(numTex, idx) if(numTex > idx)
///
/// \brief	Binds the given texture reference to a CUDA texture array.
/// 		
/// 		This macro was introduced for convenience. 
///
/// \param	numTex	Total number of textures. Used to avoid illegal bindings. 
/// \param	idx		Zero-based index the texture to bind. 
////////////////////////////////////////////////////////////////////////////////////////////////////
#define BINDTEX2ARRAY(numTex, idx) \
	if(numTex > idx) \
	{ \
		MNAssert(idx >= 0 && idx < MAX_DIFF_TEX_COUNT); \
		GENERATETEXVAR(idx).addressMode[0] = cudaAddressModeWrap; \
		GENERATETEXVAR(idx).addressMode[1] = cudaAddressModeWrap; \
		GENERATETEXVAR(idx).filterMode = cudaFilterModeLinear; \
		GENERATETEXVAR(idx).normalized = true; \
		mncudaSafeCallNoSync(cudaBindTextureToArray(GENERATETEXVAR(idx), mats.vecTexArrays[Tex_Diffuse][idx], cdClrs)); \
	}
////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	UNBINDTEX(numTex, idx)
///
/// \brief	Unbinds the given texture reference.
///
///			This macro was introduced for convenience.
///
/// \param	numTex	Total number of textures. Used to avoid illegal bindings.
/// \param	idx		Zero-based index the texture to bind.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define UNBINDTEX(numTex, idx) \
	if(numTex > idx) \
		mncudaSafeCallNoSync(cudaUnbindTexture(GENERATETEXVAR(idx)));



// Constant memory data.

/// Light data constant memory variable.
__constant__ LightData c_LightData;
/// Material properties constant memory variable.
__constant__ MaterialProperties c_MatProps;
/// Traingle data constant memory variable.
__constant__ TriangleData c_TriData;
/// Object kd-tree data constant memory variable.
__constant__ KDTreeData c_KDTree;
/// \brief	Ray epsilon.
///
///			See SceneConfig::SetRayEpsilon().
__constant__ float c_fRayEpsilon = 1e-3f;


// Warning: Do not try to move textures or constant stuff into some excluded cu-file to
// import it from other cu-files. This won't work: You'd have to rebind stuff in all cu-files.
// Furthermore it seems to lead to errors.

texture<float4, 1, cudaReadModeElementType> 
					tex_TriV0, ///< First triangle vertices.
					tex_TriV1, ///< Second triangle vertices.
					tex_TriV2; ///< Third triangle vertices.

/// \brief	First vertex triangle normals texture. 
///
///			Required for orientation of geometric normals in dev_ApproximateNormalAt().
texture<float4, 1, cudaReadModeElementType> tex_TriN0;

/// Triangle material indices. One per triangle.
texture<uint, 1, cudaReadModeElementType> tex_TriMatIdx;

texture<float2, 1, cudaReadModeElementType> 
					tex_TriTexCoordA, ///< UV-texture coordinate at first triangle vertices.
					tex_TriTexCoordB, ///< UV-texture coordinate at second triangle vertices.
					tex_TriTexCoordC; ///< UV-texture coordinate at third triangle vertices.

/// Object kd-tree texture for KDTreeData::d_preorderTree.
texture<uint, 1, cudaReadModeElementType> tex_kdTree;

/// \brief	Current number of loaded diffuse textures.
///
///			Limited by ::MAX_DIFF_TEX_COUNT.
uint f_numDiffTextures = 0;

/// \brief	Diffuse texture 1
///
///			I tried to move textures into some kind of array, but this failed since CUDA doesn't support
///			arrays of texture samplers, check the CUDA FAQ. This FAQ proposed two ways to eliminate
///			this problem:
///
///			First: 3D array, 2D textures into slices
///
///			I tried this and it worked. But it has some serious drawbacks:
///			\li Forced to use \c cudaFilterModePoint to avoid fetching from different textures.
///			\li No normalized addressing possbile.
///			\li Texture dimension limited to 2k x 2k.
///			\li	Wasting of memory: The 3D tex has to as large as the largest texture. Example:
///		        One texture 2048x2048 (16 MByte), 10 others 512x512 (10 MByte) would lead to
///		        16 * 11 = 176 MByte instead of 16 + 10 = 26 MByte.
///
///			Hence I dropped this.
///
///			Second: One texture reference for each texture and switch-instruction when fetching.
///			
///			Also some horrible drawbacks:
///			\li	Fixed maximum number of textures.
///			\li A lot of code redundancy.
///
///			Despite these problems I'll give the second approach a try.
texture<uchar4, 2, cudaReadModeNormalizedFloat> 
	tex_diffTex0, 
	tex_diffTex1,  ///< Diffuse texture 2. See ::tex_diffTex0.
	tex_diffTex2,  ///< Diffuse texture 3. See ::tex_diffTex0.
	tex_diffTex3,  ///< Diffuse texture 4. See ::tex_diffTex0.
	tex_diffTex4,  ///< Diffuse texture 5. See ::tex_diffTex0.
	tex_diffTex5,  ///< Diffuse texture 6. See ::tex_diffTex0.
	tex_diffTex6,  ///< Diffuse texture 7. See ::tex_diffTex0.
	tex_diffTex7,  ///< Diffuse texture 8. See ::tex_diffTex0.
	tex_diffTex8,  ///< Diffuse texture 9. See ::tex_diffTex0.
	tex_diffTex9,  ///< Diffuse texture 10. See ::tex_diffTex0.
	tex_diffTex10, ///< Diffuse texture 11. See ::tex_diffTex0.
	tex_diffTex11, ///< Diffuse texture 12. See ::tex_diffTex0.
	tex_diffTex12, ///< Diffuse texture 13. See ::tex_diffTex0.
	tex_diffTex13, ///< Diffuse texture 14. See ::tex_diffTex0.
	tex_diffTex14, ///< Diffuse texture 15. See ::tex_diffTex0.
	tex_diffTex15, ///< Diffuse texture 16. See ::tex_diffTex0.
	tex_diffTex16, ///< Diffuse texture 17. See ::tex_diffTex0.
	tex_diffTex17, ///< Diffuse texture 18. See ::tex_diffTex0.
	tex_diffTex18, ///< Diffuse texture 19. See ::tex_diffTex0.
	tex_diffTex19; ///< Diffuse texture 20. See ::tex_diffTex0.


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float4 dev_FetchDiffTex(int idxTex, float u, float v)
///
/// \brief	Fetches data from a given diffuse texture.	
///
/// \author	Mathias Neumann
/// \date	April 2010
/// \see	::tex_diffTex0
///
/// \param	idxTex	Zero-based texture index.
/// \param	u		The u texture coordinate value. 
/// \param	v		The v texture coordinate value.  
///
/// \return	The fetched texture value at given coordinates. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float4 dev_FetchDiffTex(int idxTex, float u, float v)
{
	switch(idxTex)
	{
	case 0: return tex2D(tex_diffTex0, u, v);
	case 1: return tex2D(tex_diffTex1, u, v);
	case 2: return tex2D(tex_diffTex2, u, v);
	case 3: return tex2D(tex_diffTex3, u, v);
	case 4: return tex2D(tex_diffTex4, u, v);
	case 5: return tex2D(tex_diffTex5, u, v);
	case 6: return tex2D(tex_diffTex6, u, v);
	case 7: return tex2D(tex_diffTex7, u, v);
	case 8: return tex2D(tex_diffTex8, u, v);
	case 9: return tex2D(tex_diffTex9, u, v);
	case 10: return tex2D(tex_diffTex10, u, v);
	case 11: return tex2D(tex_diffTex11, u, v);
	case 12: return tex2D(tex_diffTex12, u, v);
	case 13: return tex2D(tex_diffTex13, u, v);
	case 14: return tex2D(tex_diffTex14, u, v);
	case 15: return tex2D(tex_diffTex15, u, v);
	case 16: return tex2D(tex_diffTex16, u, v);
	case 17: return tex2D(tex_diffTex17, u, v);
	case 18: return tex2D(tex_diffTex18, u, v);
	case 19: return tex2D(tex_diffTex19, u, v);
	}
	return make_float4(0.f, 0.f, 0.f, 0.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_GetColorDiffuse(uint idxTri, uint idxMaterial, char4 matFlags,
/// 	float2 baryHit)
///
/// \brief	Determines the diffuse material color for given triangle hit.
///
///			When there is a diffuse texture for the given material, the color is fetched from
///			that texture. Else the material's diffuse color is used. For area light materials,
///			a white color is returned right now.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param	idxTri		The triangle index. 
/// \param	idxMaterial	The material index. 
/// \param	matFlags	Material flags array. See MaterialProperties::flags.
/// \param	baryHit		Barycentric hit coordinates.
///
/// \return	Diffuse material color (reflectance). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_GetColorDiffuse(uint idxTri, uint idxMaterial, 
									  char4 matFlags, float2 baryHit)
{
	// Early exit for area lights.
	int isAreaLight = matFlags.w & 1;
	if(isAreaLight)
		return make_float3(1.f, 1.f, 1.f);

	// Get material texture index (-1 if none).
	int idxDiffTex = matFlags.x;

	// Barycentric coordinate.
	float3 bary = make_float3(1.f - baryHit.x - baryHit.y, baryHit.x, baryHit.y);

	float3 clrDiffuse;
	if(idxDiffTex != -1 && idxDiffTex < MAX_DIFF_TEX_COUNT)
	{
		// Fetch texture coord.
		float2 texCoordA = tex1Dfetch(tex_TriTexCoordA, idxTri);
		float2 texCoordB = tex1Dfetch(tex_TriTexCoordB, idxTri);
		float2 texCoordC = tex1Dfetch(tex_TriTexCoordC, idxTri);
		float2 texCoord = texCoordA * bary.x + texCoordB * bary.y + texCoordC * bary.z;

		// Fetch from diffuse texture.
		float4 clrDiffTex = dev_FetchDiffTex(idxDiffTex, texCoord.x, texCoord.y);
		clrDiffuse.x = clrDiffTex.x;
		clrDiffuse.y = clrDiffTex.y;
		clrDiffuse.z = clrDiffTex.z;
	}
	else
	{
		// Just use material color.
		clrDiffuse = c_MatProps.clrDiff[idxMaterial];
	}

	return clrDiffuse;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_GetColorSpec(uint idxTri, uint idxMaterial, char4 matFlags,
/// 	float2 baryHit, float* outSpecExp)
///
/// \brief	Determines the diffuse material color for given triangle hit.
///
///			Right now, only the material's color is used. There is no support for specular
///			textures.
///
/// \author	Mathias Neumann
/// \date	April 2010
///
/// \param	idxTri				The triangle index. 
/// \param	idxMaterial			The material index. 
/// \param	matFlags			Material flags array. See MaterialProperties::flags. 
/// \param	baryHit				Barycentric hit coordinates. 
/// \param [out]	outSpecExp	Materials specular exponent (shininess). 
///
/// \return	Specular material color (reflectance). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_GetColorSpec(uint idxTri, uint idxMaterial, 
								   char4 matFlags, float2 baryHit, float* outSpecExp)
{
	*outSpecExp = c_MatProps.specExp[idxMaterial];
	return c_MatProps.clrSpec[idxMaterial];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_SampleLightL(float3 pt, float3 ptLightSample, float3* outW_i)
///
/// \brief	Samples the only primary light source.
/// 		
/// 		The light source is defined by the constant memory variable ::c_LightData. 
///
/// \author	Mathias Neumann
/// \date	March 2010
///
/// \param	pt				The point for which the incident radiance should be calculated. 
/// \param	ptLightSample	Sampled point on light source. 
/// \param [out]	outW_i	Incident light direction (pointing away from \a pt). 
///
/// \return	Incident, direct radiance from primary light source. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_SampleLightL(float3 pt, float3 ptLightSample, float3* outW_i)
{
	LightType type = c_LightData.type;
	float3 L = c_LightData.L_emit;

	// outW_i is incident direction (pointing from pt to light point).

	if(type == Light_Directional)
	{
		*outW_i = -c_LightData.direction;
	}
	else if(type == Light_AreaDisc)
	{		
		float discRadius = c_LightData.areaRadius;

		// Point on correct side test was done by shadow ray casting.
		*outW_i = normalize(ptLightSample - pt);

		// Compute the PDF for sampling with respect to solid angle from pt. This is
		// done by converting the density with respect to surface area to solid angle using
		//
		// \frac{d\omega_i}{dA} = \frac{\cos \theta_o}{r^2}
		//
		// where \theta_o is the angle between light ray and light's surface normal and r^2
		// is the distance between those points. Check PBR p. 702.
		float surfaceArea = MN_PI * discRadius * discRadius;
		float r2 = dev_DistanceSquared(pt, ptLightSample);
		float cos_theta_o = fabsf(dot(c_LightData.direction, -(*outW_i)));
		float pdf = r2 / (cos_theta_o * surfaceArea);

		// divide by PDF for choosing light sample point (area of light).
		L /= pdf;
	}
	else if(type == Light_AreaRect)
	{		
		// Point on correct side test was done by shadow ray casting.
		*outW_i = normalize(ptLightSample - pt);

		// Compute the PDF for sampling with respect to solid angle from pt. This is
		// done by converting the density with respect to surface area to solid angle using
		//
		// \frac{d\omega_i}{dA} = \frac{\cos \theta_o}{r^2}
		//
		// where \theta_o is the angle between light ray and light's surface normal and r^2
		// is the distance between those points. Check PBR p. 702.
		float3 v1 = c_LightData.areaV1;
		float3 v2 = c_LightData.areaV2;
		float surfaceArea = length(v1) * length(v2);
		float r2 = dev_DistanceSquared(pt, ptLightSample);
		float cos_theta_o = fabsf(dot(c_LightData.direction, -(*outW_i)));
		float pdf = r2 / (cos_theta_o * surfaceArea);

		// divide by PDF for choosing light sample point (area of light).
		L /= pdf;
	}
	else if(type == Light_Point)
	{
		// See PBR p. 603. L_emit represents intensity, here.
		*outW_i = normalize(ptLightSample - pt);
		L /= dev_DistanceSquared(ptLightSample, pt);
	}

	return L;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_GetReflectedDirectLight( float3 ptEye, float3 pt, float3 nG,
/// 	float3 nS, float3 ptLightSample, uint idxTri, uint idxMat, char4 matFlags, float2 baryHit)
///
/// \brief	Computes the reflected direct light from \a pt to \a ptEye.
///
/// \todo	Currently only diffuse materials are considered. Implement support for other BSDFs.
///
/// \author	Mathias Neumann
/// \date	25.10.2010
///
/// \param	ptEye			Eye position.
/// \param	pt				Position from which the reflected direct light shall be evaluated.
/// \param	nG				Geometric normal at surface in \a pt.
/// \param	nS				Shading normal at surface in \a pt.
/// \param	ptLightSample	Sampled point on light source.
/// \param	idxTri			Index of the triangle in \a pt.
/// \param	idxMat			Index of the material in \a pt.
/// \param	matFlags		Material flags array. See MaterialProperties::flags.
/// \param	baryHit			Barycentric hit coordinates. 
///
/// \return	Reflected direct radiance from primary light source. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_GetReflectedDirectLight(
								  float3 ptEye, float3 pt, float3 nG, float3 nS, float3 ptLightSample, 
								  uint idxTri,
								  uint idxMat, char4 matFlags,
								  float2 baryHit)
{
	// Get material's reflectance.
	float3 clrDiffuse = dev_GetColorDiffuse(idxTri, idxMat, matFlags, baryHit);
	float transAlpha = c_MatProps.transAlpha[idxMat];

	float3 w_i;
	float3 L_i = dev_SampleLightL(pt, ptLightSample, &w_i);

	float3 f = make_float3(0.f);
	float3 w_o = normalize(ptEye - pt);

	// Evaluate only if w_o and w_i lie in the same hemisphere with respect to the 
	// geometric normal. This avoids light leaks and other problems resulting from the
	// use of shading normals. See PBR, p. 465.
	if (dot(w_o, nG) * dot(w_i, nG) > 0)
		f = (clrDiffuse * transAlpha) * MN_INV_PI;

    return f * L_i * fabsf(dot(w_i, nS));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ int dev_FindNextIntersectionKDWhileWhile(const float3 rayO,
/// 	const float3 rayD, const float tMinRay, const float tMaxRay, float& outLambda,
/// 	float2& outBaryHit)
///
/// \brief	Looks for the next ray triangle intersection using a while-while standard traversal. 
///
/// \note	Required shared memory per thread block: 16 * ::INTERSECT_BLOCKSIZE bytes.
///
/// \author	Mathias Neumann
/// \date	04.03.2010
///
/// \param	rayO				The ray origin. 
/// \param	rayD				The ray direction (normalized). 
/// \param	tMinRay				The ray segment minimum. 
/// \param	tMaxRay				The ray segment maximum. 
/// \param [out]	outLambda	Intersection parameter. Only valid if returned value is not -1. 
/// \param [out]	outBaryHit	Barycentric hit coordinate. Only valid if returned value is not 
/// 							-1. Extracting the hit coordinate computation by returning just
/// 							the hit index worsens performance, even if registers are saved. 
///
/// \return	Returns index of the first intersected triangle or -1, if no intersection found. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool needClosest>
inline __device__ int dev_FindNextIntersectionKDWhileWhile(const float3 rayO, const float3 rayD, const float tMinRay, 
													const float tMaxRay, 
													float& outLambda, float2& outBaryHit)
{
	// Move some data into shared memory to save registers.
	__shared__ float s_invDir[3][INTERSECT_BLOCKSIZE];
	__shared__ int s_idxTriInter[INTERSECT_BLOCKSIZE];

	// Precompute inverse ray direction.
	s_invDir[0][threadIdx.x] = 1.f / rayD.x;
	s_invDir[1][threadIdx.x] = 1.f / rayD.y;
	s_invDir[2][threadIdx.x] = 1.f / rayD.z;
	s_idxTriInter[threadIdx.x] = -1;

	// Compute initial parametric range [tMinScene, tMaxScene] of ray inside kd-tree extent.
	float tMinScene, tMaxScene;
	const bool isIntersectingScene = 
		dev_RayBoxIntersect(c_KDTree.aabbRootMin, c_KDTree.aabbRootMax, rayO, 
			make_float3(s_invDir[0][threadIdx.x], s_invDir[1][threadIdx.x], s_invDir[2][threadIdx.x]),
			tMinRay, tMaxRay, tMinScene, tMaxScene);
	if(!isIntersectingScene)
		return -1;

	// Add epsilon to avoid floating point problems.
	tMaxScene = tMaxScene + c_fRayEpsilon;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	// NOTE: Converting to vector leads to performance drops.
	uint todoAddr[KD_MAX_HEIGHT];
	float todoTMin[KD_MAX_HEIGHT], todoTMax[KD_MAX_HEIGHT];
	uint todoPos = 0;

	// Traverse kd-tree for ray.
	int addrNode = 0;
	float tMin = tMinScene, tMax = tMaxScene;
	outLambda = tMax;

	do
	{
		if(todoPos > 0) // Stack not empty?
		{
			// Pop next node from stack.
			todoPos--;
			addrNode = todoAddr[todoPos];
			tMin = todoTMin[todoPos];
			tMax = todoTMax[todoPos];
		}

		// Read node index + leaf info (MSB).
		// NOTE: We access preorder tree data directly without texture using Fermi's L1 cache.
		uint idxNode = c_KDTree.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			// Texture fetching probably results in a lot of serialization due to cache misses.
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_KDTree.d_preorderTree[addrNode+1], c_KDTree.d_preorderTree[addrNode+2]);
			uint addrRight = parentInfo.x & 0x0FFFFFFF;
			uint splitAxis = parentInfo.x >> 30;
			float splitPos = *(float*)&parentInfo.y;

			// Compute parametric distance along ray to split plane.
			float rayOAxis = ((float*)&rayO)[splitAxis];
			float tSplit = (splitPos - rayOAxis) * s_invDir[splitAxis][threadIdx.x];

			uint addrFirst, addrSec;

			// Get node children pointers for ray.
			bool belowFirst = rayOAxis <= splitPos;
			addrFirst = (belowFirst ? addrLeft : addrRight);
			addrSec = (belowFirst ? addrRight : addrLeft);

			// Advance to next child node, possibly enqueue other child.

			// When the ray origin lies within the split plane (or very close to it), we have to
			// determine the child node using direction comparision. This case was discussed in
			// Havran's thesis "Heuristic Ray Shooting Algorithms", page 100.
			if(fabsf(rayOAxis - splitPos) < 1e-8f)
				// I use the inverse direction here instead of the actual value due to faster shared memory.
				addrNode = ((s_invDir[splitAxis][threadIdx.x] > 0) ? addrSec : addrFirst);
			// NOTE: The operators are very important! >=/<= leads to flickering in Sponza.
			else if(tSplit > tMax || tSplit < 0.f)
				addrNode = addrFirst;
			else if(tSplit < tMin)
				addrNode = addrSec;
			else
			{
				// Enqueue second child in todo list.
				todoAddr[todoPos] = addrSec;
				todoTMin[todoPos] = tSplit;
				todoTMax[todoPos] = tMax;
				todoPos++;

				addrNode = addrFirst;
				tMax = tSplit;
			}

			// Read node index + leaf info (MSB) for new node.
			idxNode = c_KDTree.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		// Now we have a leaf. 
		// Check for intersections inside leaf node.
		uint numTris = c_KDTree.d_preorderTree[addrNode+1];
		for(uint t=0; t<numTris; t++)
		{
			// Get triangle index.
			uint idxTri = c_KDTree.d_preorderTree[addrNode+2+t];

			// Texture fetching seems to be more efficient for my GTX 460. I tried to move the preorderTree
			// accesses to texture fetches and this to global memory accesses, but this hit the performance
			// drastically.
			float3 v0 = make_float3(tex1Dfetch(tex_TriV0, idxTri));
			float3 v1 = make_float3(tex1Dfetch(tex_TriV1, idxTri));
			float3 v2 = make_float3(tex1Dfetch(tex_TriV2, idxTri));

			float baryHit1, baryHit2, lambdaHit;
			bool bHit = dev_RayTriIntersect(v0, v1, v2, rayO, rayD, lambdaHit, baryHit1, baryHit2);
			if(bHit && lambdaHit > c_fRayEpsilon && lambdaHit < outLambda)
			{
				s_idxTriInter[threadIdx.x] = idxTri;
				outBaryHit = make_float2(baryHit1, baryHit2);
				outLambda = lambdaHit;

				if(!needClosest)
					return idxTri;
			}
		}
		if(outLambda < tMax)
		{
			// Early exit.
			break;
		}
	} while(todoPos > 0);

	return s_idxTriInter[threadIdx.x];
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	template <bool getMinDist> __device__ float4 dev_ApproximateNormalAt(float3 ptQuery,
/// 	float queryRadiusSqr)
///
/// \brief	Approximates normal at a given query point.
///
///			A range search is performed in the object kd-tree.
///
/// \todo	Improve normal approximation.
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \tparam	getMinDist		If \c true is passed, the minimum "distance" to triangles in the
///							environment is computed and provided in the w-component of the
///							returned value. The xyz-components contain the geometric normal of
///							the "closest" triangle. If \c false is passed, a weighted interpolation
///							of normals is performed.
///
/// \param	ptQuery			The query point. 
/// \param	queryRadiusSqr	The query radius (squared). 
///
/// \return	xyz: Approximated normal at \a ptQuery. w: Minimum "distance" to a triangle
/// 		if \a getMinDist = \c true. 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool getMinDist>
__device__ float4 dev_ApproximateNormalAt(float3 ptQuery, float queryRadiusSqr)
{
	const float* p = (float*)&ptQuery;

	// Stack gets into local memory. Therefore access is *always* coalesced!
	uint todoAddr[KD_MAX_HEIGHT];
	uint todoPos = 0;

	int addrNode = 0;

	float3 nApprox = make_float3(0.f, 0.f, 0.f);
	float weightSum = 0.f;
	float minDist = MN_INFINITY;
	while(addrNode >= 0)
	{
		// Read node index + leaf info (MSB).
		uint idxNode = c_KDTree.d_preorderTree[addrNode];
		uint isLeaf = idxNode & 0x80000000;
		idxNode &= 0x7FFFFFFF;

		// Find next leaf node.
		while(!isLeaf)
		{
			uint addrLeft = addrNode + 1 + 2;
			uint2 parentInfo = make_uint2(c_KDTree.d_preorderTree[addrNode+1], c_KDTree.d_preorderTree[addrNode+2]);
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
			idxNode = c_KDTree.d_preorderTree[addrNode];
			isLeaf = idxNode & 0x80000000;
			idxNode &= 0x7FFFFFFF;
		}

		// Now we have a leaf. 
		uint numTris = c_KDTree.d_preorderTree[addrNode+1];
		float3 v0, v1, v2;
		uint idxTri;
		float baryHit1, baryHit2, lambdaHit;
		for(uint t=0; t<numTris; t++)
		{
			// Get triangle index.
			idxTri = c_KDTree.d_preorderTree[addrNode+2+t];

			v0 = make_float3(tex1Dfetch(tex_TriV0, idxTri));
			v1 = make_float3(tex1Dfetch(tex_TriV1, idxTri));
			v2 = make_float3(tex1Dfetch(tex_TriV2, idxTri));

			float3 triCross = cross(v1 - v0, v2 - v0);
			float lengthCross = length(triCross);

			// Get geometric normal. Orient it using one triangle normal.
			float3 nG = triCross / lengthCross;
			float3 nTri0 = make_float3(tex1Dfetch(tex_TriN0, idxTri));
			if(dot(nG, nTri0) < 0.f)
				nG *= -1.f;

			// Find intersection of ray (ptQuery, -nG) and triangle.
			// WARGNING: ptQuery might lie within an object as it is a node center!
			float3 rayDir = -nG;
			bool bHit = dev_RayTriIntersect(v0, v1, v2, ptQuery, rayDir, lambdaHit, baryHit1, baryHit2);
			float curDist = fabsf(lambdaHit);
			if(getMinDist) // COMPILE TIME
			{
				if(bHit && curDist < minDist)
				{
					minDist = curDist;
					nApprox = nG;
				}
			}
			else
			{
				// In case we are called again, compute an area weighted average of
				// nearby (in kd-tree sense) triangle normals.
				if(curDist*curDist < queryRadiusSqr)
				{
					float w = 1.f;//0.5f * lengthCross; // = surface area
					nApprox += nG * w;
					weightSum += w;
				}
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

	if(!getMinDist && weightSum > 0.f)
		nApprox /= weightSum;

	return make_float4(nApprox.x, nApprox.y, nApprox.z, minDist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_AddPixelRadiance(const RayChunk& rayChunk,
/// 	const ShadingPoints& shadingPts, const uint tid, const float3 L_sample,
/// 	float4* d_ioRadiance)
///
/// \brief	Updates the radiance accumulator by adding a scaled radiance value. 
///
/// \author	Mathias Neumann
/// \date	27.06.2010
///
/// \param	rayChunk				Source ray chunk. The influence component determines the
/// 								scale factor. 
/// \param	shadingPts				The shading point data. Pixel component gives the index of
/// 								the radiance accumulator. 
/// \param	tid						Index of source ray and shading point. 
/// \param	L_sample				Unscaled radiance value to add. 
/// \param [in,out]	d_ioRadiance	Radiance accumulator (screen buffer). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_AddPixelRadiance(const RayChunk& rayChunk, const ShadingPoints& shadingPts,
									 const uint tid,
									 const float3 L_sample, 
									 float4* d_ioRadiance)
{
	float3 scaledL = make_float3(rayChunk.d_influences[tid]) * L_sample;

	uint idxPixel = shadingPts.d_pixels[tid];

	float4 L_o = d_ioRadiance[idxPixel];
	L_o.x += scaledL.x;
	L_o.y += scaledL.y;
	L_o.z += scaledL.z;
	d_ioRadiance[idxPixel] = L_o;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_FindIntersections(RayChunk rayChunk, ShadingPoints shadingPts,
/// 	uint* d_outIsValid)
///
/// \brief	Searches ray hit points for given ray chunk.
///
///			Calls dev_FindNextIntersectionKDWhileWhile() for intersection search.
///
/// \author	Mathias Neumann
/// \date	February 2010
///
/// \param	rayChunk				The ray chunk.
/// \param	shadingPts				Target data structure for hit points. Is assumed to be empty.
///									For the i-th ray, the i-th entry in this structure is used.
/// \param [out]	d_outIsValid	Binary 0/1 array. Will contain 1 for valid hit points and 0
///									for invalid hit points.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_FindIntersections(RayChunk rayChunk, ShadingPoints shadingPts, uint* d_outIsValid)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < rayChunk.numRays)
	{
		const float3 rayO = make_float3(rayChunk.d_origins[idx]);
		const float3 rayD = make_float3(rayChunk.d_dirs[idx]);

		// NOTE: Currenlty we also trace rays with zero influence, if any.
		float lambda;
		float2 baryHit;
		int idxTri = dev_FindNextIntersectionKDWhileWhile<true>(rayO, rayD, c_fRayEpsilon, MN_INFINITY, 
										lambda, baryHit);
 
		shadingPts.d_pixels[idx] = rayChunk.d_pixels[idx];
		shadingPts.d_idxTris[idx] = idxTri;

		// Avoid branching, so just calculate
		shadingPts.d_ptInter[idx] = make_float4(rayO + rayD * lambda);
		shadingPts.d_baryHit[idx] = baryHit;

		// Store if this result is valid.
		d_outIsValid[idx] = ((idxTri != -1) ? 1 : 0);
	}
}

/*#define PERSIST_ROWSIZE		32 // must be = warp size
#define PERSIST_NUMROWS		4

__device__ const uint df_batchSize = PERSIST_NUMROWS*PERSIST_ROWSIZE;
__device__ uint df_globalNextRay = 0;

__global__ void kernel_FindIntersectionsPersist(RayChunk rayChunk, ShadingPoints shadingPts, uint* d_outIsValid)
{
	__shared__ volatile uint s_nextRay[PERSIST_NUMROWS];
	__shared__ volatile uint s_rayCount[PERSIST_NUMROWS];

	// Initialize ray counts.
	if(threadIdx.x == 0)
		s_rayCount[threadIdx.y] = 0;

	volatile uint& localNextRay = s_nextRay[threadIdx.y];
	volatile uint& localRayCount = s_rayCount[threadIdx.y];

	while(true)
	{
		// Get rays from global to local pool if local pool empty.
		if(localRayCount == 0 && threadIdx.x == 0)
		{
			localNextRay = atomicAdd(&df_globalNextRay, df_batchSize);
			localRayCount = df_batchSize;
		}

		// Get rays from local pool.
		uint myRayIndex = localNextRay + threadIdx.x;
		if(myRayIndex >= rayChunk.numRays)
			return;

		if(threadIdx.x == 0)
		{
			// Update local pool.
			localNextRay += PERSIST_ROWSIZE;
			localRayCount -= PERSIST_ROWSIZE;
		}

		// Trace chosen ray. Do *not* exit loop.
		const float3 rayO = make_float3(rayChunk.d_origins[myRayIndex]);
		const float3 rayD = make_float3(rayChunk.d_dirs[myRayIndex]);

		// NOTE: Currenlty we also trace rays with zero influence, if any.
		float lambda;
		float2 baryHit;
		int idxTri = dev_FindNextIntersectionKDWhileWhile<true>(rayO, rayD, c_fRayEpsilon, MN_INFINITY, 
										lambda, baryHit);
 
		shadingPts.d_pixels[myRayIndex] = rayChunk.d_pixels[myRayIndex];
		shadingPts.d_idxTris[myRayIndex] = idxTri;

		// Avoid branching, so just calculate
		shadingPts.d_ptInter[myRayIndex] = make_float4(rayO + rayD * lambda);
		shadingPts.d_baryHit[myRayIndex] = baryHit;

		// Store if this result is valid.
		d_outIsValid[myRayIndex] = ((idxTri != -1) ? 1 : 0);
	}
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SampleAreaLight(uint numShadingPts, float idxSampleX,
/// 	float invNumSamplesX, float idxSampleY, float invNumSamplesY, float* d_randoms1,
/// 	float* d_randoms2, float4* d_outSamplePts)
///
/// \brief	Samples points on area light. 
///
///			Assumes that ::c_LightData represents an area light source.
///
/// \author	Mathias Neumann
/// \date	June 2010
///
/// \param	numShadingPts			Number of shading points. 
/// \param	idxSampleX				Index of sample X (for stratified sampling).
/// \param	invNumSamplesX			Inverse number of samples X.
/// \param	idxSampleY				Index of sample Y (for stratified sampling).
/// \param	invNumSamplesY			Inverse number of samples Y.
/// \param [in]		d_randoms1		First array of uniform random numbers, one per shading point.
/// \param [in]		d_randoms2		Second array of uniform random numbers, one per shading point.
/// \param [out]	d_outSamplePts	Sampled point on light source area for each shading point.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SampleAreaLight(uint numShadingPts, 
									   float idxSampleX, float invNumSamplesX,
									   float idxSampleY, float invNumSamplesY,
									   float* d_randoms1, float* d_randoms2,
									   float4* d_outSamplePts)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numShadingPts)
	{
		// Sample a direction by sampling a point within the area.
		float3 ptL = c_LightData.position;
		float3 areaNormal = c_LightData.direction;
		LightType type = c_LightData.type;

		float rnd1 = d_randoms1[tid];
		float rnd2 = d_randoms2[tid];

		// Stratify samples.
		rnd1 = (idxSampleX + rnd1) * invNumSamplesX;
		rnd2 = (idxSampleY + rnd2) * invNumSamplesY;

		if(type == Light_AreaDisc)
		{
			float discRadius = c_LightData.areaRadius;
			float3 ptArea = dev_SampleGeneralDisc(ptL, areaNormal, discRadius, rnd1, rnd2);
			d_outSamplePts[tid] = make_float4(ptArea.x, ptArea.y, ptArea.z, 0.f);
		}
		else if(type == Light_AreaRect)
		{
			// Sample position on rect.
			float3 v1 = c_LightData.areaV1;
			float3 v2 = c_LightData.areaV2;
			d_outSamplePts[tid] = make_float4(ptL + rnd1*v1 + rnd2*v2);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TraceShadowRaysDelta(ShadingPoints shadingPts,
/// 	uint* d_outShadowRayResult)
///
/// \brief	Traces shadow rays to delta light sources.
/// 		
/// 		Assumes that ::c_LightData represents a delta light source, i.e. a light source where
/// 		only one direction to the light source is possible (point light, directional light).
/// 		Shadow rays are traced using dev_FindNextIntersectionKDWhileWhile(). 
///
/// \author	Mathias Neumann
/// \date	June 2010
///
/// \param	shadingPts						The shading points. 
/// \param [out]	d_outShadowRayResult	Contains the result of shadow ray tracing. Binary 0/1
/// 										array. Contains 1 iff the light source is unoccluded
/// 										for a given shading point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TraceShadowRaysDelta(ShadingPoints shadingPts,
										   uint* d_outShadowRayResult)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We compacted the intersection result, so no invalid triangle indices.
	if(idx < shadingPts.numPoints)
	{
		// Get the intersection point.
		float3 ptInter = make_float3(shadingPts.d_ptInter[idx]);
		
		float3 ptL = c_LightData.position;
		LightType type = c_LightData.type;

		// Construct vector from point to light (normalized).
		float3 vecPt2L;
		float distance = MN_INFINITY;
		if(type == Light_Directional)
			vecPt2L = -c_LightData.direction;
		else if(type == Light_Point)
		{
			vecPt2L = ptL - ptInter;
			distance = length(vecPt2L);
			vecPt2L /= distance;
		}

		float lambda;
		float2 baryHit;
		int idxIntersect = dev_FindNextIntersectionKDWhileWhile<false>(
								ptInter, vecPt2L, c_fRayEpsilon, distance, lambda, baryHit);

		//__syncthreads(); 

		uint isUnoccluded = 0;
		if(idxIntersect == -1 || lambda >= distance)
			isUnoccluded = 1;

		// Store result.
		d_outShadowRayResult[idx] = isUnoccluded;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TraceShadowRaysArea(ShadingPoints shadingPts, float4* d_samplePts,
/// 	uint* d_outShadowRayResult)
///
/// \brief	Traces shadow rays to area light sources.
/// 		
/// 		Assumes that ::c_LightData represents an area light source. Shadow rays are traced
///			using dev_FindNextIntersectionKDWhileWhile().
///
/// \author	Mathias Neumann
/// \date	June 2010
///
/// \param	shadingPts						The shading points. 
/// \param [in]		d_samplePts				Contains the sample point on the area of the light
/// 										source (for each shading point). Can be generated
/// 										using kernel_SampleAreaLight(). 
/// \param [out]	d_outShadowRayResult	The result of shadow ray tracing. Binary 0/1
/// 										array. Contains 1 iff the light source is unoccluded
/// 										for a given shading point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TraceShadowRaysArea(ShadingPoints shadingPts,
									  float4* d_samplePts,
									  uint* d_outShadowRayResult)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We compacted the intersection result, so no invalid triangle indices.
	if(idx < shadingPts.numPoints)
	{
		// Get the intersection point.
		float3 ptInter = make_float3(shadingPts.d_ptInter[idx]);
		
		float4 sample = d_samplePts[idx];
		float3 vecPt2L = make_float3(sample.x, sample.y, sample.z) - ptInter;
		float distance = length(vecPt2L);
		vecPt2L /= distance;

		uint isUnoccluded = 0;
		if(dot(-vecPt2L, c_LightData.direction) > 0.f) // Point on correct side.
		{
			float lambda;
			float2 baryHit;
			int idxIntersect = dev_FindNextIntersectionKDWhileWhile<false>(
									ptInter, vecPt2L, c_fRayEpsilon, distance, lambda, baryHit); 

			
			if(idxIntersect == -1 || lambda + c_fRayEpsilon >= distance)
				isUnoccluded = 1;
		}

		// Store result.
		d_outShadowRayResult[idx] = isUnoccluded;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TracePhotons(PhotonData photons, uint* d_outIsValid,
/// 	uint* d_outHasNonSpecular, int* d_outTriHitIndex, float2* d_outHitBary,
/// 	float4* d_outHitDiffClr, float4* d_outHitSpecClr)
///
/// \brief	Traces photons into the scene.
/// 		
/// 		Uses dev_FindNextIntersectionKDWhileWhile() to find the next intersection. 
///
/// \author	Mathias Neumann
/// \date	07.04.2010
///
/// \param	photons						The photons to trace. Works inplace, that is, the new
/// 									photons replace the old ones. 
/// \param [out]	d_outIsValid		Binary 0/1 array. Contains 1 iff the corresponding photon
/// 									is valid. A photon is considered as invalid if it found
/// 									no intersection or if its flux is zero. 
/// \param [out]	d_outHasNonSpecular	Binary 0/1 array. If 1, the photon intersected a surface
/// 									that has non- specular components. Else, even for no
/// 									intersections, the element is set to 0. 
/// \param [out]	d_outTriHitIndex	Triangle index of hit triangle for each photon. Will be -1
/// 									if no hit. 
/// \param [out]	d_outHitBary		Barycentric hit coordinates. Will be invalid if no hit. 
/// \param [out]	d_outHitDiffClr		Diffuse colors at hits, will be invalid if no hit. Used
/// 									for reflected photon flux. I decided to move this to this
/// 									file because here I have access to texture and color
/// 									data.  \c xyz contains color and \c w transparency alpha. 
/// \param [out]	d_outHitSpecClr		Specular colors at hits, will be invalid if no hit. Color
/// 									in \c xyz and index of refraction in \c w. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TracePhotons(PhotonData photons, uint* d_outIsValid, uint* d_outHasNonSpecular,
									int* d_outTriHitIndex, float2* d_outHitBary, 
									float4* d_outHitDiffClr, float4* d_outHitSpecClr)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < photons.numPhotons)
	{
		// Read out photon information.
		float3 phPos, phFlux, phDir;
		dev_PhotonLoad(photons, tid, phPos, phDir, phFlux);

		// Trace photon.
		float lambda;
		float2 baryHit;
		int idxIntersect = dev_FindNextIntersectionKDWhileWhile<true>(phPos, phDir, c_fRayEpsilon, MN_INFINITY, 
									lambda, baryHit);

		__syncthreads();

		// Store the updated position (flux still fine as we do this inplace).
		float2 spherical = dev_Direction2Spherical(phDir);
		photons.d_positions[tid] = make_float4(phPos + phDir * lambda, spherical.x);

		d_outTriHitIndex[tid] = idxIntersect;
		d_outHitBary[tid] = baryHit;

		// Get the diffuse color of the surface where the intersection was found.
		float3 clrDiff = make_float3(0.f, 0.f, 0.f), clrSpec = make_float3(0.f, 0.f, 0.f);
		float transAlpha = 1.f;
		float indexTo = 1.f;
		if(idxIntersect != -1)
		{
			uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxIntersect);
			char4 matFlags = c_MatProps.flags[idxMaterial];

			clrDiff = dev_GetColorDiffuse(idxIntersect, idxMaterial, matFlags, baryHit);
			float specExp;
			clrSpec = dev_GetColorSpec(idxIntersect, idxMaterial, matFlags, baryHit, &specExp);
			transAlpha = c_MatProps.transAlpha[idxMaterial];
			indexTo = c_MatProps.indexRefrac[idxMaterial];
		}

		// Store whether the photon is valid or not. A photon is invalid when
		// - It found no intersecting surface.
		// - It's flux is zero in all components.
		bool bHitValid = idxIntersect != -1 && dot(phFlux, phFlux) > 0.f;
		d_outIsValid[tid] = (bHitValid ? 1 : 0);

		// Do we hit something not pure specular?
		bool hasDiffuse = dot(clrDiff, clrDiff) > 0.f && transAlpha > 0.f;
		d_outHasNonSpecular[tid] = ((hasDiffuse && bHitValid) ? 1 : 0);

		// Pack transmission data into 4th component.
		d_outHitDiffClr[tid] = make_float4(clrDiff.x, clrDiff.y, clrDiff.z, transAlpha);
		d_outHitSpecClr[tid] = make_float4(clrSpec.x, clrSpec.y, clrSpec.z, indexTo);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_AddDirectRadiance(RayChunk rayChunk, ShadingPoints shadingPts,
/// 	uint* d_ShadowRayResult, float4* d_lightSamples, float fScale, float4* d_ioRadiance)
///
/// \brief	Adds direct radiance from primary light source.
///
///			The function dev_GetReflectedDirectLight() is used to evaluate the direct radiance
///			reflected from each shading point into the direction given by the source ray.
///
/// \author	Mathias Neumann
/// \date	February 2010
///
/// \param	rayChunk					Source ray chunk. Compacted, so that rays hitting nothing
///										are removed.
/// \param	shadingPts					The shading points. Contains corresponding hits for ray
/// 									chunk. Compacted, so that invalid hits are removed. 
/// \param [in]		d_ShadowRayResult	The shadow ray result. Binary 0/1 buffer. Can be
/// 									generated using kernel_TraceShadowRaysArea() for area
/// 									lights and kernel_TraceShadowRaysDelta() for delta
/// 									lights. 
/// \param [in]		d_lightSamples		Sample point on area light sources for each shading
/// 									point. Set to \c NULL for delta light sources. 
/// \param	fScale						The scale factor. Radiance will be scaled by this factor,
/// 									before it is added to the accumulator. Can be used for
/// 									Monte-Carlo integration. 
/// \param [in,out]	d_ioRadiance		Radiance accumulator screen buffer, i.e. elements are
/// 									associated to screen's pixels. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AddDirectRadiance(RayChunk rayChunk, ShadingPoints shadingPts,
										 uint* d_ShadowRayResult, float4* d_lightSamples, float fScale,
										 float4* d_ioRadiance)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	// We compacted the both ray chunk and shading points, so no invalid triangle indices.
	if(tid < rayChunk.numRays)
	{
		float3 rayO = make_float3(rayChunk.d_origins[tid]);
		float2 baryHit = shadingPts.d_baryHit[tid];

		// Get the intersection point.
		int idxTri = shadingPts.d_idxTris[tid];
		float3 ptIntersect = make_float3(shadingPts.d_ptInter[tid]);
		uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxTri);
		char4 matFlags = c_MatProps.flags[idxMaterial];

		// Get the shadow ray result.
		uint isLightUnoccluded;
		if(d_ShadowRayResult)
			isLightUnoccluded = d_ShadowRayResult[tid];
		else
			isLightUnoccluded = 1;

		// Get light sample.
		float3 ptLightSample = c_LightData.position;
		if(d_lightSamples)
		{
			float4 ptL4 = d_lightSamples[tid];
			ptLightSample = make_float3(ptL4.x, ptL4.y, ptL4.z);
		}

		float3 L_segment = make_float3(0.f, 0.f, 0.f);
		float3 nS = make_float3(shadingPts.d_normalsS[tid]);
		float3 nG = make_float3(shadingPts.d_normalsG[tid]);
		if(isLightUnoccluded)
			L_segment = dev_GetReflectedDirectLight(rayO, ptIntersect, nG, nS, ptLightSample, idxTri,
							idxMaterial, matFlags, baryHit);

		// Scale by global scale factor (1 / SAMPLES).
		L_segment *= fScale; 

		dev_AddPixelRadiance(rayChunk, shadingPts, tid, L_segment, d_ioRadiance);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GetDiffuseColors(int* d_triHitIndices, float2* d_baryHit,
/// 	uint numPoints, float4* d_outClrDiffHit)
///
/// \brief	Generates array of diffuse material colors for given hits.
///
///			This method was added as I wanted to reduce redundancy. Material properties including
///			diffuse textures are available in this file, but not in other cu-files.
///
/// \author	Mathias Neumann
/// \date	25.06.2010
///
/// \param [in]		d_triHitIndices	Triangle index for each hit. Should contain -1 for invalid hits. 
/// \param [in]		d_baryHit		Barycentric coordinates for ecah hit.
/// \param	numPoints				Number of hits. 
/// \param [out]	d_outClrDiffHit	Diffuse colors at hits, will be invalid if no hit. 
///									\c xyz contains color and \c w transparency alpha. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GetDiffuseColors(int* d_triHitIndices, float2* d_baryHit, uint numPoints, 
										float4* d_outClrDiffHit)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numPoints)
	{
		int idxTriHit = d_triHitIndices[tid];
		float3 clrDiff = make_float3(0.f);
		float transAlpha = 0.f;
		if(idxTriHit != -1)
		{
			float2 baryHit = d_baryHit[tid];

			uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxTriHit);
			char4 matFlags = c_MatProps.flags[idxMaterial];
			transAlpha = c_MatProps.transAlpha[idxMaterial];

			// Get the diffuse color of the surface where the intersection was found.
			clrDiff = dev_GetColorDiffuse(idxTriHit, idxMaterial, matFlags, baryHit);		
		}
		d_outClrDiffHit[tid] = make_float4(clrDiff.x, clrDiff.y, clrDiff.z, transAlpha);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_AddEmittedAndIndirect(float4* d_indirectIllum, RayChunk rayChunk,
/// 	ShadingPoints shadingPts, float4* d_ioRadiance)
///
/// \brief	Adds emitting and indirect component of light transport equation.
///
///			The emitting component is handled by checking if the material at the hit
///			points is an area light material. The indirect component is assumed to be computed
///			by some other function and is only added in by this kernel.
///
/// \author	Mathias Neumann
/// \date	26.06.2010
///
/// \param [in]		d_indirectIllum	Computed indirect illumination for each source ray. 
/// \param	rayChunk				Source ray chunk. Compacted, so that rays hitting nothing are
/// 								removed. 
/// \param	shadingPts				The shading points. Contains corresponding hits for ray
/// 								chunk. Compacted, so that invalid hits are removed. 
/// \param [in,out]	d_ioRadiance	Radiance accumulator screen buffer, i.e. elements are
/// 								associated to screen's pixels. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AddEmittedAndIndirect(float4* d_indirectIllum,
											 RayChunk rayChunk, ShadingPoints shadingPts,
											 float4* d_ioRadiance)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < rayChunk.numRays)
	{
		float3 L_add = make_float3(0.f, 0.f, 0.f);

		// Get the intersection point.
		int idxTri = shadingPts.d_idxTris[tid];
		uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxTri);
		char4 matFlags = c_MatProps.flags[idxMaterial];
		int isAreaLight = matFlags.w & 1;
		if(isAreaLight)
		{
			// Add emitted luminance for area lights (first part of light transport equation).
			LightType type = c_LightData.type;
			if(type == Light_AreaDisc || type == Light_AreaRect)
			{
				float3 lightNormal = c_LightData.direction;
				float3 rayD = make_float3(rayChunk.d_dirs[tid]);

				if(dot(lightNormal, -rayD) > 0)
					L_add += c_LightData.L_emit;
			}
		}

		// Read out indirect illumination.
		float4 L_ind = d_indirectIllum[tid];
		L_add.x += L_ind.x;
		L_add.y += L_ind.y;
		L_add.z += L_ind.z;

		dev_AddPixelRadiance(rayChunk, shadingPts, tid, L_add, d_ioRadiance);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ApproximateNormalAt(KDFinalNodeList lstFinal, float queryRadiusMax,
/// 	float4* d_outNormals)
///
/// \brief	Approximates normals at the center of each tree node.
///
///			Calls dev_ApproximateNormalAt() twice. First, the closest triangle intersection
///			is determined. It that intersection is very close to the node center. The triangle's
///			geometric normal is used as approximated normal. Else the device function is called
///			again, now to perform a weighted interpolation of normals of nearby triangles.
///
/// \author	Mathias Neumann
/// \date	23.07.2010
///
/// \param	lstFinal				The final kd-tree node list.
/// \param	queryRadiusMax			The query radius maximum to use for queries in the object
///									kd-tree. 
/// \param [out]	d_outNormals	Approximated normal for each node. Can be zero, if approximation
///									failed.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ApproximateNormalAt(KDFinalNodeList lstFinal, float queryRadiusMax,
										   float4* d_outNormals)
{
	uint idxNode = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxNode < lstFinal.numNodes)
	{
		// Compute photon map node center.
		float3 aabbMin = make_float3(lstFinal.d_aabbMin[idxNode]);
		float3 aabbMax = make_float3(lstFinal.d_aabbMax[idxNode]);
		float3 nodeCenter = aabbMin + 0.5f*(aabbMax - aabbMin);

		// Estimate query radius from photon map node extent.
		float queryRadius = fmaxf(aabbMax.x - aabbMin.x, fmaxf(aabbMax.y - aabbMin.y, aabbMax.z - aabbMin.z));
		queryRadius = fminf(queryRadiusMax, queryRadius);

		// Approximate triangle distance first by finding the closest triangle intersection.
		float4 result = dev_ApproximateNormalAt<true>(nodeCenter, queryRadius*queryRadius);
		float minTriDist = result.w;

		// Get minimum distance normal.
		float3 nMinDist = make_float3(result);
		float3 nApprox = nMinDist;

		if(minTriDist > c_fRayEpsilon)
		{
			// Try to improve normal by area weighted approximation.
			result = dev_ApproximateNormalAt<false>(nodeCenter, queryRadius*queryRadius);
			nApprox = make_float3(result);

			// Just use minimum distance normal in case we found nothing.
			if(dot(nApprox, nApprox) == 0.f)
				nApprox = nMinDist;
		}

		// Normalize!
		float lengthN = length(nApprox);
		if(lengthN > 0.f)
			nApprox = nApprox / lengthN;
		
		d_outNormals[idxNode] = make_float4(nApprox);
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


/// Sets cache configurations for ray tracing kernels.
extern "C"
void RTInitializeKernels()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// We need no shared memory for these kernels. So prefer L1 caching.
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_FindIntersections, cudaFuncCachePreferL1));
	//mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_FindIntersectionsPersist, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_TraceShadowRaysDelta, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_TraceShadowRaysArea, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_TracePhotons, cudaFuncCachePreferL1));
	mncudaSafeCallNoSync(cudaFuncSetCacheConfig(kernel_ApproximateNormalAt, cudaFuncCachePreferL1));
}

/// Binds diffuse texture arrays to texture references.
void BindTextureTextures(const MaterialData& mats)
{
	cudaChannelFormatDesc cdClrs = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdUint2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);

	// Diffuse texture textures.
	f_numDiffTextures = mats.vecTexArrays[Tex_Diffuse].size();
	if(f_numDiffTextures > 0)
	{
		uint numTexs = min((uint)MAX_DIFF_TEX_COUNT, f_numDiffTextures);
		BINDTEX2ARRAY(numTexs, 0);
		BINDTEX2ARRAY(numTexs, 1);
		BINDTEX2ARRAY(numTexs, 2);
		BINDTEX2ARRAY(numTexs, 3);
		BINDTEX2ARRAY(numTexs, 4);
		BINDTEX2ARRAY(numTexs, 5);
		BINDTEX2ARRAY(numTexs, 6);
		BINDTEX2ARRAY(numTexs, 7);
		BINDTEX2ARRAY(numTexs, 8);
		BINDTEX2ARRAY(numTexs, 9);
		BINDTEX2ARRAY(numTexs, 10);
		BINDTEX2ARRAY(numTexs, 11);
		BINDTEX2ARRAY(numTexs, 12);
		BINDTEX2ARRAY(numTexs, 13);
		BINDTEX2ARRAY(numTexs, 14);
		BINDTEX2ARRAY(numTexs, 15);
		BINDTEX2ARRAY(numTexs, 16);
		BINDTEX2ARRAY(numTexs, 17);
		BINDTEX2ARRAY(numTexs, 18);
		BINDTEX2ARRAY(numTexs, 19);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void RTUpdateKernelData(const LightData& lights, const TriangleData& tris,
/// 	const MaterialData& mats, const KDTreeData& kdTree, float fRayEpsilon)
///
/// \brief	Moves scene data to constant memory and texture memory. 
///
/// \author	Mathias Neumann
/// \date	February 2010
///
/// \param	lights		Current scene's light data. 
/// \param	tris		Current scene's triangle data. 
/// \param	mats		Current scene's material data. 
/// \param	kdTree		Current scene's object kd-tree data. 
/// \param	fRayEpsilon	Ray epsilon for current scene.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void RTUpdateKernelData(const LightData& lights, const TriangleData& tris, const MaterialData& mats, 
						const KDTreeData& kdTree, float fRayEpsilon)
{
	// Move data (only pointers, no real data) into constant memory.
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_LightData", &lights, sizeof(LightData)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_MatProps", &mats.matProps, sizeof(MaterialProperties)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_KDTree", &kdTree, sizeof(KDTreeData)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_TriData", &tris, sizeof(TriangleData)));
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_fRayEpsilon", &fRayEpsilon, sizeof(float)));
	
	// Set texture parameters and bind the linear memory to the texture.
	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdUint2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdInt = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaChannelFormatDesc cdFloat2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdClrs = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	tex_TriV0.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV0, tris.d_verts[0], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriV1.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV1, tris.d_verts[1], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriV2.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV2, tris.d_verts[2], cdFloat4, tris.numTris*sizeof(float4)));

	tex_TriN0.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriN0, tris.d_normals[0], cdFloat4, tris.numTris*sizeof(float4)));

	tex_TriMatIdx.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriMatIdx, tris.d_idxMaterial, cdUint, tris.numTris*sizeof(uint)));

	// Texture coord.
	tex_TriTexCoordA.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordA, tris.d_texCoords[0], cdFloat2, tris.numTris*sizeof(float2)));
	tex_TriTexCoordB.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordB, tris.d_texCoords[1], cdFloat2, tris.numTris*sizeof(float2)));
	tex_TriTexCoordC.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordC, tris.d_texCoords[2], cdFloat2, tris.numTris*sizeof(float2)));

	tex_kdTree.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_kdTree, kdTree.d_preorderTree, cdUint, 
		kdTree.sizeTree*sizeof(uint)));

	BindTextureTextures(mats);
}

/// Unbinds textures used for ray tracing kernels.
extern "C"
void RTCleanupKernelData()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV0));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV1));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV2));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriN0));
	
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_kdTree));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriMatIdx));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordA));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordB));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordC));

	if(f_numDiffTextures > 0)
	{
		uint numTex = min((uint)MAX_DIFF_TEX_COUNT, f_numDiffTextures);
		UNBINDTEX(numTex, 0);
		UNBINDTEX(numTex, 1);
		UNBINDTEX(numTex, 2);
		UNBINDTEX(numTex, 3);
		UNBINDTEX(numTex, 4);
		UNBINDTEX(numTex, 5);
		UNBINDTEX(numTex, 6);
		UNBINDTEX(numTex, 7);
		UNBINDTEX(numTex, 8);
		UNBINDTEX(numTex, 9);
		UNBINDTEX(numTex, 10);
		UNBINDTEX(numTex, 11);
		UNBINDTEX(numTex, 12);
		UNBINDTEX(numTex, 13);
		UNBINDTEX(numTex, 14);
		UNBINDTEX(numTex, 15);
		UNBINDTEX(numTex, 16);
		UNBINDTEX(numTex, 17);
		UNBINDTEX(numTex, 18);
		UNBINDTEX(numTex, 19);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_FindIntersections() kernel call.
extern "C"
void KernelRTTraceRays(const RayChunk& rayChunk, ShadingPoints& outInters, uint* d_outIsValid)
{
	dim3 blockSize = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(rayChunk.numRays, blockSize.x), 1, 1);
	kernel_FindIntersections<<<gridSize, blockSize>>>(rayChunk, outInters, d_outIsValid);
	MNCUDA_CHECKERROR;


	// Launch just enough threads to "fill the machine", Aila2009.
	/*uint threadsPerBlock = PERSIST_ROWSIZE*PERSIST_NUMROWS;
	uint threadsToFill = 7*8*threadsPerBlock;
	dim3 blockSize = dim3(PERSIST_ROWSIZE, PERSIST_NUMROWS, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(threadsToFill, threadsPerBlock), 1, 1);
	kernel_FindIntersectionsPersist<<<gridSize, blockSize>>>(rayChunk, outInters, d_outIsValid);*/

	// Set number.
	outInters.numPoints = rayChunk.numRays;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelRTEvaluateLTE(const RayChunk& rayChunk,
/// 	const ShadingPoints& shadingPts, const LightData& lights, float4* d_radianceIndirect,
/// 	bool bDirectRT, bool bTraceShadowRays, uint2 areaLightSamples, float4* d_ioRadiance)
///
/// \brief	Evaluates the light transport equation.
/// 		
/// 		Actually, only emitted and direct illumination are evaluated. The indirect
/// 		illumination is assumed to be computed elsewhere and only passed as parameter. For
/// 		the direct part, evaluation depends on light source type. For delta light sources,
/// 		one shadow ray is enough. For area light sources, Monte-Carlo integration is
/// 		performed to evaluate the illumination integral. 
///
/// \author	Mathias Neumann
/// \date	25.10.2010 \see	kernel_TraceShadowRaysDelta(), kernel_SampleAreaLight(),
/// 		kernel_TraceShadowRaysArea(), kernel_AddDirectRadiance(),
/// 		kernel_AddEmittedAndIndirect()
///
/// \param	rayChunk					Source ray chunk. Compacted, so that rays hitting nothing
/// 									are removed. 
/// \param	shadingPts					The shading points. Contains corresponding hits for ray
/// 									chunk. Compacted, so that invalid hits are removed. 
/// \param	lights						Current scene's light data. 
/// \param [in]		d_radianceIndirect	Computed indirect illumination for each source ray. 
/// \param	bDirectRT					Whether to evaluate direct illumination using ray
/// 									tracing. If false is passed, only indirect and emitted
/// 									illumination are computed. 
/// \param	bTraceShadowRays			Whether to trace shadow rays for direct lighting. If
///										\c false is passed, lights are assumed to be unoccluded.
/// \param	areaLightSamples			Number of area light samples to take. Value is two-
/// 									dimensional to allow stratification of samples. 
/// \param [in,out]	d_ioRadiance		Radiance accumulator screen buffer, i.e. elements are
/// 									associated to screen's pixels. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelRTEvaluateLTE(const RayChunk& rayChunk, const ShadingPoints& shadingPts,
					     const LightData& lights,
					     float4* d_radianceIndirect,
					     bool bDirectRT, bool bTraceShadowRays, uint2 areaLightSamples,
					     float4* d_ioRadiance)
{
	dim3 blockSize = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(rayChunk.numRays, blockSize.x), 1, 1);
	dim3 blockSize2 = dim3(256, 1, 1);
	dim3 gridSize2 = dim3(MNCUDA_DIVUP(rayChunk.numRays, blockSize2.x), 1, 1);

	static StatCounter& ctrShadowRays = StatCounter::Create("Ray tracing", "Traced rays (shadow)");

	// Add direct illumination.
	if(bDirectRT)
	{
		MNCudaMemory<uint> d_ShadowRayResult(shadingPts.numPoints);

		// Shadow ray casting if requested.
		MNCudaMT& mtw = MNCudaMT::GetInstance();
		uint seed = 3494;
		srand(seed);

		if(lights.type != Light_AreaDisc && lights.type != Light_AreaRect)
		{
			if(!bTraceShadowRays)
				d_ShadowRayResult.InitConstant(1);
			else
			{
				kernel_TraceShadowRaysDelta<<<gridSize, blockSize>>>(shadingPts, d_ShadowRayResult);
				ctrShadowRays += shadingPts.numPoints;
			}
			MNCUDA_CHECKERROR;

			kernel_AddDirectRadiance<<<gridSize2, blockSize2>>>(
				rayChunk, shadingPts, d_ShadowRayResult, NULL, 1.f, d_ioRadiance);
			MNCUDA_CHECKERROR;
		}
		else
		{
			// Generate random numbers for area lights.
			uint numRnd = mtw.GetAlignedCount(shadingPts.numPoints);
			MNCudaMemory<float> d_randoms(2*numRnd);
			MNCudaMemory<float4> d_samplePts(shadingPts.numPoints, "Temporary", 256); // 256 byte alignment!

			float invSamplexX = 1.f / float(areaLightSamples.x);
			float invSamplexY = 1.f / float(areaLightSamples.y);
			for(uchar x=0; x<areaLightSamples.x; x++)
			{
				for(uchar y=0; y<areaLightSamples.y; y++)
				{
					mtw.Seed(rand());
					mncudaSafeCallNoSync(mtw.Generate(d_randoms, 2*numRnd));

					kernel_SampleAreaLight<<<gridSize2, blockSize2>>>(shadingPts.numPoints, 
						x, invSamplexX, y, invSamplexY, d_randoms, d_randoms+numRnd, d_samplePts);
					MNCUDA_CHECKERROR;

					if(!bTraceShadowRays)
						d_ShadowRayResult.InitConstant(1);
					else
						kernel_TraceShadowRaysArea<<<gridSize, blockSize>>>(shadingPts, 
							d_samplePts, d_ShadowRayResult);
					MNCUDA_CHECKERROR;

					kernel_AddDirectRadiance<<<gridSize2, blockSize2>>>(rayChunk, shadingPts, 
						d_ShadowRayResult, d_samplePts, invSamplexX*invSamplexY, d_ioRadiance);
					MNCUDA_CHECKERROR;
				}
			}

			if(bTraceShadowRays)
				ctrShadowRays += areaLightSamples.x*areaLightSamples.y*shadingPts.numPoints;
		}	
	}

	// Add emitted and in indirect illumination from photon maps.
	kernel_AddEmittedAndIndirect<<<gridSize2, blockSize2>>>(d_radianceIndirect, 
		rayChunk, shadingPts, d_ioRadiance);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_TracePhotons() kernel call.
extern "C"
void KernelRTTracePhotons(PhotonData& photons, uint* d_outIsValid, uint* d_outHasNonSpecular, 
						  int* d_outTriHitIndex, float2* d_outHitBary, 
						  float4* d_outHitDiffClr, float4* d_outHitSpecClr)
{
	dim3 blockSize = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(photons.numPhotons, blockSize.x), 1, 1);

	// Shoot out photons.
	kernel_TracePhotons<<<gridSize, blockSize>>>(photons, d_outIsValid, d_outHasNonSpecular, 
		d_outTriHitIndex, d_outHitBary, d_outHitDiffClr, d_outHitSpecClr);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_GetDiffuseColors() kernel call.
extern "C"
void KernelRTGetDiffuseColors(int* d_triHitIndices, float2* d_baryHit, uint numPoints, float4* d_outClrDiffHit)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numPoints, blockSize.x), 1, 1);

	kernel_GetDiffuseColors<<<gridSize, blockSize>>>(d_triHitIndices, d_baryHit, numPoints, d_outClrDiffHit);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ApproximateNormalAt() kernel call.
extern "C"
void KernelRTApproximateNormalAt(const KDFinalNodeList& lstFinal, float queryRadiusMax,
							     float4* d_outNormals)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(lstFinal.numNodes, blockSize.x), 1, 1);

	kernel_ApproximateNormalAt<<<gridSize, blockSize>>>(lstFinal, queryRadiusMax, 
		d_outNormals);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////