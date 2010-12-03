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
/// \file	GPU\bsdf.cu
///
/// \brief	Provides kernels related to BSDF handling.
///
///			This file basically combines some BSDF related functions to avoid redundant texture
///			declarations. Most important is the kernel that computes both geometric and shading
///			normal at a set of hit points, kernel_GetNormalsAtHit().
///
/// \author	Mathias Neumann
/// \date	01.07.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "KernelDefs.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"
#include "sample_dev.h"

/// \brief	Maximum number of bump map textures supported.
///
///			The number of bump maps has to be limited to a small amount due to the problem that
///			arrays of textures are not possible, yet.
///
///			Bump mapping is not really working yet. Would require ray differentials.
#define MAX_BUMP_TEXTURES	0

// Constant memory data.

/// \brief	Material properties constant memory variable.
///
///	\todo	I had to choose different variable names to avoid problems with constant memory variables. 
///			Having the same constant memory variable name in multiple cu-files did not work for me. 
///			Maybe the constant memory declarations can be moved somewhere else to avoid this. Using
///			the \c extern keyword is not possible according to the CUDA programming guide, section
///			B.2.4. I furthermore failed to create a single cu-file with constant memory variables,
///			which is included in other cu-files. This technique would at least require to bind
///			the constant variables within each of these cu-files.
__constant__ MaterialProperties c_Mats;

// Textures

texture<float4, 1, cudaReadModeElementType> 
					tex_TriV0, ///< First triangle vertices.
					tex_TriV1, ///< Second triangle vertices.
					tex_TriV2; ///< Third triangle vertices.
texture<float4, 1, cudaReadModeElementType> 
					tex_TriN0, ///< Triangle normals at first triangle vertices.
					tex_TriN1, ///< Triangle normals at second triangle vertices. 
					tex_TriN2; ///< Triangle normals at third triangle vertices.
texture<float2, 1, cudaReadModeElementType> 
					tex_TriTexCoordA, ///< UV-texture coordinate at first triangle vertices.
					tex_TriTexCoordB, ///< UV-texture coordinate at second triangle vertices.
					tex_TriTexCoordC; ///< UV-texture coordinate at third triangle vertices.
/// Triangle material indices. One per triangle.
texture<uint, 1, cudaReadModeElementType> tex_TriMatIdx;

/// \brief	Current number of loaded bump map textures.
///
///			Limited by ::MAX_BUMP_TEXTURES.
size_t f_numBumpTextures = 0;
texture<float, 2, cudaReadModeElementType> 
	tex_bumpMap0, ///< Bump map texture 0.
	tex_bumpMap1, ///< Bump map texture 1. 
	tex_bumpMap2, ///< Bump map texture 2. 
	tex_bumpMap3, ///< Bump map texture 3.
	tex_bumpMap4, ///< Bump map texture 4. 
	tex_bumpMap5, ///< Bump map texture 5. 
	tex_bumpMap6, ///< Bump map texture 6. 
	tex_bumpMap7; ///< Bump map texture 7.

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	GENERATETEXVAR(idx) tex_bumpMap ## idx
///
/// \brief	Macro to generate bump map texture reference names.
///
///			This macro was introduced for convenience.
///
/// \param	idx	Zero-based index the texture reference.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define GENERATETEXVAR(idx) tex_bumpMap ## idx
////////////////////////////////////////////////////////////////////////////////////////////////////
/// \def	BINDTEX2ARRAY(numTex, idx) if(numTex > idx)
///
/// \brief	Binds the given texture reference to a CUDA texture array.
///
///			This macro was introduced for convenience.
///
/// \param	numTex	Total number of textures. Used to avoid illegal bindings.
/// \param	idx		Zero-based index the texture to bind.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define BINDTEX2ARRAY(numTex, idx) \
	if(numTex > idx) \
	{ \
		MNAssert(idx >= 0 && idx < MAX_BUMP_TEXTURES); \
		GENERATETEXVAR(idx).addressMode[0] = cudaAddressModeWrap; \
		GENERATETEXVAR(idx).addressMode[1] = cudaAddressModeWrap; \
		GENERATETEXVAR(idx).filterMode = cudaFilterModeLinear; \
		GENERATETEXVAR(idx).normalized = true; \
		mncudaSafeCallNoSync(cudaBindTextureToArray(GENERATETEXVAR(idx), mats.vecTexArrays[Tex_Bump][idx], cdFloat)); \
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



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_FetchBumpTex(int idxTex, float u, float v)
///
/// \brief	Fetches data from a given bump map texture.	
///
/// \author	Mathias Neumann
/// \date	02.07.2010
///
/// \param	idxTex	Zero-based texture index.
/// \param	u		The u texture coordinate value. 
/// \param	v		The v texture coordinate value. 
///
/// \return	The fetched texture value at given coordinates. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float dev_FetchBumpTex(int idxTex, float u, float v)
{
	switch(idxTex)
	{
	case 0: return tex2D(tex_bumpMap0, u, v);
	case 1: return tex2D(tex_bumpMap1, u, v);
	case 2: return tex2D(tex_bumpMap2, u, v);
	case 3: return tex2D(tex_bumpMap3, u, v);
	case 4: return tex2D(tex_bumpMap4, u, v);
	case 5: return tex2D(tex_bumpMap5, u, v);
	case 6: return tex2D(tex_bumpMap6, u, v);
	case 7: return tex2D(tex_bumpMap7, u, v);
	}
	return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_GetTriPartialDerivates(uint idxTri, float3 p[3], float2 uv[3],
/// 	float3* outDPDU, float3* outDPDV)
///
/// \brief	Computes triangle partial derivates dP/dU and dP/dV of vertex coordinate by U or V
/// 		coordinate. 
///
/// \author	Mathias Neumann
/// \date	02.07.2010
///
/// \param	idxTri			The triangle index. 
/// \param	p				The three triangle vertices. 
/// \param	uv				The uv coordinates at the three triangle vertices. 
/// \param [out]	outDPDU	dP/dU vector. May not be \c NULL. 
/// \param [out]	outDPDV	dP/dV vector. May not be \c NULL. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_GetTriPartialDerivates(uint idxTri, float3 p[3], float2 uv[3],
										   float3* outDPDU, float3* outDPDV)
{
	// Compute deltas for triangle partial derivates.
	float du1 = uv[0].x - uv[2].x;
	float du2 = uv[1].x - uv[2].x;
	float dv1 = uv[0].y - uv[2].y;
	float dv2 = uv[1].y - uv[2].y;
	float3 dp1 = p[0] - p[2], dp2 = p[1] - p[2];

	// Get determinant.
	float det = du1 * dv2 - dv1 * du2;

	if(det == 0.f)
	{
		// In this case, we cannot invert the matrix. Build a coordinate system using geometric normal.
		float3 nG = normalize(cross(p[1] - p[0], p[2] - p[0]));
		dev_BuildCoordSystem(nG, outDPDU, outDPDV);
	}
	else
	{
		float invDet = 1.f / det;
		*outDPDU = invDet * ( dv2 * dp1 - dv1 * dp2);
		*outDPDV = invDet * (-du2 * dp1 + du1 * dp2);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_BumpMapping(uint idxTri, uint idxBumpTex, float3 p[3],
/// 	float3 nG, float2 baryHit)
///
/// \brief	Calculates a normal based on a given bump map.
///
/// \author	Mathias Neumann
/// \date	02.07.2010
///
/// \todo	Add ray differentials to calculate correct offsets for displacement fetching.
///			See \ref lit_pharr "[Pharr and Humphreys 2004]", p. 479.
///
/// \param	idxTri		The triangle index. 
/// \param	idxBumpTex	The bump map texture index. 
/// \param	p			The three triangle vertices.
/// \param	nG			The geometric triangle normal. 
/// \param	baryHit		The barycentric hit coordinates. Specifies the position at which the
///						bump mapped normal should be calculated.
///
/// \return	The calculated normal. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_BumpMapping(uint idxTri, uint idxBumpTex, float3 p[3], float3 nG, float2 baryHit)
{
	// Get vertex UV coordinates.
	float2 uv[3];
	uv[0] = tex1Dfetch(tex_TriTexCoordA, idxTri);
	uv[1] = tex1Dfetch(tex_TriTexCoordB, idxTri);
	uv[2] = tex1Dfetch(tex_TriTexCoordC, idxTri);

	// Get UV coordinate at hit point using barycentric coordinates.
	float3 bary = make_float3(1.f - baryHit.x - baryHit.y, baryHit.x, baryHit.y);
	float2 uvHit = uv[0] * bary.x + uv[1] * bary.y + uv[2] * bary.z;

	// Fetch displacements from bump texture.
	const float delta = 0.01f;
	const float invDelta = 100.f;
	float displaceHit = -dev_FetchBumpTex(idxBumpTex, uvHit.x, uvHit.y);
	float displaceU = -dev_FetchBumpTex(idxBumpTex, uvHit.x + delta, uvHit.y);
	float displaceV = -dev_FetchBumpTex(idxBumpTex, uvHit.x, uvHit.y + delta);

	// Get triangle partial derivates dP/dU and dP/dV.
	float3 dPdU, dPdV;
	dev_GetTriPartialDerivates(idxTri, p, uv, &dPdU, &dPdV);

	// Get bump-mapped partial derivates.
	float3 dPdUBump = dPdU + (displaceU - displaceHit) * invDelta * nG;
	float3 dPdVBump = dPdV + (displaceV - displaceHit) * invDelta * nG;

	// Get bump-mapped normal.
	float3 nBump = normalize(cross(dPdUBump, dPdVBump));

	// Flip shading normal, if required.
	if(dot(nG, nBump) < 0.f)
		nBump *= -1.f;

	return nBump;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GetNormalsAtHit(uint count, int* d_triHitIndices,
/// 	float2* d_hitBaryCoords, float4* d_outNormalsG, float4* d_outNormalsS)
///
/// \brief	Computes normals for BSDF computation from hit data. 
///
///			Both geometric and shading normals are generated for all hit points. There is no need
///			to compact the given hit point data, i.e. removing invalid hits with -1 for the
///			triangle index is not required. The normals for these hits are undefined.
///
/// \author	Mathias Neumann
/// \date	16.06.2010
///
/// \param	count					Number of hit points (shading points, photon hits, ...). 
/// \param [in]		d_triHitIndices	Triangle hit indices. Can be invalid, i.e. -1.
/// \param [in]		d_hitBaryCoords	Barycentric coordinates of hits. 
/// \param [out]	d_outNormalsG	Geometric normals at hits. May not be \c NULL. Array should be
///									at least as large as \a count.
/// \param [out]	d_outNormalsS	Shading normals at hits. May not be \c NULL. Array should be
///									at least as large as \a count.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GetNormalsAtHit(uint count, int* d_triHitIndices, float2* d_hitBaryCoords,
									   float4* d_outNormalsG, float4* d_outNormalsS)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < count)
	{
		int idxTri = d_triHitIndices[tid];

		float3 nG = make_float3(1.f, 0.f, 0.f), nS = make_float3(1.f, 0.f, 0.f);
		if(idxTri != -1)
		{
			float2 baryHit = d_hitBaryCoords[tid];

			// Get geometry normal from vertices.
			float3 p[3];
			p[0] = make_float3(tex1Dfetch(tex_TriV0, idxTri));
			p[1] = make_float3(tex1Dfetch(tex_TriV1, idxTri));
			p[2] = make_float3(tex1Dfetch(tex_TriV2, idxTri));
			nG = normalize(cross(p[1] - p[0], p[2] - p[0]));

			// Check if we have a bump-map.
			uint idxMat = tex1Dfetch(tex_TriMatIdx, idxTri);
			char4 matFlags = c_Mats.flags[idxMat];
			int idxBumpTex = matFlags.y;

			// Get shading normal...
			if(idxBumpTex != -1 && idxBumpTex < MAX_BUMP_TEXTURES)
			{
				// ... using bump-mapping.
				nS = dev_BumpMapping(idxTri, idxBumpTex, p, nG, baryHit);
			}
			else
			{
				//... from interpolation of vertex normals.
				float3 nA = make_float3(tex1Dfetch(tex_TriN0, idxTri));
				float3 nB = make_float3(tex1Dfetch(tex_TriN1, idxTri));
				float3 nC = make_float3(tex1Dfetch(tex_TriN2, idxTri));
				float3 bary = make_float3(1.f - baryHit.x - baryHit.y, baryHit.x, baryHit.y);
				nS = nA * bary.x + nB * bary.y + nC * bary.z;

				// Flip geometric normal, if required.
				if(dot(nG, nS) < 0.f)
					nG *= -1.f;
			}
		}

		// Pack normals to allow coalesced access.
		d_outNormalsG[tid] = make_float4(nG.x, nG.y, nG.z, 0.f);
		d_outNormalsS[tid] = make_float4(nS.x, nS.y, nS.z, 0.f);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_IrradianceToRadiance(float4* d_inIrradiance, float4* d_clrDiffuse,
/// 	uint numHits, float4* d_ioRadiance)
///
/// \brief	Converts irradiance values to radiance values for diffuse BRDFs.
/// 		
/// 		This kernel assumes a diffuse BRDF at all hit points. This enables to move the
///			BRDF out of the radiance-integral.
///
/// \author	Mathias Neumann
/// \date	12.08.2010
///
/// \param [in]		d_inIrradiance	Irradiance at each hit point.
/// \param [in]		d_clrDiffuse	Diffuse material color at each hit point. The w-component should
///									contain the transparency alpha value.
/// \param	numHits					Number of hits, i.e. number of elements in all passed arrays. 
/// \param [in,out]	d_ioRadiance	Incoming radiance accumulator.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_IrradianceToRadiance(float4* d_inIrradiance,
											float4* d_clrDiffuse, uint numHits, float4* d_ioRadiance)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numHits)
	{
		// Get diffuse color of surface.
		float4 c4 = c4 = d_clrDiffuse[tid];
		float3 clrDiffuse = make_float3(c4.x, c4.y, c4.z);
		float transAlpha = c4.w;

		float3 irradiance = make_float3(d_inIrradiance[tid]);

		// Get radiance by multiplication with diffuse BRDF.
		float3 radiance = transAlpha * clrDiffuse * MN_INV_PI * irradiance;
		d_ioRadiance[tid] += make_float4(radiance);
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void BSDFUpdateKernelData(const TriangleData& tris, const MaterialData& mats)
///
/// \brief	Binds triangle vertices, normals, material indices and bump maps to textures. 
///
/// \author	Mathias Neumann
/// \date	01.07.2010
///
/// \param	tris	Triangle data describing the scene geometry. 
/// \param	mats	Material data describing all materials of the scene. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void BSDFUpdateKernelData(const TriangleData& tris, const MaterialData& mats)
{
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_Mats", &mats.matProps, sizeof(MaterialProperties)));


	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc cdFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdFloat2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc cdFloat4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	// Bind vertices to textures.
	tex_TriV0.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV0, tris.d_verts[0], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriV1.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV1, tris.d_verts[1], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriV2.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriV2, tris.d_verts[2], cdFloat4, tris.numTris*sizeof(float4)));

	// Bind normals to textures.
	tex_TriN0.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriN0, tris.d_normals[0], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriN1.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriN1, tris.d_normals[1], cdFloat4, tris.numTris*sizeof(float4)));
	tex_TriN2.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriN2, tris.d_normals[2], cdFloat4, tris.numTris*sizeof(float4)));

	// Texture coord.
	tex_TriTexCoordA.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordA, tris.d_texCoords[0], cdFloat2, tris.numTris*sizeof(float2)));
	tex_TriTexCoordB.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordB, tris.d_texCoords[1], cdFloat2, tris.numTris*sizeof(float2)));
	tex_TriTexCoordC.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriTexCoordC, tris.d_texCoords[2], cdFloat2, tris.numTris*sizeof(float2)));


	tex_TriMatIdx.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriMatIdx, tris.d_idxMaterial, cdUint, tris.numTris*sizeof(uint)));

	// Bind bump map textures.
	f_numBumpTextures = min((size_t)MAX_BUMP_TEXTURES, mats.vecTexArrays[Tex_Bump].size());
	if(f_numBumpTextures > 0)
	{
		BINDTEX2ARRAY(f_numBumpTextures, 0);
		BINDTEX2ARRAY(f_numBumpTextures, 1);
		BINDTEX2ARRAY(f_numBumpTextures, 2);
		BINDTEX2ARRAY(f_numBumpTextures, 3);
		BINDTEX2ARRAY(f_numBumpTextures, 4);
		BINDTEX2ARRAY(f_numBumpTextures, 5);
		BINDTEX2ARRAY(f_numBumpTextures, 6);
		BINDTEX2ARRAY(f_numBumpTextures, 7);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void BSDFCleanupKernelData()
///
/// \brief	Unbinds all texture references. 
///
/// \author	Mathias Neumann
/// \date	01.07.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void BSDFCleanupKernelData()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV0));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV1));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriV2));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriN0));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriN1));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriN2));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordA));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordB));
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriTexCoordC));

	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriMatIdx));

	if(f_numBumpTextures > 0)
	{
		UNBINDTEX(f_numBumpTextures, 0);
		UNBINDTEX(f_numBumpTextures, 1);
		UNBINDTEX(f_numBumpTextures, 2);
		UNBINDTEX(f_numBumpTextures, 3);
		UNBINDTEX(f_numBumpTextures, 4);
		UNBINDTEX(f_numBumpTextures, 5);
		UNBINDTEX(f_numBumpTextures, 6);
		UNBINDTEX(f_numBumpTextures, 7);
		f_numBumpTextures = 0;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_GetNormalsAtHit() kernel call.
extern "C"
void KernelBSDFGetNormalsAtHit(uint count, int* d_triHitIndices, float2* d_hitBaryCoords,
							   float4* d_outNormalsG, float4* d_outNormalsS)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(count, blockSize.x), 1, 1);

	kernel_GetNormalsAtHit<<<gridSize, blockSize>>>(count, d_triHitIndices, d_hitBaryCoords, 
		d_outNormalsG, d_outNormalsS);
	MNCUDA_CHECKERROR;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void KernelBSDFIrradianceToRadiance(ShadingPoints shadingPts,
/// 	float4* d_inIrradiance, float4* d_clrDiffuse, float4* d_ioRadiance)
///
/// \brief	Converts irradiance values to radiance values for diffuse BRDFs.
/// 		
/// 		This kernel assumes a diffuse BRDF at all hit points. This enables to move the BRDF
/// 		out of the radiance-integral. Wraps kernel_GetNormalsAtHit() call for normal
/// 		computation and kernel_IrradianceToRadiance() call for conversion. 
///
/// \author	Mathias Neumann
/// \date	12.08.2010
///
/// \param	shadingPts				The shading points representing the hits. 
/// \param [in]		d_inIrradiance	Irradiance at each hit point. 
/// \param [in]		d_clrDiffuse	Diffuse material color at each hit point. The w-component
/// 								should contain the transparency alpha value. 
/// \param [in,out]	d_ioRadiance	Incoming radiance accumulator. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void KernelBSDFIrradianceToRadiance(ShadingPoints shadingPts, float4* d_inIrradiance,
									float4* d_clrDiffuse, float4* d_ioRadiance)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(shadingPts.numPoints, blockSize.x), 1, 1);

	MNCudaMemory<float4> d_normalsG(shadingPts.numPoints, "Temporary", 256), 
						 d_normalsS(shadingPts.numPoints, "Temporary", 256);
	kernel_GetNormalsAtHit<<<gridSize, blockSize>>>(shadingPts.numPoints, shadingPts.d_idxTris, shadingPts.d_baryHit, 
		d_normalsG, d_normalsS);
	MNCUDA_CHECKERROR;

	kernel_IrradianceToRadiance<<<gridSize, blockSize>>>(d_inIrradiance,
		d_clrDiffuse, shadingPts.numPoints, d_ioRadiance);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////