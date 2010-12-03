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
/// \file	GPU\raygen.cu
///
/// \brief	raygen kernels class. 
/// \author	Mathias Neumann
/// \date	13.02.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////


#include "KernelDefs.h"
#include "CameraModel.h"
#include "RayPool.h"
#include "MNCudaMT.h"
#include "MNCudaMemPool.h"

#include "mncudautil_dev.h"
#include "sample_dev.h"

/// Simple struct to store matrices in constant memory.
struct Matrix
{
	/// The matrix elements.
	float elems[4][4];
};

// Constant memory data.

/// Camera to world space transformation matrix constant memory variable.
__constant__ Matrix c_matCam2World;
/// Raster to camera space transformation matrix constant memory variable.
__constant__ Matrix c_matRaster2Cam;
/// Material properties for current scene. Constant memory variable.
__constant__ MaterialProperties c_Materials;

/// Triangle material index texture, one per triangle.
texture<uint, 1, cudaReadModeElementType> tex_TriMatIdx;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_transformPoint(float trans[4][4], float3 p)
///
/// \brief	Transforms a point using given transform matrix.
///
/// \author	Mathias Neumann
/// \date	05.04.2010
///
/// \param	trans		Elements of the 4x4 transformation matrix.
/// \param	p			The point. Will be converted to homogeneous representation, i.e.
///						\code [p.x, p.y, p.z, 1]^T \endcode
///
/// \return	Transformed point. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_transformPoint(float trans[4][4], float3 p)
{
	float3 res;

	// The homogeneous representation for points is [x, y, z, 1]^T.
	res.x   = trans[0][0]*p.x + trans[0][1]*p.y + trans[0][2]*p.z + trans[0][3];
	res.y   = trans[1][0]*p.x + trans[1][1]*p.y + trans[1][2]*p.z + trans[1][3];
	res.z   = trans[2][0]*p.x + trans[2][1]*p.y + trans[2][2]*p.z + trans[2][3];
	float w = trans[3][0]*p.x + trans[3][1]*p.y + trans[3][2]*p.z + trans[3][3];

	if(w != 1.f)
		res /= w;

	return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_transformVector(float trans[4][4], float3 v)
///
/// \brief	Transforms a vector using given transform matrix.
///
/// \author	Mathias Neumann
/// \date	05.04.2010
///
/// \param	trans		Elements of the 4x4 transformation matrix.
/// \param	v			The vector. Will be converted to homogeneous representation, i.e.
///						\code [v.x, v.y, v.z, 0]^T \endcode
///
/// \return	Transformed vector. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 dev_transformVector(float trans[4][4], float3 v)
{
	float3 res;

	// Note: The homogeneous coords for v are [x, y, z, 0]^T.
	res.x = trans[0][0]*v.x + trans[0][1]*v.y + trans[0][2]*v.z;
	res.y = trans[1][0]*v.x + trans[1][1]*v.y + trans[1][2]*v.z;
	res.z = trans[2][0]*v.x + trans[2][1]*v.y + trans[2][2]*v.z;

	return res;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_genPrimaryRays(uint nScreenW, uint nScreenH, float idxSampleX,
/// 	float invSamplesPerPixelX, float idxSampleY, float invSamplesPerPixelY, float clipHither,
/// 	float clipYon, float* d_randoms1, float* d_randoms2, RayChunk outChunk)
///
/// \brief	Generates primary ray for ray tracing.
///
///			The rays are ordered using the Morton order (Z-curve). This was proposed by Aila et al..
///			Also check http://en.wikipedia.org/wiki/Z-order_%28curve%29. All primary rays for
///			the given sample index are moved into a single ray chunk.
///
/// \author	Mathias Neumann
/// \date	March 2010
///
/// \param	nScreenW			Screen width in pixels.
/// \param	nScreenH			Screen height in pixels.
/// \param	idxSampleX			Sample index X (for stratified sampling).
/// \param	invSamplesPerPixelX	Inverse of the number of samples per pixel X.
/// \param	idxSampleY			Sample index Y (for stratified sampling).
/// \param	invSamplesPerPixelY	Inverse of the number of samples per pixel Y.
/// \param	clipHither			Near clipping plane distance.
/// \param	clipYon				Far clipping plane distance.
/// \param [in]		d_randoms1	First uniform random numbers, one for each pixel. Used for
///								stratified sampling.
/// \param [in]		d_randoms2	Second uniform random numbers, one for each pixel. Used for
///								stratified sampling.
/// \param	outChunk			The target ray chunk. Is assumed to be empty. Do not forget to
///								set ray count after kernel execution.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_genPrimaryRays(uint nScreenW, uint nScreenH, 
									  float idxSampleX, float invSamplesPerPixelX, 
									  float idxSampleY, float invSamplesPerPixelY,
									  float clipHither, float clipYon,
									  float* d_randoms1, float* d_randoms2,
									  RayChunk outChunk)
{
	uint idxPixel = blockIdx.x*blockDim.x + threadIdx.x;	

	if(idxPixel < nScreenW*nScreenH)
	{
		// Assign rays following the Morton order (Z-curve). This was proposed by Aila2009.
		// See http://en.wikipedia.org/wiki/Z-order_%28curve%29
		
		// Extract even bits for x and odd bits for y raster coordinate.
		uint x = 0, y = 0;
		uint srcPos = 0; // Starting with lsb bit 0.
		uint targetPos = 0;
		uint mask = 1;

		// Get raster coordinates for this thread.
		while(mask <= idxPixel)
		{
			bool isOdd = srcPos & 1;
			if(!isOdd && (mask & idxPixel)) // even bit set?
				x |= 1 << targetPos;
			if( isOdd && (mask & idxPixel)) // odd bit set?
				y |= 1 << targetPos;

			// Update mask.
			mask <<= 1;
			srcPos++;

			// Increase target position in case we are done with the odd bit.
			if(isOdd)
				targetPos++;
		}

		float rnd1 = d_randoms1[idxPixel];
		float rnd2 = d_randoms2[idxPixel];
		// Stratify samples.
		rnd1 = (idxSampleX + rnd1) * invSamplesPerPixelX;
		rnd1 = (idxSampleY + rnd1) * invSamplesPerPixelY;

		// Generate camera sample from raster sample.
		float3 ptRaster;
		if(invSamplesPerPixelX*invSamplesPerPixelY < 1.f)
			ptRaster = make_float3(float(x) + rnd1, float(y) + rnd2, 0.f); // See PBR p. 309
		else
			ptRaster = make_float3(float(x) + 0.5f, float(y) + 0.5f, 0.f);
		float3 originCam = dev_transformPoint(c_matRaster2Cam.elems, ptRaster);
		float3 originWorld = dev_transformPoint(c_matCam2World.elems, originCam);

		// originCam is also our direction in *camera* space, but normalized!
		float3 dirCam = normalize(originCam);
		float3 dirWorld = dev_transformVector(c_matCam2World.elems, dirCam);
		dirWorld = normalize(dirWorld);

		// The world origin is generated by transformation
		outChunk.d_origins[idxPixel] = make_float4(originWorld);
		outChunk.d_dirs[idxPixel] = make_float4(dirWorld);

		// Initialize with filter value.
		outChunk.d_influences[idxPixel] = make_float4(1.0f);

		// Set pixel's vertex buffer object index.
		outChunk.d_pixels[idxPixel] = y * nScreenW + x;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_genReflectedRays(RayChunk chunkSrc, ShadingPoints shadingPts,
/// 	RayChunk outChunk, uint* d_outIsValidRay)
///
/// \brief	Generates secondary rays for specular reflection.
///
///			Calls ::dev_SampleDirectionSpecReflect() to generate reflected direction. Rays are
///			flagged as invalid when their influence RayChunk::d_influences[i] falls below \c 0.01f
///			for all components.
///
/// \author	Mathias Neumann
/// \date	March 2010
///
/// \param	chunkSrc				Source ray chunk. 
/// \param	shadingPts				Shading points (hit points) of source rays.
/// \param	outChunk				Target ray chunk. Is assumed to be empty. Do not forget to
///									set ray count after kernel execution.
/// \param [out]	d_outIsValidRay	Binary 0/1 array. Will contain 1 for valid rays, 0 for invalid
///									rays. The latter can be removed by compaction.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_genReflectedRays(RayChunk chunkSrc, ShadingPoints shadingPts,
										RayChunk outChunk, uint* d_outIsValidRay)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We compacted the shading points, so no invalid triangle indices.
	if(idx < shadingPts.numPoints)
	{
		// Intersection point is ray source.
		outChunk.d_origins[idx] = shadingPts.d_ptInter[idx];

		// Get ray direction.
		float3 vSrcRayDir = make_float3(chunkSrc.d_dirs[idx]);

		int idxTri = shadingPts.d_idxTris[idx];

		// Fetch shading normal.
		float4 n4 = shadingPts.d_normalsS[idx];
		float3 nS = make_float3(n4.x, n4.y, n4.z);

		uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxTri);
		float3 specColor = c_Materials.clrSpec[idxMaterial];

		float3 vReflected;
		float pdf; // Will be one, so no need to divide by.
		float3 f = dev_SampleDirectionSpecReflect(-vSrcRayDir, nS, 
			0.f, 0.f, specColor, &vReflected, &pdf);

		outChunk.d_dirs[idx] = make_float4(vReflected);

		float3 infl = f * fabsf(dot(vReflected, nS)) * make_float3(chunkSrc.d_influences[idx]);

		outChunk.d_influences[idx] = make_float4(infl);
		outChunk.d_pixels[idx] = chunkSrc.d_pixels[idx];

		// Mark low influence rays as invalid to avoid tracing them.
		uint isValid = ((infl.x >= 0.01f || infl.y >= 0.01f || infl.z >= 0.01f) ? 1 : 0);
		d_outIsValidRay[idx] = isValid;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_genTransmittedRays(RayChunk chunkSrc, ShadingPoints shadingPts,
/// 	RayChunk outChunk, uint* d_outIsValidRay)
///
/// \brief	Generates secondary rays for specular transmission.
///
///			Calls ::dev_SampleDirectionSpecTransmit() to generate transmitted direction. Rays are
///			flagged as invalid when their influence RayChunk::d_influences[i] falls below \c 0.01f
///			for all components.
///
/// \author	Mathias Neumann
/// \date	March 2010
///
/// \param	chunkSrc				Source ray chunk. 
/// \param	shadingPts				Shading points (hit points) of source rays.
/// \param	outChunk				Target ray chunk. Is assumed to be empty. Do not forget to
///									set ray count after kernel execution.
/// \param [out]	d_outIsValidRay	Binary 0/1 array. Will contain 1 for valid rays, 0 for invalid
///									rays. The latter can be removed by compaction.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_genTransmittedRays(RayChunk chunkSrc, ShadingPoints shadingPts,
										  RayChunk outChunk, uint* d_outIsValidRay)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < shadingPts.numPoints)
	{
		// Triangle index is valid since we compacted the shading point array.
		int idxTri = shadingPts.d_idxTris[idx];
		uint idxMaterial = tex1Dfetch(tex_TriMatIdx, idxTri);

		// Intersection point is ray source.
		outChunk.d_origins[idx] = shadingPts.d_ptInter[idx];

		// Fetch shading normal.
		float4 n4 = shadingPts.d_normalsS[idx];
		float3 nS = make_float3(n4.x, n4.y, n4.z);

		// Get source ray direction.
		float3 vSrcRayDir = make_float3(chunkSrc.d_dirs[idx]);

		// Get indices of refraction in correct order.
		float n_from = 1.f;
		float n_to = c_Materials.indexRefrac[idxMaterial];
		if(dot(nS, -vSrcRayDir) < 0.f)
		{
			// Swap...
			float temp = n_from;
			n_from = n_to;
			n_to = temp;

			// Now ensure normal and -vSrcRayDir lie in the same hemisphere.
			nS *= -1.f;
		}

		// Sample refracted direction from BTDF, see PBR, p. 433.
		float transAlpha = c_Materials.transAlpha[idxMaterial];
		float3 clrTransmit = c_Materials.clrDiff[idxMaterial] * (1.f - transAlpha);
		float3 vRefract;
		float pdf; // Will be one, so no need to divide by.
		float3 f = dev_SampleDirectionSpecTransmit(-vSrcRayDir, nS, 
			0.f, 0.f, clrTransmit, n_from/n_to, false, &vRefract, &pdf);

		outChunk.d_dirs[idx] = make_float4(vRefract);
		
		float3 infl = f * fabsf(dot(vRefract, nS)) * make_float3(chunkSrc.d_influences[idx]);
		outChunk.d_influences[idx] = make_float4(infl);
		outChunk.d_pixels[idx] = chunkSrc.d_pixels[idx];

		// Mark low influence rays as invalid to avoid tracing them.
		uint isValid = ((infl.x >= 0.01f || infl.y >= 0.01f || infl.z >= 0.01f) ? 1 : 0);
		d_outIsValidRay[idx] = isValid;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_genFinalGatherRays(ShadingPoints shadingPts, float4* d_clrDiffHit,
/// 	float* d_randoms1, float* d_randoms2, float idxFGRayX, float invNumFGRaysX,
/// 	float idxFGRayY, float invNumFGRaysY, RayChunk outChunk)
///
/// \brief	Generates gather rays for final gathering. 
///
///			Currently only diffuse BRDFs are supported for final gathering. For them, the
///			::dev_SampleDirectionLambertian() function is used to sample directions for gather rays.
///
/// \author	Mathias Neumann
/// \date	12.04.2010
///
/// \param	shadingPts				Source shading points for final gather rays. 
/// \param [in]		d_clrDiffHit	Diffuse material color at each shading point. Used for
/// 								diffuse BRDF evaluation. \c xyz contains color and \c w
/// 								transparency alpha. 
/// \param [in]		d_randoms1		First uniform random number array, one per shading point.
/// 								Used for direction sampling. 
/// \param [in]		d_randoms2		Second uniform random number array, one per shading point.
/// 								Used for direction sampling. 
/// \param	idxFGRayX				Final gather ray index X (for stratified sampling). 
/// \param	invNumFGRaysX			Inverse of the number of final gather rays X. 
/// \param	idxFGRayY				Final gather ray index Y (for stratified sampling). 
/// \param	invNumFGRaysY			Inverse of the number of final gather rays Y. 
/// \param	outChunk				Target ray chunk for gather rays. Is assumed to be empty. Do
/// 								not forget to set ray count after kernel execution. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_genFinalGatherRays(ShadingPoints shadingPts, float4* d_clrDiffHit,
										  float* d_randoms1, float* d_randoms2,
										  float idxFGRayX, float invNumFGRaysX,
										  float idxFGRayY, float invNumFGRaysY,
										  RayChunk outChunk)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	// We compacted the intersection result, so no invalid triangle indices.
	if(tid < shadingPts.numPoints)
	{
		// It's important to use the geometric normal here. Else gathering would not work correctly
		// as ray casting depends on the geometry! See PBR p. 761.
		float3 n_g = make_float3(shadingPts.d_normalsG[tid]);

		// Get diffuse color of hit triangle.
		float4 clrDiffHit4 = d_clrDiffHit[tid];
		float transAlpha = clrDiffHit4.w;
		float3 clrDiffHit = make_float3(clrDiffHit4) * transAlpha;

		// Sample a random direction in the same hemisphere for diffuse reflection.
		// NOTE: I tried multiple RNGs for this. The QRNG using radical inverses didn't work and
		//       lead to high noise and errors in the picture. A simple LCG RNG wasn't better.
		//		 Therefore I now use pregenerated random numbers from the Mersenne Twister of
		//		 the CUDA SDK 3.0.
		float rnd1 = d_randoms1[tid];
		float rnd2 = d_randoms2[tid]; 

		// Stratify samples. In both directions to get best cache performance. I use a very basic
		// stratification here, but noise reduction results are OK.
		rnd1 = (idxFGRayX + rnd1) * invNumFGRaysX;
		rnd2 = (idxFGRayY + rnd2) * invNumFGRaysY;

		// Assume that the source ray arrived from the upper hemisphere with respect to the
		// geometric normal. As we do not perform final gathering at specular surfaces, this
		// assumption is valid as long as the camera is not within some object. Basically we
		// can avoid keeping track of the incoming directions for the given shading points
		// when using this assumption.
		float pdf = 0.f;
		float3 w_i = make_float3(1.f, 0.f, 0.f);
		float3 w_o = n_g; // See above.
		float3 f_r = dev_SampleDirectionLambertian(w_o, n_g, rnd1, rnd2, clrDiffHit, &w_i, &pdf);

		// Do not perform final gathering for specular surfaces. There is just NO WAY to generate
		// final gather rays physically.
		bool hasNonSpecular = clrDiffHit.x != 0.f || clrDiffHit.y != 0.f || clrDiffHit.z != 0.f;

		// Alpha, that is the ray influence, should contain the PI / numSamples value for final gathering
		// for irradinace, see PBR p. 762.
		float fgScale = 0.f;
		if(pdf != 0.f && hasNonSpecular)
			fgScale = MN_PI * (invNumFGRaysX * invNumFGRaysY);

		// Avoid evaluation in case w_i and w_o lie in different hemispheres 
		// with respect to n_g. PBR p. 465 or VeachPhD, p. 153.
		//if(dot(w_o, n_g) * dot(w_i, n_g) <= 0.f)
		//	fgScale = 0.f;

		// Store the new ray.
		outChunk.d_origins[tid] = shadingPts.d_ptInter[tid]; 
		outChunk.d_dirs[tid] = make_float4(w_i);
		outChunk.d_influences[tid] = make_float4(fgScale);
		outChunk.d_pixels[tid] = shadingPts.d_pixels[tid];
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void RTUpdateRayGenKernels(const TriangleData& tris, const MaterialData& mats)
///
/// \brief	Binds textures and sets constant memory variables.
///
/// \author	Mathias Neumann
/// \date	13.02.2010
///
/// \param	tris	Triangle data for current scene.
/// \param	mats	Material data for current scene.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void RTUpdateRayGenKernels(const TriangleData& tris, const MaterialData& mats)
{
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_Materials", &mats.matProps, sizeof(MaterialProperties)));

	cudaChannelFormatDesc cdUint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

	tex_TriMatIdx.normalized = false;
	mncudaSafeCallNoSync(cudaBindTexture(NULL, tex_TriMatIdx, tris.d_idxMaterial, cdUint, tris.numTris*sizeof(uint)));
}

/// Unbinds textures used for ray generation kernels.
extern "C"
void RTCleanupRayGenKernels()
{
	mncudaSafeCallNoSync(cudaUnbindTexture(tex_TriMatIdx));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_genPrimaryRays() kernel call.
extern "C"
void KernelRTPrimaryKernel(CameraModel* pCamera, 
						   uint idxSampleX, uint samplesPerPixelX, uint idxSampleY, uint samplesPerPixelY,
						   RayChunk& outChunk)
{
	uint screenW = pCamera->GetScreenWidth();
	uint screenH = pCamera->GetScreenHeight();
	MNTransform cam2world = pCamera->GetCamera2World();
	MNTransform raster2cam = pCamera->GetRaster2Camera();

	// Move matrices to constant memory.
	// WARNING: Cannot pass matrix[4][4] variables per parameter in kernel!
	Matrix matCam2World;
	for(uint i=0; i<4; i++)
		for(uint j=0; j<4; j++)
			matCam2World.elems[i][j] = cam2world.GetMatrix(i, j);
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_matCam2World", &matCam2World, sizeof(Matrix)));
	Matrix matRaster2Cam;
	for(uint i=0; i<4; i++)
		for(uint j=0; j<4; j++)
			matRaster2Cam.elems[i][j] = raster2cam.GetMatrix(i, j);
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_matRaster2Cam", &matRaster2Cam, sizeof(Matrix)));

	uint numPixels = screenW*screenH;
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numPixels, blockSize.x), 1);

	MNCudaMT& mtw = MNCudaMT::GetInstance();
	uint numRnd = mtw.GetAlignedCount(numPixels);
	MNCudaMemory<float> d_randoms(2*numRnd);

	mtw.Seed(rand());
	mncudaSafeCallNoSync(mtw.Generate(d_randoms, 2*numRnd));

	// Generate primary rays.
	float invSamplesPerPixelX = ((samplesPerPixelX > 1) ? 1.f / float(samplesPerPixelX) : 1.f);
	float invSamplesPerPixelY = ((samplesPerPixelY > 1) ? 1.f / float(samplesPerPixelY) : 1.f);
	kernel_genPrimaryRays<<<gridSize, blockSize>>>(
		screenW, screenH, (float)idxSampleX, invSamplesPerPixelX, (float)idxSampleY, invSamplesPerPixelY, 
		pCamera->GetClipHither(), pCamera->GetClipYon(), 
		d_randoms, d_randoms+numRnd, outChunk);
	MNCUDA_CHECKERROR;

	// Update chunk status.
	outChunk.rekDepth = 0;
	outChunk.numRays = numPixels;
}

/// Wraps kernel_genReflectedRays() kernel call.
extern "C"
void KernelRTReflectedKernel(RayChunk& chunkSrc, ShadingPoints& shadingPts,
						   TriangleData& triData, RayChunk& outChunk, uint* d_outIsValid)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(chunkSrc.numRays, blockSize.x), 1, 1);

	// Generate rays into out chunk.
	kernel_genReflectedRays<<<gridSize, blockSize>>>(
		chunkSrc, shadingPts, outChunk, d_outIsValid);
	MNCUDA_CHECKERROR;

	// Increase recursion depth.
	outChunk.rekDepth = chunkSrc.rekDepth + 1;
	outChunk.numRays = chunkSrc.numRays;
}

/// Wraps kernel_genTransmittedRays() kernel call.
extern "C"
void KernelRTTransmittedKernel(RayChunk& chunkSrc, ShadingPoints& shadingPts,
						   TriangleData& triData, RayChunk& outChunk, uint* d_outIsValid)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(chunkSrc.numRays, blockSize.x), 1, 1);

	// Generate rays into out chunk.
	kernel_genTransmittedRays<<<gridSize, blockSize>>>(
		chunkSrc, shadingPts, outChunk, d_outIsValid);
	MNCUDA_CHECKERROR;

	// Increase recursion depth.
	outChunk.rekDepth = chunkSrc.rekDepth + 1;
	outChunk.numRays = chunkSrc.numRays;
}

/// Wraps kernel_genFinalGatherRays() kernel call.
extern "C"
void KernelRTFinalGatherRays(const ShadingPoints& shadingPts, float4* d_clrDiffHit,
						     float* d_randoms1, float* d_randoms2,
						     uint idxFGRayX, uint numFGRaysX,
						     uint idxFGRayY, uint numFGRaysY,
						     RayChunk& outChunk)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(shadingPts.numPoints, blockSize.x), 1, 1);

	float invNumRaysX = ((numFGRaysX > 1) ? 1.f / float(numFGRaysX) : 1.f);
	float invNumRaysY = ((numFGRaysY > 1) ? 1.f / float(numFGRaysY) : 1.f);
	kernel_genFinalGatherRays<<<gridSize, blockSize>>>(shadingPts, d_clrDiffHit, d_randoms1, d_randoms2, 
		(float)idxFGRayX, invNumRaysX, (float)idxFGRayY, invNumRaysY, outChunk);
	MNCUDA_CHECKERROR;

	outChunk.rekDepth = 0;
	outChunk.numRays = shadingPts.numPoints;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////