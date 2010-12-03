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
/// \file	GPU\photon_build.cu
///
/// \brief	Kernels for photon map construction, specifically photon tracing.
///
/// \author	Mathias Neumann
/// \date	09.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////


#include "KernelDefs.h"

#include "photon_dev.h"
#include "sample_dev.h"

/// Light data constant memory variable.
__constant__ LightData c_Lights;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SpawnLightPhotons(uint photonOffset, uint numToSpawn,
/// 	float3 worldCenter, float worldRadius, PhotonData outPhotonSpawn)
///
/// \brief	Spawns photons from the given light source.
/// 		
/// 		The number of photons is given fixed and controls the number of threads to spawn, as
/// 		each thread handles a photon. 
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	photonOffset	The photon offset (depends on how many photons spawned already). Is
/// 						used to compute distinct members of the halton sequence for all
/// 						spawned photons. 
/// \param	numToSpawn		Number of photons to spawn. 
/// \param	worldCenter		The world center. Used for directional lights.
/// \param	worldRadius		The world radius. Used for directional lights. 
/// \param	outPhotonSpawn	Will contain the spawned photons. All previous contents are overwritten.
///							Remember to set the new photon count after kernel execution.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SpawnLightPhotons(uint photonOffset, uint numToSpawn,
										 float3 worldCenter, float worldRadius,
										 PhotonData outPhotonSpawn)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numToSpawn)
	{
		float3 myPos, myFlux, myDir;

		float rnd1 = dev_RadicalInverse(photonOffset + tid+1, 2);
		float rnd2 = dev_RadicalInverse(photonOffset + tid+1, 3);

		float3 ptL = c_Lights.position;

		LightType type = c_Lights.type;
		if(type == Light_Point)
		{
			myPos = ptL;
			// Note that point lights store intensity instead of emitted radiance.
			float3 intensity = c_Lights.L_emit;

			// Power (flux) emitted by a point light is found by integrating intensity over the
			// entire sphere of directions, see PBR page 603. This is exactly the same as we would
			// get when dividing the intensity by the PDF of sampling the point light source,
			// 1.f / (4.f * MN_PI).
			myFlux = intensity * 4.f * MN_PI;

			// Generate photons direction using QRNG. We use the 3D Halton sequence we can 
			// generate from the radical inverse function. As proposed in "Physically based
			// rendering", we use the first 3 prime numbers as bases.
			myDir.x = 2.f * rnd1 - 1.f;
			myDir.y = 2.f * rnd2 - 1.f;
			myDir.z = 2.f * dev_RadicalInverse(photonOffset + tid+1, 5) - 1.f;
			// Avoid myDir = 0.
			if(myDir.x == myDir.y && myDir.y == myDir.z && myDir.z == 0.f)
				myDir.x = 1.f;
			// Normalize direction.
			myDir = normalize(myDir);
		}
		else if(type == Light_AreaDisc)
		{
			float3 discNormal = c_Lights.direction;
			float3 L_emit = c_Lights.L_emit;
			float discRadius = c_Lights.areaRadius;

			// Sample position on disc.
			myPos = dev_SampleGeneralDisc(ptL, discNormal, discRadius, rnd1, rnd2);

			// Cosine-sample direction.
			float rnd3 = dev_RadicalInverse(photonOffset + tid+1, 5);
			float rnd4 = dev_RadicalInverse(photonOffset + tid+1, 7);
			float pdfCosine;
			myDir = dev_SampleHemisphereCosine(discNormal, rnd3, rnd4, &pdfCosine);

			// Compute PDF for sampling directions from the area light. That is the product
			// of the PDF for sampling the ray origin myPos with respect to the surface area
			// with the PDF of sampling the direction.
			// See PBR, p704. Note that the PDF for sampling a point on the surface area is
			// just 1 / area.
			float surfaceArea = MN_PI * discRadius * discRadius;
			float pdf = pdfCosine / surfaceArea;

			// No need to check whether we are emiting to the wrong side.
			if(pdf != 0.f)
				myFlux = fabsf(dot(myDir, discNormal)) * L_emit / pdf;
			else
				myFlux = make_float3(0.f);
		}
		else if(type == Light_AreaRect)
		{
			float3 rectNormal = c_Lights.direction;
			float3 L_emit = c_Lights.L_emit;

			// Sample position on rect.
			float3 v1 = c_Lights.areaV1;
			float3 v2 = c_Lights.areaV2;
			myPos = ptL + rnd1*v1 + rnd2*v2;

			// Cosine-sample direction.
			float rnd3 = dev_RadicalInverse(photonOffset + tid+1, 5);
			float rnd4 = dev_RadicalInverse(photonOffset + tid+1, 7);
			float pdfCosine;
			myDir = dev_SampleHemisphereCosine(rectNormal, rnd3, rnd4, &pdfCosine);

			// Compute PDF for sampling directions from the area light. That is the product
			// of the PDF for sampling the ray origin myPos with respect to the surface area
			// with the PDF of sampling the direction.
			// See PBR, p704. Note that the PDF for sampling a point on the surface area is
			// just 1 / area.
			float surfaceArea = length(v1) * length(v2);
			float pdf = pdfCosine / surfaceArea;

			// No need to check whether we are emiting to the wrong side.
			if(pdf != 0.f)
				myFlux = fabsf(dot(myDir, rectNormal)) * L_emit / pdf;
			else
				myFlux = make_float3(0.f);
		}
		else if(type == Light_Directional)
		{
			float3 lightDir = c_Lights.direction;
			float3 L_emit = c_Lights.L_emit;

			float3 ptDisk = dev_SampleGeneralDisc(worldCenter, lightDir, worldRadius, 
										rnd1, rnd2);

			float pdf = MN_INV_PI / (worldRadius*worldRadius);

			// Now set photon properties.
			myPos = ptDisk - worldRadius * lightDir; // Offset point
			myDir = lightDir;
			myFlux = L_emit / pdf;
		}

		// Store the photon in the spawn list.
		dev_PhotonStore(outPhotonSpawn, tid, myPos, myDir, myFlux);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_RussianRoulette(float* d_randoms, float contProbability,
/// 	float invContProbability, PhotonData ioPhotons, uint* d_ioIsValid)
///
/// \brief	Marks photons for elimination using russian roulette.
///
///			Due to parallel exectution, this kernel does not perform the actual elimination. Only
///			a valid flag array is updated. However, photon powers are scaled according to russian
///			roulette.
///
/// \author	Mathias Neumann
/// \date	23.06.2010
///
/// \param [in]		d_randoms	Uniform random numbers, one for each photon.
/// \param	contProbability		The continue probability. 
/// \param	invContProbability	The inverse continue probability. 
/// \param	ioPhotons			Photon data to consider. All photon powers are scaled by the inverse
///								continue probability according to the russian roulette scheme.
/// \param [in,out]	d_ioIsValid	Pass in the old valid flags (binary 0/1 array). For each eliminated
///								photon its flag is forced to 0.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_RussianRoulette(float* d_randoms, float contProbability,
									   float invContProbability,
									   PhotonData ioPhotons, uint* d_ioIsValid)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < ioPhotons.numPhotons)
	{
		// Just read, even when corresponding photon not valid.
		float rnd = d_randoms[tid];

		uint oldValid = d_ioIsValid[tid];
		d_ioIsValid[tid] = ((rnd <= contProbability && oldValid) ? 1 : 0);

		// Update flux to account for missing contributions of terminated paths (PBR p. 781).
		float4 phFlux = ioPhotons.d_powers[tid];
		phFlux.x *= invContProbability; // alpha /= contProbability
		phFlux.y *= invContProbability;
		phFlux.z *= invContProbability;
		ioPhotons.d_powers[tid] = phFlux;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_SpawnScatteredPhotons(PhotonData ioPhotons, float4* d_hitDiffClrs,
/// 	float4* d_hitSpecClrs, float4* d_normalG, float4* d_normalS, float* d_randoms1,
/// 	float* d_randoms2, float* d_randoms3, uint* d_outIsLastSpecular, uint* d_outIsValid)
///
/// \brief	Spawns scattered photons for given photon data (inplace).
/// 		
/// 		Each thread works on a single photon and generates a new, scattered photon. Up to
/// 		three random numbers are required for a single photon (BSDF selection, new direction
/// 		selection). 
///
///			Right now I disabled BSDF selection and handle diffuse BRDFs only. Specular surfaces
///			were added, but never fully implemented, so that parts of MNRT are not ready for them.
///
/// \author	Mathias Neumann
/// \date	12.04.2010
///
/// \param	ioPhotons					The photon data to update inplace. 
/// \param [in]		d_hitDiffClrs		Diffuse color of the surface hit by each photon. Color in
/// 									\c xyz and transparency alpha in \c w. 
/// \param [in]		d_hitSpecClrs		Specular color of the surface hit by each photon. Color
/// 									in \c xyz and index of refraction in \c w. 
/// \param [in]		d_normalG			Geometric normal at photon intersection for each photon. 
/// \param [in]		d_normalS			Shading normal at photon intersection for each photon. 
/// \param [in]		d_randoms1			First uniform random number array. One random number for
/// 									each photon. 
/// \param [in]		d_randoms2			Second uniform random number array. One random number for
/// 									each photon. 
/// \param [in]		d_randoms3			Third uniform random number array. One random number for
/// 									each photon. 
/// \param [out]	d_outIsLastSpecular	Binary 0/1 array. Will contain 1 for photons that underwent
///										a specular reflection/transmission, else 0.
/// \param [out]	d_outIsValid		Binary 0/1 array. Will contain 1 for valid and 0 for
/// 									invalid photons. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_SpawnScatteredPhotons(PhotonData ioPhotons,
											 float4* d_hitDiffClrs, float4* d_hitSpecClrs,
											 float4* d_normalG, float4* d_normalS,
											 float* d_randoms1, float* d_randoms2, float* d_randoms3,
											 uint* d_outIsLastSpecular, uint* d_outIsValid)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < ioPhotons.numPhotons)
	{
		// Read out source photon direction.
		float azimuthal = ioPhotons.d_positions[tid].w;
		float polar = ioPhotons.d_powers[tid].w;
		float3 w_o = -dev_Spherical2Direction(azimuthal, polar);

		float4 clrDiffHit4 = d_hitDiffClrs[tid];
		float3 clrDiffHit = make_float3(clrDiffHit4.x, clrDiffHit4.y, clrDiffHit4.z);
		float4 clrSpecHit4 = d_hitSpecClrs[tid];
		float3 clrSpecHit = make_float3(clrSpecHit4.x, clrSpecHit4.y, clrSpecHit4.z);
		float transAlpha = clrDiffHit4.w;
		//float nTo = clrSpecHit4.w;

		float3 n_s = make_float3(d_normalS[tid]);

		// Calculate n_from/n_to in correct order depending on which direction the photon
		// is traveling.
		/*float indexRefrFromTo;
		if(dot(n_s, w_o) < 0.f)
			indexRefrFromTo = nTo; // Swap: n_to/n_from (leaving medium).
		else
			indexRefrFromTo = 1.f / nTo; // n_from = 1.f !*/

		// I use pseudo random numbers here because in PBR, page 781, "the advantages of
		// low-discrepancy points are mostly lost as more bounces occur".
		float rnd1 = d_randoms1[tid];
		float rnd2 = d_randoms2[tid];
		//float rnd3 = d_randoms3[tid];

		// Choose which BSDF to use. Currently I support:
		//
		// - Lambertian (perfect diffuse)
		// - Perfect specular
		//
		// However not all BSDFs have to be present.
		bool hasDiffuse = dot(clrDiffHit, clrDiffHit) > 0.f && (transAlpha > 0.f);
		bool hasSpecular = dot(clrSpecHit, clrSpecHit) > 0.f;
		bool hasTransmit = dot(clrDiffHit, clrDiffHit) > 0.f && (transAlpha < 1.f);

		float3 n_g = make_float3(d_normalG[tid]);

		// Adding in PDFs from other BxDFs not required as:
		// - For Lambertian, the other are specular and have PDF of zero.
		// - For Specular, adding in not useful (see PBR p. 693).
		// Adding in f() value from other BxDFs not required as:
		// - For Lambertian, the other are specular with F() = 0 w.p. 1 (See PBR p. 693 and p. 428).
		// - For Specular, adding in not useful (see PBR p. 693).
		float pdf = 0.f;
		float3 f, w_i;
		bool bIsSpecReflect = false, bIsSpecTransmit = false;

		// Temporary simplification.
		if(hasDiffuse)
		{
			// Lambertian only.
			// According to Veach, p. 154, particle directions w_i have to be sampled with respect to the
			// geometric normal density |w_i dot N_g|.
			f = dev_SampleDirectionLambertian(w_o, n_g, 
				rnd1, rnd2, clrDiffHit, &w_i, &pdf);
		}
		/*if(!hasSpecular && !hasTransmit)
		{
			// Lambertian only.
			// According to Veach, p. 154, particle directions w_i have to be sampled with respect to the
			// geometric normal density |w_i dot N_g|.
			f = dev_SampleDirectionLambertian(w_o, n_g, 
				rnd1, rnd2, clrDiffHit, &w_i, &pdf);
		}
		else if(!hasDiffuse && !hasTransmit)
		{
			// Specular only.
			f = dev_SampleDirectionSpecReflect(w_o, n_s, 
				rnd1, rnd2, clrSpecHit, &w_i, &pdf);
			bIsSpecReflect = true;
		}
		else if(!hasDiffuse && !hasSpecular)
		{
			// Transmit only.
			f = dev_SampleDirectionSpecTransmit(w_o, n_s, 
				rnd1, rnd2, clrDiffHit, indexRefrFromTo, true, &w_i, &pdf);
			bIsSpecTransmit = true;
		}
		else if(hasDiffuse && hasSpecular && !hasTransmit)
		{
			if(rnd3 < 0.5f)
				f = dev_SampleDirectionLambertian(w_o, n_g, 
					rnd1, rnd2, clrDiffHit, &w_i, &pdf);
			else
			{
				f = dev_SampleDirectionSpecReflect(w_o, n_s, 
					rnd1, rnd2, clrSpecHit, &w_i, &pdf);
				bIsSpecReflect = true;
			}

			pdf *= 0.5f;
		}
		else if(hasDiffuse && hasTransmit && !hasSpecular)
		{
			if(rnd3 < 0.5f)
				f = dev_SampleDirectionLambertian(w_o, n_g, 
					rnd1, rnd2, clrDiffHit * transAlpha, &w_i, &pdf);
			else
			{
				f = dev_SampleDirectionSpecTransmit(w_o, n_s, 
						rnd1, rnd2, clrDiffHit * (1.f - transAlpha), indexRefrFromTo, true, &w_i, &pdf);
				bIsSpecTransmit = true;
			}

			pdf *= 0.5f;
		}
		else if(hasDiffuse && hasSpecular && hasTransmit)
		{
			// NOTE: Determine how to handle internal specular reflections after and before
			//		 specular transmissions (e.g. within a sphere). There would lead to an
			//		 exorbitant PDF as such paths would only be taken by very few photons.
			//		 They could lead to bright spots of different color and can be identified
			//		 by visualizing the photons and scaling the gathering result appropriately.
			if(rnd3 < 0.33333333333f)
				f = dev_SampleDirectionLambertian(w_o, n_g, 
					rnd1, rnd2, clrDiffHit * transAlpha, &w_i, &pdf);
			else if(rnd3 < 0.66666666666f)
			{
				f = dev_SampleDirectionSpecReflect(w_o, n_s, 
					rnd1, rnd2, clrSpecHit, &w_i, &pdf);
				bIsSpecReflect = true;
			}
			else
			{
				f = dev_SampleDirectionSpecTransmit(w_o, n_s, 
						rnd1, rnd2, clrDiffHit * (1.f - transAlpha), indexRefrFromTo, true, &w_i, &pdf);
				bIsSpecTransmit = true;
			}

			pdf *= 0.33333333333f;
		}*/
		else
		{
			// Not supported / nothing to sample.
			pdf = 0.f;
			w_i = -w_o;
			f = make_float3(0.f, 0.f, 0.f);
		}

		// Store new photon direction.
		float2 sphericalNew = dev_Direction2Spherical(w_i);
		float4 oldPos = ioPhotons.d_positions[tid];
		ioPhotons.d_positions[tid] = make_float4(oldPos.x, oldPos.y, oldPos.z, sphericalNew.x);
		float polarNew = sphericalNew.y;

		// Avoid reflection in case w_i and w_o lie in different hemispheres 
		// with respect to n_g. PBR p. 465 or VeachPhD, p. 153.
		if(!bIsSpecTransmit && dot(w_i, n_g) * dot(w_o, n_g) <= 0.f)
			pdf = 0.f;
		// Avoid transmission in case w_i and w_o lie in the same hemisphere.
		if(bIsSpecTransmit && dot(w_i, n_g) * dot(w_o, n_g) > 0.f)
			pdf = 0.f;

		// Store if this was a specular reflection.
		d_outIsLastSpecular[tid] = ((bIsSpecReflect || bIsSpecTransmit) ? 1 : 0);

		// Set flux to zero in case the PDF is zero. Those photons will be eliminated after the
		// next tracing step.
		float alpha = 0.f;

		// See Veach1997, page 154, where the problem using shading normals is described
		// and this weighting formula for particle tracing was developed.
		if(pdf != 0.f)
			alpha =		   fabsf(dot(w_o, n_s)) * fabsf(dot(w_i, n_g)) / 
					(pdf * fabsf(dot(w_o, n_g)));

		// Read out old flux.
		float3 phFlux = make_float3(ioPhotons.d_powers[tid]);

		float3 myFlux = phFlux * f * alpha;
		d_outIsValid[tid] = myFlux.x > 0.f || myFlux.y > 0.f || myFlux.z > 0.f;


		// Update photon power. Leave position alone as it isn't changed.
		ioPhotons.d_powers[tid] = make_float4(myFlux.x, myFlux.y, myFlux.z, polarNew);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ScaleFlux(PhotonData ioPhotons, float scale)
///
/// \brief	Scales the flux component of each photon.
///
///			Note that this kernel is not replaceable with ::mncudaScaleVectorArray(). As the
///			spherical polar coordinate is stored in the w-component of the powers, using this
///			utility function will not work.
///
/// \author	Mathias Neumann
/// \date	August 2010
///
/// \param	ioPhotons	The photon data.
/// \param	scale		The power scale factor. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ScaleFlux(PhotonData ioPhotons, float scale)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < ioPhotons.numPhotons)
	{
		// Do not manipulate w coordinate (spherical polar)!
		float4 phFlux = ioPhotons.d_powers[tid];
		phFlux.x *= scale;
		phFlux.y *= scale;
		phFlux.z *= scale;
		ioPhotons.d_powers[tid] = phFlux;
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


/// Moves light data to constant memory.
extern "C"
void PMUpdateBuildData(const LightData& lights)
{
	mncudaSafeCallNoSync(cudaMemcpyToSymbol("c_Lights", &lights, sizeof(LightData)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_SpawnLightPhotons() kernel call.
extern "C"
void KernelPMSpawnLightPhotons(LightType type, uint photonOffset, uint numToSpawn,
							   float3 worldCenter, float worldRadius,
							   PhotonData& outPhotonSpawn)
{
	MNAssert(outPhotonSpawn.numPhotons == 0);
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numToSpawn, blockSize.x), 1, 1);

	kernel_SpawnLightPhotons<<<gridSize, blockSize>>>(photonOffset, numToSpawn, 
		worldCenter, worldRadius, outPhotonSpawn);
	MNCUDA_CHECKERROR;
	outPhotonSpawn.numPhotons = numToSpawn;
}

/// Wraps kernel_RussianRoulette() kernel call.
extern "C"
void KernelPMRussianRoulette(float* d_randoms, float contProbability,
						     PhotonData& ioPhotons, uint* d_ioIsValid)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(ioPhotons.numPhotons, blockSize.x), 1, 1);

	kernel_RussianRoulette<<<gridSize, blockSize>>>(d_randoms, contProbability, 1.f / contProbability, 
		ioPhotons, d_ioIsValid);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_SpawnScatteredPhotons() kernel call.
extern "C"
void KernelPMSpawnScatteredPhotons(PhotonData& ioPhotons,
								   float4* d_normalsG, float4* d_normalsS, 
								   float4* d_hitDiffClrs, float4* d_hitSpecClrs,
								   float* d_randoms1, float* d_randoms2, float* d_randoms3,
								   uint* d_outIsLastSpecular, uint* d_outIsValid)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(ioPhotons.numPhotons, blockSize.x), 1, 1);

	kernel_SpawnScatteredPhotons<<<gridSize, blockSize>>>(ioPhotons,
		d_hitDiffClrs, d_hitSpecClrs, d_normalsG, d_normalsS, 
		d_randoms1, d_randoms2, d_randoms3, d_outIsLastSpecular, d_outIsValid);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_ScaleFlux() kernel call.
extern "C"
void KernelPMScaleFlux(PhotonData& ioPhotons, float scale)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(ioPhotons.numPhotons, blockSize.x), 1, 1);

	kernel_ScaleFlux<<<gridSize, blockSize>>>(ioPhotons, scale);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////