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
/// \file	GPU\sample_dev.h
///
/// \brief	Provides device functions for sampling.
///
///	\note	This file can be included in multiple cu-files!
///
/// \author	Mathias Neumann
/// \date	09.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_SAMPLE_DEV_H__
#define __MN_SAMPLE_DEV_H__

#include "MNMath.h"
#include "bsdf_dev.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_UniformSampleDisk(float rnd1, float rnd2, float* x, float* y)
///
/// \brief	Simple approach to sample a disk of radius one.
/// 		
/// 		It takes sqrt(rnd1) as radius and 2*PI*rnd2 as angle to create the sample within the
/// 		disk. See \ref lit_pharr "[Pharr and Humphreys 2004]", page 653.
/// 		
/// \warning	Distorts area on the disk. See cited book.
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	rnd1		First uniform random number. 
/// \param	rnd2		Second uniform random number. 
/// \param [out]	x	Generated x coordinate of the sample. 
/// \param [out]	y	Generated y coordinate of the sample. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_UniformSampleDisk(float rnd1, float rnd2, float* x, float* y)
{
	float r = sqrtf(rnd1);
	float theta = 2.f * MN_PI * rnd2;
	*x = r * cosf(theta);
	*y = r * sinf(theta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_ConcentricSampleDisk(float rnd1, float rnd2, float* x,
/// 	float* y)
///
/// \brief	Performs a concentric mapping from the unit square to the unit circle.
/// 		
/// 		Avoids the distortion problem of dev_UniformSampleDisk(). Wedges of the square are
/// 		mapped to slices of the disc. See \ref lit_pharr "[Pharr and Humphreys 2004]",
/// 		page 653. 
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	rnd1		First uniform random number. 
/// \param	rnd2		Second uniform random number. 
/// \param [out]	x	Generated x coordinate of the sample. 
/// \param [out]	y	Generated y coordinate of the sample. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_ConcentricSampleDisk(float rnd1, float rnd2, float* x, float* y)
{
	float r, theta;

	// Map uniform random numbers to [-1, 1]^2.
	float sx = 2.f * rnd1 - 1.f;
	float sy = 2.f * rnd2 - 1.f;

	// Handle degeneracy at the origin
	if(sx == 0.0f && sy == 0.0f) 
	{
		*x = 0.0f;
		*y = 0.0f;
		return;
	}

	// Map square to (r, theta).
	if(sx >= -sy) 
	{
		if(sx > sy) 
		{
			// Handle first region of disk.
			r = sx;
			if(sy > 0.0f)
				theta = sy / r;
			else
				theta = 8.0f + sy / r;
		}
		else 
		{
			// Handle second region of disk.
			r = sy;
			theta = 2.0f - sx/r;
		}
	}
	else 
	{
		if(sx <= sy) 
		{
			// Handle third region of disk.
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else 
		{
			// Handle fourth region of disk.
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}

	theta *= MN_PI / 4.f;

	*x = r * cosf(theta);
	*y = r * sinf(theta);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_SampleHemisphereMalley(float rnd1, float rnd2)
///
/// \brief	Samples the hemisphere defined by the standard normal n = (0, 0, 1).
/// 		
/// 		This is useful if generated sample direction should be more likely pointing to the
/// 		top of the hemisphere. Mathematically, we sample directions w from a PDF: p(w) =
/// 		alpha * cos(theta). See \ref lit_pharr "[Pharr and Humphreys 2004]", page 656.
/// 		
/// 		We use Malley's method to get the direction: First we choose points uniformly on the
/// 		disk. Then we generate directions by projecting the points to the hemisphere. For a
/// 		proof, see cited book. 
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	rnd1	First uniform random number. 
/// \param	rnd2	Second uniform random number. 
///
/// \return	The sampled direction in the (0, 0, 1)-hemisphere. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleHemisphereMalley(float rnd1, float rnd2)
{
	float3 dir;

	// As pointed out in PBR, page 657, we can just use the simple uniform sampling method,
	// which should work as well as the concentric sample method here. I'll do that because
	// the concentric sample method contains lots of branching...
	dev_ConcentricSampleDisk(rnd1, rnd2, &dir.x, &dir.y);

	// Project to hemisphere.
	dir.z = sqrtf(fmaxf(0.f, 1.f - dir.x*dir.x - dir.y*dir.y));

	return dir;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_SampleSphereUniform(float rnd1, float rnd2)
///
/// \brief	Uniformly samples a direction on the full sphere, that is with PDF 1/(4*PI). 
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \param	rnd1	First uniform random number.
/// \param	rnd2	Second uniform random number.
///
/// \return	Sampled direction. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleSphereUniform(float rnd1, float rnd2)
{
	//	The result is normalized since
	//
	//		sqrt(r^2(sin^2 + cos^2) + z^2) = sqrt((1 - z^2) + z^2) = 1
	//	
	//	for |z| < 1 and (x, y, z) = (0, 0, |z|) for |z| = 1.
	//	
	//	See Pharr and Humphreys, "Physically based rendering", p. 650.
	float z = 1.f - 2.f * rnd1;
	float r = sqrtf(fmaxf(0.f, 1.f - z*z));
	float phi = 2.f * MN_PI * rnd2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);

	return make_float3(x, y, z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_SampleSphereUniformPDF()
///
/// \brief	Computes PDF for uniform sphere sampling with dev_SampleSphereUniform().
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \return	The PDF. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_SampleSphereUniformPDF()
{
	// 1 / (4*PI) = 1/2 * 1/(2*PI).
	return 0.5f * MN_INV_TWOPI;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_SampleHemisphereCosine(float3 normal, float rnd1, float rnd2,
/// 	float* outPDF)
///
/// \brief	Samples the hemisphere defined by given normal using sampling according to the PDF of
/// 		the cosine function. 
///
/// \author	Mathias Neumann
/// \date	10.08.2010
///
/// \param	normal			The normal defining the hemisphere (normalized). 
/// \param	rnd1			First uniform random number. 
/// \param	rnd2			Second uniform random number. 
/// \param [out]	outPDF	The PDF for sampling the returned direction. 
///
/// \return	The sampled direction. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleHemisphereCosine(float3 normal, float rnd1, float rnd2,
													float* outPDF)
{
	// Cosine-sample the local hemisphere.
	float3 wi = dev_SampleHemisphereMalley(rnd1, rnd2);

	// Method from Jensen:
	/*float phi = 2*MN_PI*rnd2;
	float theta = asinf(sqrtf(rnd1));
	float3 wi = dev_Spherical2Direction(phi, theta);*/

	// The returned direction lies in the hemisphere of the standard normal (0, 0, 1).
	// We have to transform it into the hemisphere of *normal*.
	float3 s, t;
	dev_BuildCoordSystem(normal, &s, &t);
	wi = normalize(dev_BSDFLocal2World(wi, s, t, normal));

	*outPDF = fabsf(dot(normal, wi)) * MN_INV_PI;

	return wi;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float dev_SampleDirectionLambertianPDF(const float3& wo,
/// 	const float3& normal, const float3& wi)
///
/// \brief	Computes PDF for sampling a direction \a wi using lambertian reflection, i.e. by
/// 		cosine sampling the hemisphere for the given normal. 
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \param	wo		The given outgoing direction (pointing away). 
/// \param	normal	The normal (normalized). 
/// \param	wi		The sampled incoming direction (pointing away). 
///
/// \return	The PDF for sampling \a wi from Lambertian BRDF. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_SampleDirectionLambertianPDF(const float3& wo, const float3& normal, const float3& wi)
{
	// Check if wo and wi are in the same hemisphere.
	if(dot(normal, wo) * dot(normal, wi) > 0.f)
		return fabsf(dot(normal, wi)) * MN_INV_PI;
	else
		return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_SampleDirectionLambertian(float3 wo, float3 normal,
/// 	float rnd1, float rnd2, float3 clrDiffHit, float3* out_wi, float* outPDF)
///
/// \brief	Sample a reflection direction using Lambertian reflection. 
///
/// \author	Mathias Neumann
/// \date	10.04.2010
///
/// \param	wo				The outgoing direction (pointing away). 
/// \param	normal			The normal (normalized). Type depends on what to do with the sampled
/// 						direction. For particle tracing, pass geometric normal (see
/// 						\ref lit_veach "[Veach 1997]", p. 153), for raytracing, pass shading 
///							normal. 
/// \param	rnd1			First uniform random number. 
/// \param	rnd2			Second uniform random number. 
/// \param	clrDiffHit		The diffuse color material at hit. 
/// \param [out]	out_wi	Sampled incoming direction (pointing away). 
/// \param [out]	outPDF	The PDF for sampling \a out_wi. 
///
/// \return	BRDF value.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleDirectionLambertian(float3 wo, float3 normal, 
											    float rnd1, float rnd2, float3 clrDiffHit,
											    float3* out_wi, float* outPDF)
{
	*out_wi = dev_SampleHemisphereCosine(normal, rnd1, rnd2, outPDF);

	// Now check whether normal and wo lie in same or different hemisphere. In case wo lies in the
	// wrong hemisphere, swap wi.
	if(dot(normal, wo) < 0.f)
		*out_wi *= -1.f;

	// Return R/Pi, where R is the fraction of light scattered (diffuse reflectance), see PBR, p. 435.
	return clrDiffHit * MN_INV_PI;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_SampleDirectionSpecReflectPDF(const float3& wo, const float3& normal,
/// 	const float3& wi)
///
/// \brief	Computes PDF for sampling a specular reflection BRDF. 
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \param	wo		The given outgoing direction (pointing away). 
/// \param	normal	The normal. 
/// \param	wi		The sampled incoming direction (pointing away). 
///
/// \return	The PDF. It is zero in all cases as there is zero probability to choose the matching
/// 		\a wi for a given \a wo. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_SampleDirectionSpecReflectPDF(const float3& wo, const float3& normal, const float3& wi)
{
	return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_SampleDirectionSpecReflect(float3 wo, float3 normal,
/// 	float rnd1, float rnd2, float3 clrSpecHit, float3* out_wi, float* outPDF)
///
/// \brief	Samples a direction from the specular reflection BRDF.
/// 		
/// 		There is only one single direction, and the PDF for sampling this direction will be
/// 		\c 1.f. 
///
/// \author	Mathias Neumann
/// \date	23.06.2010
///
/// \todo	Implement Fresnel term. I currently use the approximation from 
///			\ref lit_jensen "[Jensen 2001]", page 22.
///
/// \param	wo				The outgoing direction (pointing away). 
/// \param	normal			The normal (normalized). 
/// \param	rnd1			First uniform random number. Unused here. 
/// \param	rnd2			Second uniform random number. Unused here. 
/// \param	clrSpecHit		The specular material color at hit. 
/// \param [out]	out_wi	Sampled incoming direction (pointing away). 
/// \param [out]	outPDF	The PDF for sampling \a outDir, i.e. \c 1.f. 
///
/// \return	BRDF value.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleDirectionSpecReflect(float3 wo, float3 normal, 
											     float rnd1, float rnd2, float3 clrSpecHit,
											     float3* out_wi, float* outPDF)
{
	// Reflect at normal.
	// R = 2*(normal dot wo)*normal - wo
	float cosThetaOut = dot(normal, wo);
	*out_wi = 2.f*cosThetaOut*normal - wo;
	
	// We do no monte carlo sampling since there is only one single direction for wo. Therefore
	// The probability density for sampling this single direction has to be one.
	// NOTE: This is different from just picking a random direction as with other BSDFs. Here we
	//		 directly pick the single direction.
	*outPDF = 1.f;

	// Use the approximation from Jensen, p. 22.
	float fresnel = 2.f;

	// The f() value is determined by scaling by 1.f / cos(theta_in), see PBR p. 428, as
	// else the scattering equation would lead to incorrect results.
	return fresnel * clrSpecHit / fabsf(dot(normal, *out_wi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float dev_SampleDirectionSpecTransmitPDF(const float3& wo, const float3& normal,
/// 	const float3& wi)
///
/// \brief	Computes PDF for sampling a specular transmission BTDF. 
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \param	wo		The given outgoing direction (pointing away). 
/// \param	normal	The normal (normalized). 
/// \param	wi		The sampled incoming direction (pointing away). 
///
/// \return	The PDF. It is zero in all cases as there is zero probability to choose the matching
/// 		\a wi for a given \a wo. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float dev_SampleDirectionSpecTransmitPDF(const float3& wo, const float3& normal, const float3& wi)
{
	return 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_SampleDirectionSpecTransmit(float3 wo, float3 normal, float rnd1,
/// 	float rnd2, float3 clrTransmit, float indexRefrFromTo, bool adjoint, float3* out_wi,
/// 	float* outPDF)
///
/// \brief	Samples a direction from the specular reflection BRDF. There is only one single
/// 		direction, and the PDF will be 1.f. 
///
/// \author	Mathias Neumann
/// \date	23.06.2010
///
/// \param	wo				The outgoing direction (pointing away). 
/// \param	normal			The normal (normalized). 
/// \param	rnd1			First uniform random number. Unused here. 
/// \param	rnd2			Second uniform random number. Unused here. 
/// \param	clrTransmit		The specular material color at hit.
/// \param	indexRefrFromTo	The index of refraction ratio from/to. 
/// \param	adjoint			Pass \c true when sampling for light particles, else \c false.
/// \param [out]	out_wi	Sampled incoming direction (pointing away). 
/// \param [out]	outPDF	The PDF for sampling \a outDir, \c 1.f. 
///
/// \return	BRDF value.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleDirectionSpecTransmit(float3 wo, float3 normal, 
											     float rnd1, float rnd2, float3 clrTransmit,
												 float indexRefrFromTo, bool adjoint,
											     float3* out_wi, float* outPDF)
{
	// Refract wo.
	bool isTotalReflect = false;
	float3 n_trans = normal * ((dot(normal, wo) < 0.f) ? -1.f : 1.f);
	*out_wi = dev_Refract(wo, n_trans, indexRefrFromTo, &isTotalReflect);

	if(isTotalReflect)
	{
		*outPDF = 0.f;
		return make_float3(0.f, 0.f, 0.f);
	}
	
	// We do no monte carlo sampling since there is only one single direction for inDir. Therefore
	// The probability density for sampling this single direction has to be one.
	// NOTE: This is different from just picking a random direction as with other BSDFs. Here we
	//		 directly pick the single direction.
	*outPDF = 1.f;

	// The f() value is determined by scaling by 1.f / cos(theta_in), see PBR p. 428, as
	// else the scattering equation would lead to incorrect results.
	float3 f = clrTransmit / fabsf(dot(n_trans, *out_wi));

	// See Veach PhD p. 147. 
	if(!adjoint) // Scale with n_t^2/n_i^2 if not adjoint BSDF.
		f *= (1.f / (indexRefrFromTo*indexRefrFromTo));

	return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_SampleGeneralDisc(float3 center, float3 normal, float radius,
/// 	float rnd1, float rnd2)
///
/// \brief	Samples a point on a general disc using concentric disc sampling.
///
/// \author	Mathias Neumann
/// \date	12.06.2010
/// \see	dev_ConcentricSampleDisk()
///
/// \param	center	The center of the disc. 
/// \param	normal	The normal of the disc. 
/// \param	radius	The radius of the disc. 
/// \param	rnd1	First uniform random number. 
/// \param	rnd2	Second uniform random number. 
///
/// \return	A sampled point on the given disc. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_SampleGeneralDisc(float3 center, float3 normal, float radius, 
										float rnd1, float rnd2)
{
	// Build coordinate system using normal. See MNVector3.
	// Resulting vectors are normalized!
	float3 v1, v2;
	dev_BuildCoordSystem(normal, &v1, &v2);

	// Sample from disc.
	float d1, d2;
	dev_ConcentricSampleDisk(rnd1, rnd2, &d1, &d2);

	return center + radius * (d1 * v1 + d2* v2);
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


#endif // __MN_SAMPLE_DEV_H__