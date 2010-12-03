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
/// \file	GPU\bsdf_dev.h
///
/// \brief	Declares CUDA device functions for BSDF handling.
///
///	\note	This file can be included in multiple cu-files!
///
/// \author	Mathias Neumann
/// \date	09.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_BSDF_DEV_H__
#define __MN_BSDF_DEV_H__

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ float3 dev_BSDFWorld2Local(float3 v, float3 sWorld, float3 tWorld,
/// 	float3 nWorld)
///
/// \brief	Transforms the given world space vector v into the local coordinate system.
/// 		
/// 		The \e local coordinate system uses (0, 0, 1) as normal, (1, 0, 0) as primary tangent
/// 		and (0, 1, 0) as secondary tangent. 
///
/// \author	Mathias Neumann
/// \date	10.04.2010
///
/// \param	v		The world vector to transform. 
/// \param	sWorld	Primary tangent vector in world space. 
/// \param	tWorld	Secondary tangend vector in world space. 
/// \param	nWorld	Normal vector in world space. 
///
/// \return	The transformed vector in local space. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_BSDFWorld2Local(float3 v, float3 sWorld, float3 tWorld, float3 nWorld)
{
	// Here the world space basis vectors have to be placed in the rows of the
	// transformation matrix.
	return make_float3(dot(v, sWorld), dot(v, tWorld), dot(v, nWorld));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_BSDFLocal2World(float3 v, float3 sWorld, float3 tWorld,
/// 	float3 nWorld)
///
/// \brief	Transforms the given local vector into world space.
///
///			The given world space coordinate system vectors are assumed to be normalized.
///
/// \author	Mathias Neumann
/// \date	10.04.2010
///
/// \param	v		The local vector to transform.
/// \param	sWorld	Primary tangent vector in world space. 
/// \param	tWorld	Secondary tangend vector in world space.
/// \param	nWorld	Normal vector in world space.
///
/// \return	Transformed vector in world space. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_BSDFLocal2World(float3 v, float3 sWorld, float3 tWorld, float3 nWorld)
{
	// Here the world space basis vectors have to be placed in the columns of the
	// transformation matrix. That's possible because the world space basis vectors
	// form an orthonormal basis, hence the inverse transformation matrix (local -> world) is
	// the transpose of the original transformation matrix (world -> local).
	return make_float3(sWorld.x * v.x + tWorld.x * v.y + nWorld.x * v.z,
					   sWorld.y * v.x + tWorld.y * v.y + nWorld.y * v.z,
					   sWorld.z * v.x + tWorld.z * v.y + nWorld.z * v.z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_GetTangentSpace(float3 pA, float3 pB, float3 pC, float3* sWorld,
/// 	float3* tWorld, float3* nWorld)
///
/// \brief	Constructs the tangent space for the given triangle by using the triangle vertices.
///
/// 		The normal is generated by taking the cross product of the triangle edges and so on. 
///
/// \author	Mathias Neumann
/// \date	10.04.2010
///
/// \param	pA				First triangle vertex. 
/// \param	pB				Second triangle vertex.
/// \param	pC				Third triangle vertex.
/// \param [out]	sWorld	Primary tangent vector in world space. May not be \c NULL.
/// \param [out]	tWorld	Secondary tangent vector in world space. May not be \c NULL.
/// \param [out]	nWorld	Normal vector in world space. May not be \c NULL.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_GetTangentSpace(float3 pA, float3 pB, float3 pC, 
										   float3* sWorld, float3* tWorld, float3* nWorld)
{
	// Construct side vectors.
	float3 v1 = normalize(pB - pA), v2 = normalize(pC - pA);

	// Get normal.
	*nWorld = normalize(cross(v1, v2));

	// Use v1 as first tangent.
	*sWorld = v1;

	// Compute second tangent.
	*tWorld = cross(*nWorld, *sWorld);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ void dev_BuildCoordSystem(float3 v, float3* outS, float3* outT)
///
/// \brief	Builds a coordinate system from the given normalized vector.
///
///			This method constructs a random coordinate system by choosing two normalized
///			vectors so that all three vectors are orthonormal. The resulting coordinate system
///			is unique up to a rotation around \a v.
///
/// \author	Mathias Neumann
/// \date	13.06.2010
///
/// \param	v				The \em normalized vector.
/// \param [out]	outS	First tangent vector, normalized. May not be \c NULL.
/// \param [out]	outT	Second tangent vector, normalized. May not be \c NULL.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_BuildCoordSystem(float3 v, float3* outS, float3* outT)
{
	if(fabsf(v.x) > fabsf(v.y))
	{
		// Here we zero out v.y for *outS.
		float invLengthS = rsqrtf(v.x * v.x + v.z * v.z);
		// Swap remaining components and negate one.
		*outS = make_float3(-v.z * invLengthS, 0.f, v.x * invLengthS);
		// Hence dot(v, *outS) = -v.z*v.x/invL + v.x*v.z/invL = 0 and length(*outS) = 1.
	}
	else
	{
		// Here we zero out v.x for *outS.
		float invLengthS = rsqrtf(v.y * v.y + v.z * v.z);
		// Swap remaining components and negate one.
		*outS = make_float3(0.f, v.z * invLengthS, -v.y * invLengthS);
		// Hence dot(v, *outS) = -v.z*v.x/invL + v.x*v.z/invL = 0 and length(*outS) = 1.
	}
	*outT = cross(v, *outS);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__device__ float3 dev_Refract(const float3& dirFrom, const float3& normal,
/// 	const float& n_from_By_n_to, bool* outIsTotalReflect)
///
/// \brief	Computes refracted direction when moving from medium 1 (\c n_from) to medium 2 (\c n_to)
/// 		using Snell's law:
/// 		
/// 		\code n_from * sin(theta_from) = n_to * sin(theta_to) \endcode
///
/// 		The used formula for the refracted direction is taken from 
///			\ref lit_jensen "[Jensen 2001]", page 24. 
///
/// \author	Mathias Neumann
/// \date	17.06.2010
///
/// \param	dirFrom						Normalized incoming direction (pointing away from hit). 
/// \param	normal						The normal at which the ray/particle hit the surface. 
/// \param	n_from_By_n_to				\c n_from / \c n_to (index of refraction ratio from/to).
/// \param [out]	outIsTotalReflect	This value is set to \c true when total reflection occurred.
///										May not be \c NULL.
///
/// \return	Refracted direction, normalized. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 dev_Refract(const float3& dirFrom, const float3& normal, const float& n_from_By_n_to,
							  bool* outIsTotalReflect)
{
	float ratio = n_from_By_n_to;
	float cosFrom = dot(dirFrom, normal);
	float3 refr = (- ratio) * (dirFrom - cosFrom*normal);

	float beta = sqrtf(fmaxf(0.f, 1.f - (ratio*ratio)*(1.f - cosFrom*cosFrom)));
	refr = refr - beta*normal;

	// If the squared value of sin(theta_to) = n_from/n_to*sin(theta_from) is larger than 1,
	// no transmission is possible (PBR p. 435).
	float sinTo2 = ratio*ratio*(1.f - cosFrom*cosFrom);
	*outIsTotalReflect = (sinTo2 > 1.f);

	return normalize(refr);
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __MN_BSDF_DEV_H__