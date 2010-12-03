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
/// \file	GPU\intersect_dev.h
///
/// \brief	Provides CUDA device functions for intersection tests.
///
///	\note	This file can be included in multiple cu-files!
///
/// \author	Mathias Neumann
/// \date	27.02.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_INTERSECT_DEV_H__
#define __MN_INTERSECT_DEV_H__

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ bool dev_RayTriIntersect(const float3 a, const float3 b, const float3 c,
/// 	const float3 o, const float3 d, float& out_lambda, float& out_bary1, float& out_bary2)
///
/// \brief	Ray triangle intersection test.
/// 		
/// 		Implemented following http://graphics.stanford.edu/papers/i3dkdtree/. 
///
/// \author	Mathias Neumann
/// \date	04.03.2010
///
/// \param	a					First triangle vertex. 
/// \param	b					Second triangle vertex. 
/// \param	c					Third triangle vertex. 
/// \param	o					Ray origin. 
/// \param	d					Ray direction. 
/// \param [out]	out_lambda	Intersection parameter lambda. 
/// \param [out]	out_bary1	Barycentric hit coordinate 1. 
/// \param [out]	out_bary2	Barycentric hit coordinate 2. 
///
/// \return	Returns \c true if an intersection was found, else \c false. Note that the
/// 		intersection parameter might be negative. For ray tracing you'd have to ignore those
///			intersections.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ bool dev_RayTriIntersect(const float3 a, const float3 b, const float3 c, 
										   const float3 o, const float3 d,
										   float& out_lambda, float& out_bary1, float& out_bary2)
{
	float3 edge1 = b - a;
	float3 edge2 = c - a;

	float3 pvec = cross(d, edge2);
	float det = dot(edge1, pvec);
	if(det == 0.f)
		return false;
	float inv_det = 1.0f / det;

	float3 tvec = o - a;
	out_bary1 = dot(tvec, pvec) * inv_det;

	float3 qvec = cross(tvec, edge1);
	out_bary2 = dot(d, qvec) * inv_det;
	out_lambda = dot(edge2, qvec) * inv_det;

	bool hit = (out_bary1 >= 0.0f && out_bary2 >= 0.0f && (out_bary1 + out_bary2) <= 1.0f);
	return hit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ bool dev_RayBoxIntersect(const float3 aabbMin, const float3 aabbMax,
/// 	const float3 rayO, const float3 invRayD, const float tMin, const float tMax,
/// 	float& tMinInter, float& tMaxInter)
///
/// \brief	Axis-aligned bounding box (AABB) ray intersection test.
/// 		
/// 		Implemented following \ref lit_pharr "[Pharr and Humphreys 2004]", p. 179. 
///
/// \author	Mathias Neumann
/// \date	27.02.2010
///
/// \param	aabbMin				AABB minimum vertex. 
/// \param	aabbMax				AABB maximum vertex. 
/// \param	rayO				Ray origin. 
/// \param	invRayD				Inverse ray direction (component-wise inversion).
/// \param	tMin				Current ray parameter segment minimum. 
/// \param	tMax				Current ray parameter segment maximum. 
/// \param [out]	tMinInter	Intersection parameter segment minimum. 
/// \param [out]	tMaxInter	Intersection parameter segment maximum. 
///
/// \return	\c true if there is an intersection, else \c false. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ bool dev_RayBoxIntersect(const float3 aabbMin, const float3 aabbMax,
									const float3 rayO, const float3 invRayD, 
									const float tMin, const float tMax,
									float& tMinInter, float& tMaxInter)
{
	float t0 = tMin;
	float t1 = tMax;

	float* o = (float*)&rayO;
	float* ptMin = (float*)&aabbMin;
	float* ptMax = (float*)&aabbMax;

	bool intersect = true;

	#pragma unroll
	for(uint i=0; i<3; i++)
	{
		// Update interval for ith bounding box slab.
		float val1 = (ptMin[i] - o[i]) * ((float*)&invRayD)[i];
		float val2 = (ptMax[i] - o[i]) * ((float*)&invRayD)[i];

		// Update parametric interval from slab intersection.
		float tNear = val1;
		float tFar = val2;
		if(val1 > val2)
			tNear = val2;
		if(val1 > val2)
			tFar = val1;
		t0 = ((tNear > t0) ? tNear : t0);
		t1 = ((tFar < t1) ? tFar : t1);

		// DO NOT break or return here to avoid divergent branches.
		if(t0 > t1)
			intersect = false;
	}

	tMinInter = t0;
	tMaxInter = t1;

	return intersect;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __MN_INTERSECT_DEV_H__