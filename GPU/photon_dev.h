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
/// \file	GPU\photon_dev.h
///
/// \brief	Provides device functions for photon loading and storing.
///
///	\note	This file can be included in multiple cu-files!
///
/// \author	Mathias Neumann
/// \date	09.04.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_PHOTON_DEV_H__
#define __MN_PHOTON_DEV_H__

#include "mncudautil_dev.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \DEVICEFN
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_PhotonStore(const PhotonData& photons, uint idx, float3 phPos,
/// 	float3 phDir, float3 phFlux)
///
/// \brief	Adds a photon to the given photon data structure. 
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	photons	The photon data structure to update. 
/// \param	idx		Zero-based target index for photon. 
/// \param	phPos	Photon position. 
/// \param	phDir	Photon direction (normalized). 
/// \param	phFlux	Photon flux. 
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_PhotonStore(const PhotonData& photons, uint idx,
							    float3 phPos, float3 phDir, float3 phFlux)
{
	float2 spherical = dev_Direction2Spherical(phDir);

	// Position
	photons.d_positions[idx] = make_float4(phPos, spherical.x);

	// Power
	photons.d_powers[idx] = make_float4(phFlux, spherical.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	inline __device__ void dev_PhotonLoad(const PhotonData& photons, uint idx, float3& phPos,
/// 	float3& phDir, float3& phFlux)
///
/// \brief	Loads a photon from the given photon data structure. 
///
/// \author	Mathias Neumann
/// \date	09.04.2010
///
/// \param	photons			Source photon data structure.
/// \param	idx				Zero-based source index of photon.
/// \param [out]	phPos	Loaded photon position.
/// \param [out]	phDir	Loaded photon direction (normalized).
/// \param [out]	phFlux	Loaded photon flux.
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void dev_PhotonLoad(const PhotonData& photons, uint idx,
							   float3& phPos, float3& phDir, float3& phFlux)
{
	// Position
	float4 tmp = photons.d_positions[idx];
	phPos = make_float3(tmp);
	float azimuthal = tmp.w;

	// Power
	tmp = photons.d_powers[idx];
	phFlux = make_float3(tmp);
	float polar = tmp.w;

	// Direction
	phDir = dev_Spherical2Direction(azimuthal, polar);
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __MN_PHOTON_DEV_H__