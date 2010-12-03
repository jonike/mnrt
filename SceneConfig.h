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
/// \file	MNRT\SceneConfig.h
///
/// \brief	Declares the SceneConfig class. 
/// \author	Mathias Neumann
/// \date	08.10.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MN_SCENECONFIG_H_
#define _MN_SCENECONFIG_H_

#pragma once

#include <string>
#include "KernelDefs.h"
#include "Geometry/MNBBox.h"
#include "CameraModel.h"

class BasicScene;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	SceneConfig
///
/// \brief	Scene specific configuration.
///
///			Here all settings related to the current scene are combined. Currently there is no
///			way to store these settings in some persistent way. Instead I hardcoded basic
///			configurations for test scenes.
///
/// \todo	Implement way to make scene specific configuration persistent.
/// \todo	Improve automatic selection of parameters.
///
/// \author	Mathias Neumann
/// \date	08.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class SceneConfig
{
public:
	SceneConfig(void);
	~SceneConfig(void);

private:
	// Scene path (lower case).
	std::string m_strScenePath;
	// Loaded scene.
	BasicScene* m_pScene;
	// Light information.
	LightData m_Light;
	// Camera information.
	CameraModel* m_pCamera;
	// Has the scene a specular component?
	bool m_bHasSpecular;
	// Ray epsilon.
	float m_fRayEpsilon;

	// Global photon map target photon count. Set to -1 to use global settings.
	int m_nTargetCountGlobal;
	// Caustics photon map target photon count. Set to -1 to use global settings.
	int m_nTargetCountCaustics;
	// Global maximum for photon kNN query radius.
	float m_fRadiusPMapMax;
	// k-means-algorithm search radius maximum (Wang).
	float m_fRadiusWangKMeansMax;
	// Sample interpolation (Wang) for Final Gathering. Maximum search radius.
	float m_fRadiusWangInterpolMax;
	// Number of initial samples for final gathering interpolation. Final number of
	// samples might differ since samples might be eliminated during selection process.
	uint m_nWangInitialSamples;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool LoadFromFile(const std::string& scenePath)
	///
	/// \brief	Loads a given scene from file.
	/// 		
	/// 		The scene specified by \a scenePath is loaded, currently using AssimpScene only.
	/// 		After that, default parameters are chosen to provide some basic configuration for
	/// 		unknown scenes. Finally the scene path is tested against some known scene names. In
	/// 		case of a match, hardcoded parameters are assigned. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	scenePath	Full path of the scene file. 
	///
	/// \return	true if it succeeds, false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool LoadFromFile(const std::string& scenePath);

	/// Returns the scene path for the loaded scene.
	const std::string& GetScenePath() const { return m_strScenePath; }
	/// Returns the loaded scene.
	BasicScene* GetScene() { return m_pScene; }
	/// Returns the scene's bounding box.
	MNBBox GetSceneBounds() const;
	/// Returns the light for the given scene.
	LightData& GetLight() { return m_Light; }
	/// Returns the camera for the given scene.
	CameraModel* GetCamera() { return m_pCamera; }
	/// \brief	Sets whether this scene has a specular material component.
	///
	///			This is used mainly to avoid tracing photons for the caustics photon
	///			map for scenes without specular materials.
	void SetHasSpecular(bool r) { m_bHasSpecular = r; }
	/// Returns whether this scene has a specular material component.
	bool GetHasSpecular() const { return m_bHasSpecular; }
	/// \brief	Ray epsilon for ray tracing.
	/// 		
	/// 		Small constant that is used to avoid finding an intersection on the same surface
	/// 		region the ray origin is located. These intersections have a very small distance from
	/// 		the ray origin. This constant describes the minimum distance for an intersection to
	/// 		be valid. Do not choose it too small. Else we get bad intersections when tracing
	/// 		shadow rays or final gather rays.
	void SetRayEpsilon(float f) { m_fRayEpsilon = f; }
	/// Returns the current ray epsilon value.
	float GetRayEpsilon() const { return m_fRayEpsilon; }

	/// Sets global photon map photon target count. Set to -1 to use global settings.
	void SetTargetCountGlobal(int c) { m_nTargetCountGlobal = c; }
	/// Returns current global photon map photon target count. Will be negative to use global settings.
	int GetTargetCountGlobal() const { return m_nTargetCountGlobal; }
	/// Sets caustics photon map photon target count. Set to -1 to use global settings.
	void SetTargetCountCaustics(int c) { m_nTargetCountCaustics = c; }
	/// Returns current caustics photon map photon target count. Will be negative to use global settings.
	int GetTargetCountCaustics() const { return m_nTargetCountCaustics; }
	/// Sets the global maximum for the query radius for density estimation kNN searches.
	void SetRadiusPMapMax(float r) { m_fRadiusPMapMax = r; }
	/// Returns the global maximum for the query radius for density estimation kNN searches.
	float GetRadiusPMapMax() const { return m_fRadiusPMapMax; }
	/// Sets the maximum for the search radius for samles to use for sample interpolation.
	void SetRadiusWangInterpolMax(float r) { m_fRadiusWangInterpolMax = r; }
	/// Returns the maximum for the search radius for samles to use for sample interpolation.
	float GetRadiusWangInterpolMax() const { return m_fRadiusWangInterpolMax; }
	/// Sets the maximum query radius for cluster search during k-means algorithm.
	void SetRadiusWangKMeansMax(float r) { m_fRadiusWangKMeansMax = r; }
	/// Returns the maximum query radius for cluster search during k-means algorithm.
	float GetRadiusWangKMeansMax() const { return m_fRadiusWangKMeansMax; }
	/// \brief	Sets the number of initial samples for adaptive sample seeding.
	///
	///			Final number of samples might differ since samples might get eliminated 
	///			during selection process.
	void SetWangInitialSamples(uint n) { m_nWangInitialSamples = n; }
	/// Returns the number of initial samples for adaptive sample seeding.
	uint GetWangInitialSamples() const { return m_nWangInitialSamples; }

private:
	void AssignDefault();
	bool AssignFromStatic();
};

#endif // _MN_SCENECONFIG_H_