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
/// \file	MNRT\MNRTSettings.h
///
/// \brief	Declares the MNRTSettings class. 
/// \author	Mathias Neumann
/// \date	08.10.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_MNRTSETTINGS_H__
#define __MN_MNRTSETTINGS_H__

#pragma once

#include <string>
#include "MNMath.h"

class MNBBox;

/// MNRT view mode.
enum MNRTViewMode
{
	/// Shows the global illumination result (default).
	MNRTView_Result = 0,
	/// Visualizes the initial sample distribution.
	MNRTView_InitialSamples,
	/// Shows the clusters using "flat shading".
	MNRTView_Cluster,
	/// Shows the cluster centers as dots on raytraced image.
	MNRTView_ClusterCenters,
};

/// Photon mapping mode.
enum PhotonMapMode
{
	/// No photon mapping.
	PhotonMap_Disabled = 0,
	/// Visualizes the photons stored.
	PhotonMap_Visualize,
	/// Final gathering for all shading points. Very expensive.
	PhotonMap_FullFinalGather,
	/// Final gathering only at adaptively seeded sample points. Best fit (no interpolation).
	PhotonMap_AdaptiveSamplesBestFit,
	/// Final gathering only at adaptively seeded sample points. Interpolation (Wang).
	PhotonMap_AdaptiveSamplesWang,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNRTSettings
///
/// \brief	Combines all global MNRT settings.
///
///			This class does not provide any means to load or store settings. However it is
///			laid out for subclassing. Subclasses can use the LoadItem() and SaveItem() methods
///			to define how the items are stored. The default implementation of these methods does
///			nothing but assigning default values.
///
/// \warning Do not use this class before calling Load(). Else all settings are undefined.
///
/// \author	Mathias Neumann
/// \date	08.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNRTSettings
{
public:
	/// Default constructor.
	MNRTSettings();
	virtual ~MNRTSettings();

// Attributes
private:
	// View mode.
	MNRTViewMode m_ViewMode;
	// Whether data structures should be rebuilt every frame.
	bool m_bDynamicScene;

	// Direct lighting by raytracing enabled?
	bool m_bEnableDirectRT;
	// Whether to traced shadow rays.
	bool m_bEnableShadowRays;
	// Specular reflection enabled?
	bool m_bEnableSpecReflect;
	// Specular transmission enabled?
	bool m_bEnableSpecTransmit;
	// Area light samples X (to enable stratification).
	uint m_nAreaLightSamplesX;
	// Area light samples Y (to enable stratification).
	uint m_nAreaLightSamplesY;

	// Photon mapping mode.
	PhotonMapMode m_PhotonMapMode;
	// Maximum of photon bounces recorded during photon tracing.
	uint m_nMaxPhotonBounces;
	// Global photon map target photon count.
	uint m_nTargetCountGlobal;
	// Caustics photon map target photon count.
	uint m_nTargetCountCaustics;
	// Global photon map k in kNN search.
	uint m_nKinKNNSearchGlobal;
	// Caustics photon map k in kNN search.
	uint m_nKinKNNSearchCaustics;
	// KNN refinement iterations.
	uint m_nKNNRefineIters;

	// Final gather rays X (to enable stratification).
	uint m_nFinalGatherRaysX;
	// Final gather rays Y (to enable stratification).
	uint m_nFinalGatherRaysY;
	// Geometric variation alpha used in adaptive sample seeding (Wang), for normalized
	// scene geometry. Should be in range 0.1 - 0.5. Larger = more influence for position.
	float m_fGeoVarAlpha;
	// Geometric variation propagation factor (from leafs to root). Set to 0.f to disable propagation.
	float m_fGeoVarPropagation;
	// Maximum of k-Means iterations for initial sample refinement.
	uint m_nKMeansItersMax;
	// Use illumination cuts?
	bool m_bUseIllumCuts;
	// Use all leafs as cut nodes simplification?
	bool m_bICutUseLeafs;
	// Photon map node level for E_min computation.
	uint m_nICutLevelEmin;
	// Number of illumination cut refinement iterations.
	uint m_nICutRefineIters;
	// Required accuracy of estimated irradiance for final cut nodes.
	float m_fICutAccuracy;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Load()
	///
	/// \brief	Loads the settings from some kind of configuration.
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	/// \see	LoadItem()
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Load();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Save()
	///
	/// \brief	Saves the settings to some kind of configuration.
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	/// \see	SaveItem()
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Save();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool LoadItem(const std::string& strKey, bool* val, bool defaultVal)
	///
	/// \brief	Loads a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey		The key. 
	/// \param [out]	val	The loaded value of key. 
	/// \param	defaultVal	Default value, used if key not found. 
	///
	/// \return	\c true if key was found, else \c false. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool LoadItem(const std::string& strKey, bool* val, bool defaultVal) { *val = defaultVal; return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool SaveItem(const std::string& strKey, bool val)
	///
	/// \brief	Saves a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey	The key. 
	/// \param	val		The value for key. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool SaveItem(const std::string& strKey, bool val) { return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool LoadItem(const std::string& strKey, long* val, long defaultVal)
	///
	/// \brief	Loads a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey		The key. 
	/// \param [out]	val	The loaded value of key. 
	/// \param	defaultVal	Default value, used if key not found. 
	///
	/// \return	\c true if key was found, else \c false. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool LoadItem(const std::string& strKey, long* val, long defaultVal) { *val = defaultVal; return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool SaveItem(const std::string& strKey, long val)
	///
	/// \brief	Saves a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey	The key. 
	/// \param	val		The value for key. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool SaveItem(const std::string& strKey, long val) { return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool LoadItem(const std::string& strKey, unsigned int* val,
	/// 	unsigned int defaultVal)
	///
	/// \brief	Loads a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey		The key. 
	/// \param [out]	val	The loaded value of key. 
	/// \param	defaultVal	Default value, used if key not found. 
	///
	/// \return	\c true if key was found, else \c false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool LoadItem(const std::string& strKey, unsigned int* val, unsigned int defaultVal) { *val = defaultVal; return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool SaveItem(const std::string& strKey, unsigned int val)
	///
	/// \brief	Saves a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey	The key. 
	/// \param	val		The value for key. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool SaveItem(const std::string& strKey, unsigned int val) { return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool LoadItem(const std::string& strKey, double* val, double defaultVal)
	///
	/// \brief	Loads a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey		The key. 
	/// \param [out]	val	The loaded value of key. 
	/// \param	defaultVal	Default value, used if key not found. 
	///
	/// \return	\c true if key was found, else \c false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool LoadItem(const std::string& strKey, double* val, double defaultVal) { *val = defaultVal; return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool SaveItem(const std::string& strKey, double val)
	///
	/// \brief	Saves a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey	The key. 
	/// \param	val		The value for key. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool SaveItem(const std::string& strKey, double val) { return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool LoadItem(const std::string& strKey, float* val, float defaultVal)
	///
	/// \brief	Loads a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey		The key. 
	/// \param [out]	val	The loaded value of key. 
	/// \param	defaultVal	Default value, used if key not found. 
	///
	/// \return	\c true if key was found, else \c false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool LoadItem(const std::string& strKey, float* val, float defaultVal) { *val = defaultVal; return true; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool SaveItem(const std::string& strKey, float val)
	///
	/// \brief	Saves a configuration item. 
	///
	/// \author	Mathias Neumann
	/// \date	08.10.2010
	///
	/// \param	strKey	The key. 
	/// \param	val		The value for key. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool SaveItem(const std::string& strKey, float val) { return true; }

// Accessors
public:
	/// Sets MNRT view mode.
	void SetViewMode(MNRTViewMode mode) { m_ViewMode = mode; }
	/// Returns current MNRT view mode.
	MNRTViewMode GetViewMode() const { return m_ViewMode; }
	/// Returns default MNRT view mode.
	MNRTViewMode GetViewModeDef() const { return MNRTView_Result; }
	/// \brief	Sets whether a dynamic scene should be assumed.
	///
	///			Right now, setting this to \c true will only result in rebuilding the
	///			kd-trees for objects and photons every frame.
	void SetDynamicScene(bool b) { m_bDynamicScene = b; }
	/// Gets whether direct lighting by ray tracing is enabled.
	bool GetDynamicScene() const { return m_bDynamicScene; }
	/// Returns default for direct lighting by ray tracing.
	bool GetDynamicSceneDef() const { return false; }

	/// Sets whether direct lighting by ray tracing is enabled.
	void SetEnableDirectRT(bool b) { m_bEnableDirectRT = b; }
	/// Gets whether direct lighting by ray tracing is enabled.
	bool GetEnableDirectRT() const { return m_bEnableDirectRT; }
	/// Returns default for direct lighting by ray tracing.
	bool GetEnableDirectRTDef() const { return true; }
	/// Sets whether to trace shadow rays.
	void SetEnableShadowRays(bool b) { m_bEnableShadowRays = b; }
	/// Gets whether to trace shadow rays.
	bool GetEnableShadowRays() const { return m_bEnableShadowRays; }
	/// Returns whether to trace shadow rays (default).
	bool GetEnableShadowRaysDef() const { return true; }
	/// Sets whether to trace specular reflection rays.
	void SetEnableSpecReflect(bool b) { m_bEnableSpecReflect = b; }
	/// Returns whether specular reflection rays are traced.
	bool GetEnableSpecReflect() const { return m_bEnableSpecReflect; }
	/// Returns default for tracing specular reflection rays.
	bool GetEnableSpecReflectDef() const { return false; }
	/// Sets whether to trace specular transmission rays.
	void SetEnableSpecTransmit(bool b) { m_bEnableSpecTransmit = b; }
	/// Returns whether specular transmission rays are traced.
	bool GetEnableSpecTransmit() const { return m_bEnableSpecTransmit; }
	/// Returns default for tracing specular transmission rays.
	bool GetEnableSpecTransmitDef() const { return false; }
	/// Sets area light samples X (to enable stratification).
	void SetAreaLightSamplesX(uint n) { m_nAreaLightSamplesX = n; }
	/// Returns area light samples X.
	uint GetAreaLightSamplesX() const { return m_nAreaLightSamplesX; }
	/// Returns area light samples X (default).
	uint GetAreaLightSamplesXDef() const { return 8; }
	/// Sets area light samples Y (to enable stratification).
	void SetAreaLightSamplesY(uint n) { m_nAreaLightSamplesY = n; }
	/// Returns area light samples Y.
	uint GetAreaLightSamplesY() const { return m_nAreaLightSamplesY; }
	/// Returns area light samples Y (default).
	uint GetAreaLightSamplesYDef() const { return 8; }

	/// Sets photon mapping mode.
	void SetPhotonMapMode(PhotonMapMode mode) { m_PhotonMapMode = mode; }
	/// Returns current photon mapping mode.
	PhotonMapMode GetPhotonMapMode() const { return m_PhotonMapMode; }
	/// Returns default photon mapping mode.
	PhotonMapMode GetPhotonMapModeDef() const { return PhotonMap_AdaptiveSamplesWang; }
	/// \brief Sets maximum number of photon bounces recorded during photon tracing. 
	///
	///			A bounce is a scattering event of a photon at a surface. When we have 
	///			a maximum of N, we get at most N+1 photon-surface-interactions from a single photon.
	void SetMaxPhotonBounces(uint m) { m_nMaxPhotonBounces = m; }
	/// Returns current maximum for photon bounces.
	uint GetMaxPhotonBounces() const { return m_nMaxPhotonBounces; }
	/// Returns default maximum for photon bounces.
	uint GetMaxPhotonBouncesDef() const { return 5; }
	/// Sets global photon map photon target count.
	void SetTargetCountGlobal(uint c) { m_nTargetCountGlobal = c; }
	/// Returns current global photon map photon target count.
	uint GetTargetCountGlobal() const { return m_nTargetCountGlobal; }
	/// Returns default global photon map photon target count.
	uint GetTargetCountGlobalDef() const { return 200000; }
	/// Sets caustics photon map photon target count.
	void SetTargetCountCaustics(uint c) { m_nTargetCountCaustics = c; }
	/// Returns current caustics photon map photon target count.
	uint GetTargetCountCaustics() const { return m_nTargetCountCaustics; }
	/// Returns default caustics photon map photon target count.
	uint GetTargetCountCausticsDef() const { return 200000; }
	/// Sets k for global photon map kNN search.
	void SetKinKNNSearchGlobal(uint k) { m_nKinKNNSearchGlobal = k; }
	/// Returns current k for global photon map kNN search.
	uint GetKinKNNSearchGlobal() const { return m_nKinKNNSearchGlobal; }
	/// Returns default k for global photon map kNN search.
	uint GetKinKNNSearchGlobalDef() const { return 100; }
	/// Sets k for caustics photon map kNN search.
	void SetKinKNNSearchCaustics(uint k) { m_nKinKNNSearchCaustics = k; }
	/// Returns current k for caustics photon map kNN search.
	uint GetKinKNNSearchCaustics() const { return m_nKinKNNSearchCaustics; }
	/// Returns default k for caustics photon map kNN search.
	uint GetKinKNNSearchCausticsDef() const { return 200; }
	/// Sets number of kNN query radius refinement iterations. 
	void SetKNNRefineIters(uint n) { m_nKNNRefineIters = n; }
	/// Returns current number of kNN query radius refinement iterations.
	uint GetKNNRefineIters() const { return m_nKNNRefineIters; }
	/// Returns default number of kNN query radius refinement iterations.
	uint GetKNNRefineItersDef() const { return 2; }

	/// Sets final gather rays X (to enable stratification).
	void SetFinalGatherRaysX(uint n) { m_nFinalGatherRaysX = n; }
	/// Returns current final gather rays X.
	uint GetFinalGatherRaysX() const { return m_nFinalGatherRaysX; }
	/// Returns default final gather rays X.
	uint GetFinalGatherRaysXDef() const { return 16; }
	/// Sets final gather rays Y (to enable stratification).
	void SetFinalGatherRaysY(uint n) { m_nFinalGatherRaysY = n; }
	/// Returns current final gather rays Y.
	uint GetFinalGatherRaysY() const { return m_nFinalGatherRaysY; }
	/// Returns default final gather rays Y.
	uint GetFinalGatherRaysYDef() const { return 16; }
	/// \brief	Sets normalized geometric variation alpha. 
	///			
	///			Geometric variation alpha is used for adaptive sample seeding. \a should be in 
	///			range 0.1 - 0.5 for normalized scene geometry. Larger = more influence for position.
	void SetGeoVarAlpha(float a) { m_fGeoVarAlpha = a; }
	/// Returns current geometric variation alpha for normalized scene geometry.
	float GetGeoVarAlpha() const  { return m_fGeoVarAlpha; }
	/// Returns default geometric variation alpha for normalized scene geometry.
	float GetGeoVarAlphaDef() const  { return 0.3f; }
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float GetGeoVarAlpha(const MNBBox& boundsScene) const
	///
	/// \brief	Computes the scaled geometric variation alpha for given geometry extent.
	///
	/// \param	boundsScene	The scene bounding box. 
	///
	/// \return	The scaled geometric variation.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float GetGeoVarAlpha(const MNBBox& boundsScene) const;
	/// \brief	Sets geometric variation propagation factor.
	///
	///			Propagation is performed from from leafs to root. Set to 0 to disable propagation.
	void SetGeoVarPropagation(float f) { m_fGeoVarPropagation = f; }
	/// Returns current geometric variation propagation factor.
	float GetGeoVarPropagation() const { return m_fGeoVarPropagation; }
	/// Returns default geometric variation propagation factor.
	float GetGeoVarPropagationDef() const { return 0.f; }
	/// Sets maximum number of k-means algorithm iterations.
	void SetKMeansItersMax(uint n) { m_nKMeansItersMax = n; }
	/// Returns current maximum number of k-means algorithm iterations.
	uint GetKMeansItersMax() const { return m_nKMeansItersMax; }
	/// Returns default maximum number of k-means algorithm iterations.
	uint GetKMeansItersMaxDef() const { return 3; }
	/// Sets whether illumination cuts are used.
	void SetUseIllumCuts(bool b) { m_bUseIllumCuts = b; }
	/// Returns whether illumination cuts are used.
	bool GetUseIllumCuts() const { return m_bUseIllumCuts; }
	/// Returns whether illumination cuts are used (default).
	bool GetUseIllumCutsDef() const { return true; }
	/// Sets to use all leafs as cut nodes instead of computing an illumination cut.
	void SetICutUseLeafs(bool b) { m_bICutUseLeafs = b; }
	/// Returns whether the leafs as cut notes simplification is used.
	bool GetICutUseLeafs() const { return m_bICutUseLeafs; }
	/// Returns whether the leafs as cut notes simplification is used (default).
	bool GetICutUseLeafsDef() const { return true; }
	/// Sets photon map node level for E_min computation (illumination cuts).
	void SetICutLevelEmin(uint lvl) { m_nICutLevelEmin = lvl; }
	/// Gets photon map node level for E_min computation (illumination cuts).
	uint GetICutLevelEmin() const { return m_nICutLevelEmin; }
	/// Gets photon map node level for E_min computation (illumination cuts, default).
	uint GetICutLevelEminDef() const { return 12; }
	/// Sets number of illumination cut refinement iterations.
	void SetICutRefineIters(uint iters) { m_nICutRefineIters = iters; }
	/// Gets number of illumination cut refinement iterations.
	uint GetICutRefineIters() const { return m_nICutRefineIters; }
	/// Gets number of illumination cut refinement iterations (default).
	uint GetICutRefineItersDef() const { return 4; }
	/// \brief	Sets required accuracy of estimated irradiance for final cut nodes.
	///
	///			Inner nodes with
	///			\code fabsf(estimateIrr - exactIrr) >= accuracy * exactIrr \endcode
	///			are removed from the cut and replaced with their children.
	void SetICutAccuracy(float a) { m_fICutAccuracy = a; }
	/// Gets required accuracy of estimated irradiance for final cut nodes.
	float GetICutAccuracy() const { return m_fICutAccuracy; }
	/// Gets required accuracy of estimated irradiance for final cut nodes (default).
	float GetICutAccuracyDef() const { return .2f; }
};


#endif // __MN_MNRTSETTINGS_H__