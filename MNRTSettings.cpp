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

#include "MNRTSettings.h"
#include "Geometry/MNGeometry.h"


MNRTSettings::MNRTSettings()
{
}

MNRTSettings::~MNRTSettings()
{
}

bool MNRTSettings::Load()
{
	// Need to check if the long values are in range.
	long nMode;

	bool result;
	result &= LoadItem("MNRT_ViewMode", &nMode, (long)GetViewModeDef());
	if(nMode >= 0 && nMode <= (long)MNRTView_ClusterCenters)
		m_ViewMode = (MNRTViewMode)nMode;
	result &= LoadItem("MNRT_DynamicScene", &m_bDynamicScene, GetDynamicScene());

	result &= LoadItem("MNRT_EnableDirectRT", &m_bEnableDirectRT, GetEnableDirectRTDef());
	result &= LoadItem("MNRT_EnableShadowRays", &m_bEnableShadowRays, GetEnableShadowRaysDef());
	result &= LoadItem("MNRT_EnableSpecReflect", &m_bEnableSpecReflect, GetEnableSpecReflectDef());
	result &= LoadItem("MNRT_EnableSpecTransmit", &m_bEnableSpecTransmit, GetEnableSpecTransmitDef());
	result &= LoadItem("MNRT_AreaLightSamplesX", &m_nAreaLightSamplesX, GetAreaLightSamplesXDef());
	result &= LoadItem("MNRT_AreaLightSamplesY", &m_nAreaLightSamplesY, GetAreaLightSamplesYDef());

	result &= LoadItem("MNRT_PhotonMapMode", &nMode, (long)GetPhotonMapModeDef());
	if(nMode >= 0 && nMode <= (long)PhotonMap_AdaptiveSamplesWang)
		m_PhotonMapMode = (PhotonMapMode)nMode;
	result &= LoadItem("MNRT_MaxPhotonBounces", &m_nMaxPhotonBounces, GetMaxPhotonBouncesDef());
	result &= LoadItem("MNRT_TargetCountGlobal", &m_nTargetCountGlobal, GetTargetCountGlobalDef());
	result &= LoadItem("MNRT_TargetCountCaustics", &m_nTargetCountCaustics, GetTargetCountCausticsDef());
	result &= LoadItem("MNRT_KinKNNSearchGlobal", &m_nKinKNNSearchGlobal, GetKinKNNSearchGlobalDef());
	result &= LoadItem("MNRT_KinKNNSearchCaustics", &m_nKinKNNSearchCaustics, GetKinKNNSearchCausticsDef());
	result &= LoadItem("MNRT_KNNRefineIters", &m_nKNNRefineIters, GetKNNRefineItersDef());

	result &= LoadItem("MNRT_FinalGatherRaysX", &m_nFinalGatherRaysX, GetFinalGatherRaysXDef());
	result &= LoadItem("MNRT_FinalGatherRaysY", &m_nFinalGatherRaysY, GetFinalGatherRaysYDef());
	result &= LoadItem("MNRT_GeoVarAlpha", &m_fGeoVarAlpha, GetGeoVarAlphaDef());
	if(m_fGeoVarAlpha < 0.f) // Ensure a positive value.
		m_fGeoVarAlpha = 0.3f;
	result &= LoadItem("MNRT_GeoVarPropagation", &m_fGeoVarPropagation, GetGeoVarPropagationDef());
	result &= LoadItem("MNRT_KMeansItersMax", &m_nKMeansItersMax, GetKMeansItersMaxDef());
	result &= LoadItem("MNRT_UseIllumCuts", &m_bUseIllumCuts, GetUseIllumCutsDef());
	result &= LoadItem("MNRT_ICutUseLeafs", &m_bICutUseLeafs, GetICutUseLeafsDef());
	result &= LoadItem("MNRT_ICutLevelEmin", &m_nICutLevelEmin, GetICutLevelEminDef());
	result &= LoadItem("MNRT_ICutRefineIters", &m_nICutRefineIters, GetICutRefineItersDef());
	result &= LoadItem("MNRT_ICutAccuracy", &m_fICutAccuracy, GetICutAccuracyDef());

	return result;
}

bool MNRTSettings::Save()
{
	bool result;
	result &= SaveItem("MNRT_ViewMode", (long)m_ViewMode);
	result &= SaveItem("MNRT_DynamicScene", m_bDynamicScene);

	result &= SaveItem("MNRT_EnableDirectRT", m_bEnableDirectRT);
	result &= SaveItem("MNRT_EnableShadowRays", m_bEnableShadowRays);
	result &= SaveItem("MNRT_EnableSpecReflect", m_bEnableSpecReflect);
	result &= SaveItem("MNRT_EnableSpecTransmit", m_bEnableSpecTransmit);
	result &= SaveItem("MNRT_AreaLightSamplesX", m_nAreaLightSamplesX);
	result &= SaveItem("MNRT_AreaLightSamplesY", m_nAreaLightSamplesY);

	result &= SaveItem("MNRT_PhotonMapMode", (long)m_PhotonMapMode);
	result &= SaveItem("MNRT_MaxPhotonBounces", m_nMaxPhotonBounces);
	result &= SaveItem("MNRT_TargetCountGlobal", m_nTargetCountGlobal);
	result &= SaveItem("MNRT_TargetCountCaustics", m_nTargetCountCaustics);
	result &= SaveItem("MNRT_KinKNNSearchGlobal", m_nKinKNNSearchGlobal);
	result &= SaveItem("MNRT_KinKNNSearchCaustics", m_nKinKNNSearchCaustics);
	result &= SaveItem("MNRT_KNNRefineIters", m_nKNNRefineIters);
	result &= SaveItem("MNRT_FinalGatherRaysX", m_nFinalGatherRaysX);
	result &= SaveItem("MNRT_FinalGatherRaysY", m_nFinalGatherRaysY);

	result &= SaveItem("MNRT_GeoVarAlpha", m_fGeoVarAlpha);
	result &= SaveItem("MNRT_GeoVarPropagation", m_fGeoVarPropagation);
	result &= SaveItem("MNRT_KMeansItersMax", m_nKMeansItersMax);
	result &= SaveItem("MNRT_UseIllumCuts", m_bUseIllumCuts);
	result &= SaveItem("MNRT_ICutUseLeafs", m_bICutUseLeafs);
	result &= SaveItem("MNRT_ICutLevelEmin", m_nICutLevelEmin);
	result &= SaveItem("MNRT_ICutRefineIters", m_nICutRefineIters);
	result &= SaveItem("MNRT_ICutAccuracy", m_fICutAccuracy);

	return result;
}

float MNRTSettings::GetGeoVarAlpha(const MNBBox& boundsScene) const 
{
	MNVector3 vDiagonal = boundsScene.ptMax - boundsScene.ptMin;
	float fMaxSide = std::max(vDiagonal.x, std::max(vDiagonal.y, vDiagonal.z));

	// Geometric variation alpha should be in range 0.1 - 0.5 for *normalized* scene
	// geometry, that is for fMaxSide = 2.f. For scenes with larger geometric proprotions,
	// we have to reduce the alpha accordingly as we want to restrict position's influence
	// regarding to normal's influence in the geometric variation formula.
	return m_fGeoVarAlpha * (2.f / fMaxSide);
}