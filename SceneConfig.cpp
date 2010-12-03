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

#include "SceneConfig.h"
#include "AssimpScene.h"
#include <cutil_math.h>

SceneConfig::SceneConfig(void)
{
	m_strScenePath = "";
	m_pScene = NULL;
	m_pCamera = NULL;
}

SceneConfig::~SceneConfig(void)
{
	SAFE_DELETE(m_pScene);
	SAFE_DELETE(m_pCamera);
}

bool SceneConfig::LoadFromFile(const std::string& scenePath)
{
	if(scenePath.empty())
		m_strScenePath = std::string("Test Scenes/MNSimple.obj");
	else
		m_strScenePath = std::string(scenePath);

	MNMessage("Loading scene: \"%s\".", m_strScenePath.c_str());

	// Load scene.
	m_pScene = new AssimpScene();
	if(!m_pScene->Load(m_strScenePath))
		return false;
	
	MNMessage("Scene loaded: %d triangles.", m_pScene->GetNumTris());

	// Set default values. These can be overwritten more specific values.
	AssignDefault();

	// Try to assign values for given scene path based on static information.
	if(AssignFromStatic())
		MNMessage("Using static scene parameters.");

	return true;
}

void SceneConfig::AssignDefault()
{
	// Deduce settings from scene size.
	MNBBox bounds = m_pScene->GetSceneBounds();
	MNVector3 vDiagonal = bounds.ptMax - bounds.ptMin;
	float fSize = vDiagonal.Length();
	float fMaxSide = std::max(vDiagonal.x, std::max(vDiagonal.y, vDiagonal.z));
	MNPoint3 ptCenter = bounds.GetCenter();

	// Maximum KNN search radius for photon gathering queries. Is refined within KNN search.
	m_fRadiusPMapMax = 0.1f * fSize;

	// k-means radius has to be large enough to find a cluster for each shading point within
	// k-means clustering. It is based on the cluster density, however clusters density might
	// vary greatly as it is the case for shading points.
	// WARNING: Picking this too low would result in clusters without shading points and therefore
	//			illegal cluster shading point indices.
	m_fRadiusWangKMeansMax = 0.05f * fSize;

	// Irradiance sample interpolation radius (where to look for irradiance samples around
	// a given shading point). This is used to search the cluster center kd-tree. Impact on performance
	// shouldn't be that bad as the cluster center kd-tree contains only small numbers of points (e.g. 4k).
	// However, choosing it too large would definitely result in performance problems.
	// WARNING: Choosing to low *can* result in finding only one interpolation sample in sparsely
	//			sampled regions.
	m_fRadiusWangInterpolMax = 0.1f * fSize;

	m_nWangInitialSamples = 4000;

	// Just use global settings.
	m_nTargetCountGlobal = -1;
	m_nTargetCountCaustics = -1;
	// Default scenes have no specular component.
	m_bHasSpecular = false;

	m_fRayEpsilon = fSize * 0.00005f;

	// Add a standard point light on top of the scene.
	m_Light.type = Light_Point;
	m_Light.position = make_float3(ptCenter.x, bounds.ptMax.y - 0.1f*fSize, ptCenter.z);
	m_Light.direction = make_float3(0.0f, 0.0f, 0.0f);
	m_Light.L_emit = 2000.f * make_float3(1.0f, 1.0f, 1.0f);
	m_Light.areaV1 = make_float3(10.f, 0.0f, 0.f);
	m_Light.areaV2 = make_float3(0.f, 0.0f, 10.f);
	m_Light.areaRadius = 0.f;

	// Use a camera pointing to the center of the scene.
	m_pCamera = new CameraModel();
	m_pCamera->LookAt(MNPoint3(ptCenter.x, ptCenter.y, ptCenter.z + 0.6f*fSize), 
		ptCenter, MNVector3(0.f, 1.f, 0.f));
	m_pCamera->SetClipDistances(m_fRayEpsilon, 1e20f);
}

bool SceneConfig::AssignFromStatic()
{
	MNBBox bounds = m_pScene->GetSceneBounds();
	MNPoint3 ptCenter = bounds.GetCenter();

	// Use lower case internally to avoid comparision problems.
	std::string strLower = m_strScenePath;
	std::transform(strLower.begin(), strLower.end(), strLower.begin(), tolower);

	bool bHasData = false;
	MNPoint3 ptCamEye, ptCamLookAt;
	MNVector3 vCamUp;
	if(strLower.find("mnsimple.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 1.0f;
		m_fRadiusWangInterpolMax = 5.0f;
		m_nWangInitialSamples = 2000;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(0.f, 10.f, 0.f);
		m_Light.direction = make_float3(0.0f, 0.0f, 0.0f);
		m_Light.L_emit = 200.f * make_float3(1.0f, 1.0f, 1.0f);

		ptCamEye = MNPoint3(0.2f, 8.6f, 10.f);
		ptCamLookAt = MNPoint3(0.8f, 5.8f, -2.6f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("mnroom.obj") != std::string::npos)
	{
		m_bHasSpecular = true;
		m_fRadiusPMapMax = 1.0f;
		m_fRadiusWangInterpolMax = 5.0f;
		m_nWangInitialSamples = 3000;

		m_Light.type = Light_AreaRect;
		m_Light.position = make_float3(-2.f, 8.99f, -2.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = 30.f * make_float3(1.0f, 1.0f, 1.0f);
		m_Light.areaV1 = make_float3(4.f, 0.0f, 0.f);
		m_Light.areaV2 = make_float3(0.f, 0.0f, 4.f);

		ptCamEye = MNPoint3(-6.6f, 6.9f, 8.6f);
		ptCamLookAt = MNPoint3(0.f, ptCenter.y, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("mnsimpledragon.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 1.0f;
		m_fRadiusWangInterpolMax = 5.0f;
		m_nWangInitialSamples = 5000;

		m_Light.type = Light_AreaRect;
		m_Light.position = make_float3(-2.f, 13.0f, -2.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = 20.0f * make_float3(1.0f, 1.0f, 1.0f);
		m_Light.areaV1 = make_float3(4.f, 0.0f, 0.f);
		m_Light.areaV2 = make_float3(0.f, 0.0f, 4.f);

		ptCamEye = MNPoint3(ptCenter.x - 3.f, ptCenter.y + 1.f, 11.f);
		ptCamLookAt = MNPoint3(0.f, ptCenter.y - 3.f, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("mncaustics.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 1.0f;
		m_bHasSpecular = true;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(0.f, bounds.ptMax.y - 0.001f, 0.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = 80.0f * make_float3(1.0f, 1.0f, 1.0f);

		ptCamEye = MNPoint3(ptCenter.x - 3.f, ptCenter.y - 3.f, 13.f);
		ptCamLookAt = MNPoint3(0.f, ptCenter.y - 3.f, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("mncardioid.obj") != std::string::npos)
	{
		m_nTargetCountGlobal = 0;
		m_nTargetCountCaustics = 100000;
		m_fRadiusPMapMax = 1.2f;
		m_bHasSpecular = true;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(0.f, 4.f, 6.f);
		m_Light.L_emit = 100.0f * make_float3(1.0f, 1.0f, 1.0f);

		ptCamEye = MNPoint3(-7.f, 7.f, 7.f);
		ptCamLookAt = MNPoint3(0.0f, 0.f, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("mnring.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 1.2f;
		m_bHasSpecular = true;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(0.0f, 8.0f, 10.0f);
		m_Light.direction = make_float3(0.0f, 0.0f, 0.0f);
		m_Light.L_emit = 300.f * make_float3(1.0f, 1.0f, 1.0f);

		ptCamEye = MNPoint3(-6.f, 6.f, 6.f);
		ptCamLookAt = MNPoint3(0.0f, 0.f, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("sponza.3ds") != std::string::npos)	// CG
	{
		m_fRadiusPMapMax = 0.6f;
		m_fRadiusWangKMeansMax = 0.5f;

		m_Light.type = Light_AreaRect;
		m_Light.position = make_float3(-10.f, 15.6f, -3.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = make_float3(7.0f, 7.5f, 7.0f);
		m_Light.areaV1 = make_float3(20.f, 0.0f, 0.f);
		m_Light.areaV2 = make_float3(0.f, 0.0f, 6.f);

		// Use custom camera. This scene contains camera information, but they seem useless.
		ptCamEye = MNPoint3(7.164896f, 3.388182f, -2.231596f);
		ptCamLookAt = MNPoint3(-2.146986f, 1.661487f, 7.461062f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("sponza.obj") != std::string::npos) // CRYTEK SPONZA
	{
		m_fRadiusPMapMax = 6.f;

		m_Light.type = Light_AreaRect;
		m_Light.position = make_float3(ptCenter.x - 50.f, bounds.ptMax.y, ptCenter.z - 10.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = 50.f * make_float3(0.9f, 1.0f, 0.9f);
		m_Light.areaV1 = make_float3(100.f, 0.0f, 0.f);
		m_Light.areaV2 = make_float3(0.f, 0.0f, 20.f);

		ptCamEye = MNPoint3(82.3f, 63.3f, -17.6f);
		ptCamLookAt = MNPoint3(1.3f, 31.8f, 5.6f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("sibenik.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 0.4f;
		m_fRadiusWangInterpolMax = 1.0f;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(0.0f, 3.0f, 0.0f);
		m_Light.direction = make_float3(0.0f, 0.0f, 0.0f);
		m_Light.L_emit = 20.0f * make_float3(0.7f, 0.9f, 0.8f);

		ptCamEye = MNPoint3(4.921351f, 1.017859f, -1.883862f);
		ptCamLookAt = MNPoint3(1.339943f, 0.92758f, 0.131692f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("kitchen.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 5.0f;
		m_fRadiusWangInterpolMax = 20.0f;
		m_nWangInitialSamples = 5000;

		m_Light.type = Light_Point;
		m_Light.position = make_float3(-20.f, 5.0f, 20.f);
		m_Light.direction = make_float3(0.0f, 0.0f, 0.0f);
		m_Light.L_emit = 7000.0f * make_float3(1.0f, 1.0f, 1.0f);

		ptCamEye = MNPoint3(50.f, 0.0f, -10.f);
		ptCamLookAt = MNPoint3(-105.f, -21.f, 52.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else if(strLower.find("conference.obj") != std::string::npos)
	{
		m_fRadiusPMapMax = 0.8f;

		m_Light.type = Light_AreaRect;
		m_Light.position = make_float3(-6.f, 8.4f, -1.f);
		m_Light.direction = make_float3(0.0f, -1.0f, 0.0f);
		m_Light.L_emit = 70.0f * make_float3(1.0f, 1.0f, 1.0f);
		m_Light.areaV1 = make_float3(0.f, 0.0f, 2.f);
		m_Light.areaV2 = make_float3(12.f, 0.0f, 0.f);

		ptCamEye = MNPoint3(13.f, 6.f, -9.f);
		ptCamLookAt = MNPoint3(0.f, 3.f, 0.f);
		vCamUp = MNVector3(0.f, 1.f, 0.f);
		bHasData = true;
	}
	else
	{
		bHasData = m_pScene->GetLightCount() > 0 && m_pScene->HasCamera();
		if(m_pScene->GetLightCount() > 0)
		{
			// Just use the first light.
			const BasicLight& light = m_pScene->GetLight(0);
			m_Light.type = light.type;
			m_Light.position = *(float3*)&light.position;
			m_Light.L_emit = *(float3*)&light.L_emit;
			m_Light.direction = *(float3*)&light.direction;
		}
		if(m_pScene->HasCamera())
		{
			ptCamEye = m_pScene->GetCameraPos();
			ptCamLookAt = m_pScene->GetCameraLookAt();
			vCamUp = m_pScene->GetCameraUp();
		}
	}

	// Force to false to disable caustics photon tracing (temporarily).
	m_bHasSpecular = false;

	if(bHasData)
		m_pCamera->LookAt(ptCamEye, ptCamLookAt, vCamUp);

	return bHasData;
}

MNBBox SceneConfig::GetSceneBounds() const 
{ 
	return m_pScene->GetSceneBounds(); 
}