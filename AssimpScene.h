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
/// \file	MNRT\AssimpScene.h
///
/// \brief	Declares the AssimpScene class. 
/// \author	Mathias Neumann
/// \date	01.02.2010
/// \ingroup	scene
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MN_ASSIMPSCENE_H_
#define _MN_ASSIMPSCENE_H_

#pragma once

#include "BasicScene.h"

struct aiScene;
struct aiNode;
struct aiCamera;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	AssimpScene
///
/// \brief	Implements the BasicScene class by loading a scene using ASSIMP.
/// 		
/// 		The Open Asset Import Library (ASSIMP) is an open source library that supports
/// 		loading a lot of scene types. Check http://assimp.sourceforge.net/ for more
/// 		information on ASSIMP.
///
/// \todo	Improve detection of area lights. Right now, I just use materials with "Light" in the
///			name as area light materials.
///
/// \author	Mathias Neumann
/// \date	01.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class AssimpScene : public BasicScene
{
public:
	AssimpScene(void);
	virtual ~AssimpScene(void);

// Attributes
private:
	// Camera data.
	bool m_bHasCamera;
	MNPoint3 m_cameraPos;
	MNPoint3 m_cameraLookAt;
	MNVector3 m_cameraUp;
	float m_cameraFOV;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool Load(const std::string& strFile)
	///
	/// \brief	Loads the scene using ASSIMP.
	///
	/// \author	Mathias Neumann
	/// \date	01.02.2010
	///
	/// \param	strFile	The file to load.
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool Load(const std::string& strFile);
	virtual void Unload();

// Accessors
public:
	virtual const bool HasCamera() { return m_bHasCamera; }
	virtual const MNPoint3 GetCameraPos() { return m_cameraPos; }
	virtual const MNPoint3 GetCameraLookAt() { return m_cameraLookAt; }
	virtual const MNVector3 GetCameraUp() { return m_cameraUp; }
	virtual const float GetCameraFOV() { return m_cameraFOV; }

	virtual const uint GetLightCount() { return m_Lights.size(); }

// Implementation
private:
	/// Reads in material information as required.
	void ReadMaterials(const aiScene* pScene);
	/// Reads the geometry (as far as needed).
	bool ReadGeometry(const aiScene* pScene);
	/// Reads the camera configurations, if avaiable.
	void ReadCamera(const aiScene* pScene);
	/// Reads the lights, if avaliable.
	void ReadLights(const aiScene* pScene);
	
	/// Counts the triangles.
	uint CountTriangles(const aiScene* pScene);
	/// Calculates a transform that moves the scene into the [-5..5] box.
	MNMatrix4x4 GetScaleTransform(const aiScene* pScene);
	/// Calculates the bounds of the scene (pass the root node). Recursive.
	void CalculateBounds(const aiScene* pScene, aiNode* pNode, MNBBox& outBounds);
	/// Computes the camera node transformation matrix for the given camera.
	MNTransform GetCameraNodeTrans(const aiScene* pScene, const aiCamera* pCamera);
};

#endif // _MN_ASSIMPSCENE_H_