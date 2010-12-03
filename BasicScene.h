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
/// \file	MNRT\BasicScene.h
///
/// \brief	Declares the BasicScene class.
/// \author	Mathias Neumann
/// \date	05.03.2010
/// \ingroup	scene
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	scene	Scene Processing
/// 
/// \brief	Scene file input and scene processing components of MNRT.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_BASICSCENE_H__
#define __MN_BASICSCENE_H__

#pragma once

#include <string>
#include <vector>
#include "KernelDefs.h"
#include "Geometry/MNGeometry.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	BasicLight
///
/// \brief	Describes a basic scene light source.
///
/// \author	Mathias Neumann
/// \date	05.03.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct BasicLight
{
	/// Light type.
	LightType type;
	/// Light position. Undefined for directional lights.
	MNPoint3 position;
	/// Luminance (area lights, directional lights) or intensity (point light).
	MNVector3 L_emit;
	/// Normalized light direction. Undefined for point lights.
	MNVector3 direction;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \struct	BasicMaterial
///
/// \brief	Describes a basic scene material.
///
/// \author	Mathias Neumann
/// \date	05.03.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
struct BasicMaterial
{
	/// Whether this is an area light material.
	bool isAreaLight;
	/// Diffuse material color.
	MNVector3 colorDiffuse;
	/// Specular material color.
	MNVector3 colorSpec;
	/// Specular exponent (shininess).
	float specExp;
	/// Transparency alpha (opaque = 1, completely transparent = 0).
	float transAlpha;
	/// Index of refraction of the material.
	float indexRefrac;
	/// Whether this material has textures.
	bool hasTex[NUM_TEX_TYPES];
	/// Texture paths.
	std::string strTexPath[NUM_TEX_TYPES];
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	BasicScene
///
/// \brief	Basic scene class.
/// 		
/// 		Used to abstract from the concrete way a scene is loaded. I introduced this as
/// 		initially I was not sure how to load scene models for MNRT. This basic class does
///			nothing and should be subclassed by concrete scene loaders.
///
///			MNRT was designed to handle triangle primitives only. Therefore this class assumes
///			that all geometric primitives are triangles. Subclasses have to ensure this.
///
/// \author	Mathias Neumann
/// \date	05.03.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class BasicScene
{
public:
	/// Default constructor. Use Load() to load a scene.
	BasicScene(void);
	virtual ~BasicScene(void);

// Attributes
protected:
	/// Scene source file path.
	std::string m_strSceneFile;

	/// Total triangle count.
	uint m_numTris;
	/// Scene bounds as AABB.
	MNBBox m_Bounds;
	/// Material vector that contains all materials in form of BasicMaterial objects.
	std::vector<BasicMaterial> m_vecMaterials;

	/// Triangle vertex arrays, stored as three arrays of #m_numTris points each.
	MNPoint3* m_v[3];
	/// Triangle normal arrays, stored as three arrays of #m_numTris normals each.
	MNNormal3* m_n[3];
	/// Material index array. One per triangle.
	uint* m_idxMaterial;
	/// Texture coordinate arrays, stored as three arrays of #m_numTris UVW-vectors each.
	/// So at most three-dimensional textures are supported. 
	MNVector3* m_texCoords[3];

	/// Lights vector that stores all lights in form of BasicLight objects.
	std::vector<BasicLight> m_Lights;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual bool Load(const std::string& strFile)
	///
	/// \brief	Loads the scene from the given file. This BasicScene::Load() method does nothing
	///			but setting assigning the scene path to #m_strSceneFile.
	///
	/// \author	Mathias Neumann.
	/// \date	05.03.2010
	///
	/// \param	strFile	File path to load the scene from. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool Load(const std::string& strFile);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void Unload()
	///
	/// \brief	Unloads the loaded scene.
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void Unload();

// Accessors
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const bool IsLoaded()
	///
	/// \brief	Query if a scene is loaded. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	\c true if a scene is loaded, \c false if not. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const bool IsLoaded() { return m_numTris != 0; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const std::string GetSourceFile()
	///
	/// \brief	Gets the source file path. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The source file path. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const std::string GetSourceFile() { return m_strSceneFile; }


// Accessors - GEOMETRY DATA
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const size_t GetMaterialCount()
	///
	/// \brief	Gets the material count. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The material count. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const size_t GetMaterialCount() { return m_vecMaterials.size(); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const BasicMaterial& GetMaterial(size_t idx)
	///
	/// \brief	Gets a material in form of a BasicMaterial. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \param	idx	Zero-based index of the material. It is assumed that this index is valid.
	///
	/// \return	The material object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const BasicMaterial& GetMaterial(size_t idx) { return m_vecMaterials.at(idx); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const uint GetNumTris()
	///
	/// \brief	Gets the number triangles. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The number triangles. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const uint GetNumTris() { return m_numTris; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	const MNBBox GetSceneBounds()
	///
	/// \brief	Gets the scene bounds as AABB. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The scene bounds. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	const MNBBox GetSceneBounds() { return m_Bounds; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNPoint3* GetVertices(uint i)
	///
	/// \brief	Gets all i-th triangle vertices.
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \param	i	Trianlge vertex index, 0 <= i <= 2. 
	///
	/// \return	An array of points, where each point is the i-th triangle vertex for some triangle. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNPoint3* GetVertices(uint i)  { return m_v[i]; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNNormal3* GetNormals(uint i)
	///
	/// \brief	Gets all i-th triangle normals.
	//
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \param	i	Trianlge vertex index, 0 <= i <= 2. 
	///
	/// \return	An array of normals, where each normal is the i-th triangle normal for some triangle. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNNormal3* GetNormals(uint i) { return m_n[i]; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	uint* GetMaterialIndices()
	///
	/// \brief	Gets the material indices, one per triangle. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The material index array. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	uint* GetMaterialIndices() { return m_idxMaterial; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNVector3* GetTextureCoords(uint i)
	///
	/// \brief	Gets all i-th texture coordinates.
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \param	i	Trianlge vertex index, 0 <= i < 3. 
	///
	/// \return	An array of texture coordinates, where each texture coordinate is the i-th triangle
	/// 		texture coordinate for some triangle. A texture coordinate is represented as a
	/// 		MNVector3 and can therefore represent up to three dimensional texture coordinates. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNVector3* GetTextureCoords(uint i) { return m_texCoords[i]; }

// Accessors - CAMERA DATA
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const bool HasCamera()
	///
	/// \brief	Query if this object has camera information. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	\c true if information available, \c false if not. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const bool HasCamera() { return false; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const MNPoint3 GetCameraPos()
	///
	/// \brief	Gets the camera position. Only valid if HasCamera() returned \c true. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The camera position. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const MNPoint3 GetCameraPos() { return MNPoint3(); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const MNPoint3 GetCameraLookAt()
	///
	/// \brief	Gets the camera look-at position. Only valid if HasCamera() returned \c true. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The camera look-at position. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const MNPoint3 GetCameraLookAt() { return MNPoint3(); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const MNVector3 GetCameraUp()
	///
	/// \brief	Gets the camera up vector. Only valid if HasCamera() returned \c true. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The camera up vector. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const MNVector3 GetCameraUp() { return MNVector3(); }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const float GetCameraFOV()
	///
	/// \brief	Gets the camera field of view (FOV) in radians. Only valid if HasCamera() returned \c
	/// 		true. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The camera FOV value in radians. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const float GetCameraFOV() { return 0.f; }

// Accessors - LIGHT DATA
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const uint GetLightCount()
	///
	/// \brief	Gets the light count. 
	///
	/// \author	Mathias Neumann
	/// \date	05.03.2010
	///
	/// \return	The light source count. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const uint GetLightCount() { return 0; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual const BasicLight& GetLight(uint i)
	///
	/// \brief	Gets the i-th light. 
	///
	/// \author	Mathias Neumann
	/// \date	10.04.2010
	///
	/// \param	i	The light index. It is assumed that this index is valid.
	///
	/// \return	The light in form of a BasicLight. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual const BasicLight& GetLight(uint i)  { return m_Lights.at(i); };
};


#endif // __MN_BASICSCENE_H__