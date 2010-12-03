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

#include "AssimpScene.h"
#include <aiScene.h>
#include <assimp.hpp>      // C++ importer interface
#include <aiPostProcess.h> // Post processing flags

using namespace std;

AssimpScene::AssimpScene(void)
	: BasicScene()
{
	m_bHasCamera = false;
}

AssimpScene::~AssimpScene(void)
{
}


bool AssimpScene::Load(const std::string& strFile)
{
	if(!BasicScene::Load(strFile))
		return false;

	Assimp::Importer importer;

	// Remove lines and points.
	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE|aiPrimitiveType_POINT);

	const aiScene* pScene = importer.ReadFile(strFile,
		aiProcess_GenNormals			|	// Need normals.
        aiProcess_Triangulate           |	// Need triangles.
		aiProcess_PreTransformVertices	|	// we currently have no scene graph...
		//aiProcess_JoinIdenticalVertices |	// This seems to kill some normals for some scenes (e.g. sibernik.3ds).
        aiProcess_SortByPType			|
		//aiProcess_FindInvalidData       |

		//aiProcess_SplitLargeMeshes		| // These options might destroy some valid information!
		//aiProcess_OptimizeMeshes		|
		//aiProcess_RemoveRedundantMaterials |

		aiProcess_GenUVCoords);				// This step converts non-UV mappings (such as spherical or 
											// cylindrical mapping) to proper texture coordinate channels.
	
	if(pScene == NULL)
		return false;

	if(!ReadGeometry(pScene))
		return false;
	MNMessage("Geometry loaded.");

	ReadMaterials(pScene);
	MNMessage("Found %d materials.", pScene->mNumMaterials);

	ReadCamera(pScene);
	MNMessage("Found %d camera configurations.", pScene->mNumCameras);

	ReadLights(pScene);
	MNMessage("Found %d lights.", pScene->mNumLights);

	/*for(uint i=0; i<pScene->mNumMaterials; i++)
	{
		aiMaterial* pMat = pScene->mMaterials[i];

		float val;
		aiColor3D color(1.f,1.f,1.f);
		pMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
		MNMessage("Mat %d: diffuse(%.02f, %.02f, %.02f).", i+1, color.r, color.g, color.b);
		pMat->Get(AI_MATKEY_REFLECTIVITY, val);
		MNMessage("Mat %d: reflectivity(%.04f).", i+1, val);
		pMat->Get(AI_MATKEY_COLOR_REFLECTIVE, color);
		MNMessage("Mat %d: reflective(%.02f, %.02f, %.02f).", i+1, color.r, color.g, color.b);
		pMat->Get(AI_MATKEY_OPACITY, val);
		MNMessage("Mat %d: opacity(%.04f).", i+1, val);
	}*/

	return true;
}

void AssimpScene::Unload()
{
	BasicScene::Unload();
}

void AssimpScene::ReadMaterials(const aiScene* pScene)
{
	// According to ASSIMP doc: If the AI_SCENE_FLAGS_INCOMPLETE flag is not set there 
	// will always be at least ONE material. See
	// http://assimp.sourceforge.net/lib_html/structai_material.html
	MNAssert(pScene->mNumMaterials > 0);

	for(uint i=0; i<pScene->mNumMaterials; i++)
	{
		aiMaterial* pMaterial = pScene->mMaterials[i];
		BasicMaterial mat;
		
		mat.hasTex[Tex_Diffuse] = (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0);
		mat.hasTex[Tex_Bump] = (pMaterial->GetTextureCount(aiTextureType_HEIGHT) > 0);

		// Just use materials with name "Light" as area lights.
		aiString matName;
		pMaterial->Get(AI_MATKEY_NAME, matName);
		string strMatName;
		strMatName.append(matName.data, matName.length);
		//MNMessage("Mat name: %s.", strMatName.c_str());
		mat.isAreaLight = 
			(strMatName.find("Light", 0) != string::npos ||
			 strMatName.find("light", 0) != string::npos );

		// Get colors.
		aiColor3D clrDiffuse(1.f, 1.f, 1.f);
		pMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, clrDiffuse);
		mat.colorDiffuse = *(MNVector3*)&clrDiffuse;
		aiColor3D clrSpec(0.5f, 0.5f, 0.5f);
		pMaterial->Get(AI_MATKEY_COLOR_SPECULAR, clrSpec);
		mat.colorSpec = *(MNVector3*)&clrSpec;
		//MNMessage("MAT %d: SPEC CLR %.3f %.3f %.3f.", i+1, clrSpec.r, clrSpec.g, clrSpec.b);


		// Specular exponent (shininess).
		pMaterial->Get(AI_MATKEY_SHININESS, mat.specExp);
		// ASSIMP scales .OBJ-file "Ns"-value by a factor of 4.f.
		mat.specExp *= .25f;

		// Transparency alpha.
		pMaterial->Get(AI_MATKEY_OPACITY, mat.transAlpha);
		//MNMessage("MAT %d: OPACITY %.3f.", i+1, mat.transAlpha);
		//mat.transAlpha = 1.0f;

		// Index of refraction. Support only in OBJ blender exporter. Collada
		// has support, but value is not exported.
		pMaterial->Get(AI_MATKEY_REFRACTI, mat.indexRefrac);
		//MNMessage("MAT %d: REFRACTI %.3f.", i+1, mat.indexRefrac);

		// Reflectivity. Seems to be unsupported by most formats.
		//float fReflectivity;
		//pMaterial->Get(AI_MATKEY_REFLECTIVITY, fReflectivity);
		//MNMessage("MAT %d: REFLECT %.3f.", i+1, fReflectivity);

		aiString strTexFile;
		if(mat.hasTex[Tex_Diffuse])
		{
			// Read out first texture path.
			pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &strTexFile);
			// Construct full file path. We assume that strTexFile is relative to scene file.
			mat.strTexPath[Tex_Diffuse] = m_strSceneFile.substr(0, m_strSceneFile.find_last_of("\\/")+1);
			mat.strTexPath[Tex_Diffuse].append(strTexFile.data, strTexFile.length);
		}
		if(mat.hasTex[Tex_Bump])
		{
			// Read out first texture path.
			pMaterial->GetTexture(aiTextureType_HEIGHT, 0, &strTexFile);
			// Construct full file path. We assume that strTexFile is relative to scene file.
			mat.strTexPath[Tex_Bump] = m_strSceneFile.substr(0, m_strSceneFile.find_last_of("\\/")+1);
			mat.strTexPath[Tex_Bump].append(strTexFile.data, strTexFile.length);
		}

		m_vecMaterials.push_back(mat);
	}
}

bool AssimpScene::ReadGeometry(const aiScene* pScene)
{
	MNAssert(pScene);

	// Count the triangles.
	m_numTris = CountTriangles(pScene);

	// Get scene bounds.
	m_Bounds = MNBBox();
	CalculateBounds(pScene, pScene->mRootNode, m_Bounds);
	MNMessage("Scene extents: min(%.3f, %.3f, %.3f) - max(%.3f, %.3f, %.3f).", 
		m_Bounds.ptMin.x, m_Bounds.ptMin.y, m_Bounds.ptMin.z,
		m_Bounds.ptMax.x, m_Bounds.ptMax.y, m_Bounds.ptMax.z);
	
	for(uint i=0; i<3; i++)
	{
		m_v[i] = new MNPoint3[m_numTris];
		m_n[i] = new MNNormal3[m_numTris];
		m_texCoords[i] = new MNVector3[m_numTris];
	}
	m_idxMaterial = new uint[m_numTris];

	// Extract the triangles.
	uint idxTri = 0;
	bool bHasZeroNormal = false;
	for(uint m=0; m<pScene->mNumMeshes; m++)
	{
		aiMesh* pMesh = pScene->mMeshes[m];

		MNPoint3* verts = (MNPoint3*)pMesh->mVertices;
		MNNormal3* normals = (MNNormal3*)pMesh->mNormals;
		MNVector3* texCoords = NULL;
		if(pMesh->GetNumUVChannels() > 0)
			texCoords = (MNVector3*)pMesh->mTextureCoords[0];

		for(uint t=0; t<pMesh->mNumFaces; t++)
		{
			aiFace face = pMesh->mFaces[t];

			// We need triangles!
			if(face.mNumIndices != 3)
			{
				MNError("Import error: Triangles required.\n");
				return false;
			}

			// Material index.
			m_idxMaterial[idxTri] = pMesh->mMaterialIndex;

			// Add the triangle.
			for(uint i=0; i<3; i++)
			{
				// WARNING: Do not transform the vertices here. This *will* lead to
				// rounding errors that would destroy the triangle connectivity.
				m_v[i][idxTri] = *(MNPoint3*)&verts[face.mIndices[i]];
				// Normals from file might be unnormalized, if not generated.
				m_n[i][idxTri] = *(MNNormal3*)&normals[face.mIndices[i]];
				float length = m_n[i][idxTri].Length();
				if(length > 0.f && length != 1.f)
					m_n[i][idxTri] = Normalize(m_n[i][idxTri]);
				else if(length == 0.f)
					bHasZeroNormal = true;
					
				// Texture coordinates. Most times only UV. Range 0..1.
				if(texCoords)
					m_texCoords[i][idxTri] = *(MNVector3*)&texCoords[face.mIndices[i]];
				else
					m_texCoords[i][idxTri] = MNVector3();
			}

			// Test if we have smooth normals.
			//if((m_n[0][idxTri] - m_n[1][idxTri]).Length() != 0.f)
			//	MNMessage("Smooth normals!");

			idxTri++;
		}
	}

	if(bHasZeroNormal)
		MNWarning("Normals of length zero detected.");

	return true;
}

void AssimpScene::ReadCamera(const aiScene* pScene)
{
	MNAssert(pScene);

	m_bHasCamera = pScene->HasCameras();

	if(m_bHasCamera)
	{
		// Just use the first cam config.
		aiCamera* pCamera = pScene->mCameras[0];

		// No camera transformation required since we remove the node graph when importing.
		m_cameraPos =  *(MNPoint3*)&pCamera->mPosition;

		// ASSIMP stores the viewing direction in mLookAt. Therefore convert to target position.
		MNVector3 vViewDir = *(MNVector3*)&pCamera->mLookAt;
		m_cameraLookAt = m_cameraPos + vViewDir;

		m_cameraUp = *(MNVector3*)&pCamera->mUp;
		m_cameraUp = Normalize(m_cameraUp);

		// ASSIMP stores *half* of the total FOV angle, that is from center to left / right side.
		m_cameraFOV = 2.f * pCamera->mHorizontalFOV;
	}

	// Simple validity check.
	m_bHasCamera = (m_cameraLookAt - m_cameraPos).LengthSquared() > 0.f;
}

// See http://assimp.sourceforge.net/lib_html/structai_camera.html for more details.
MNTransform AssimpScene::GetCameraNodeTrans(const aiScene* pScene, const aiCamera* pCamera)
{
	// Get the node for pCamera.
	aiNode* pNodeCam = pScene->mRootNode->FindNode(pCamera->mName);

	// Start with identity.
	aiMatrix4x4 matTrans = aiMatrix4x4();

	// Go back to root and generate transformation matrix.
	aiNode* pNodeCur = pNodeCam;
	while(pNodeCur)
	{
		matTrans = pNodeCur->mTransformation * matTrans;
		pNodeCur = pNodeCur->mParent;
	}

	return MNTransform(MNMatrix4x4(
		matTrans.a1, matTrans.a2, matTrans.a3, matTrans.a4,
		matTrans.b1, matTrans.b2, matTrans.b3, matTrans.b4,
		matTrans.c1, matTrans.c2, matTrans.c3, matTrans.c4,
		matTrans.d1, matTrans.d2, matTrans.d3, matTrans.d4));
}

uint AssimpScene::CountTriangles(const aiScene* pScene)
{
	int count = 0;
	for(uint m=0; m<pScene->mNumMeshes; m++)
	{
		aiMesh* mesh = pScene->mMeshes[m];
		count += mesh->mNumFaces;
	}

	return count;
}

MNMatrix4x4 AssimpScene::GetScaleTransform(const aiScene* pScene)
{
	// Default: no transform (identity matrix).
	MNMatrix4x4 mTransform;

	MNVector3 vecDelta = m_Bounds.ptMax - m_Bounds.ptMin;
	MNPoint3 ptCenter = m_Bounds.ptMin + (vecDelta / 2.0f);
	float fScale = 10.0f / vecDelta.Length();

	mTransform = MNMatrix4x4(				// Moves to origin
					1.0f,0.0f,0.0f,0.0f,
					0.0f,1.0f,0.0f,0.0f,
					0.0f,0.0f,1.0f,0.0f,
					-ptCenter.x,-ptCenter.y,-ptCenter.z,1.0f)
				 *
				 MNMatrix4x4(				// Scales to -5..5
					fScale,0.0f,0.0f,0.0f,
					0.0f,fScale,0.0f,0.0f,
					0.0f,0.0f,fScale,0.0f,
					0.0f,  0.0f,0.0f,1.0f);

	return mTransform;
}

void AssimpScene::CalculateBounds(const aiScene* pScene, aiNode* pNode, 
								  MNBBox& outBounds)
{
	// Check the meshes of this node.
	for(uint m=0; m<pNode->mNumMeshes; m++)
	{
		aiMesh* pMesh = pScene->mMeshes[pNode->mMeshes[m]];
		
		// Add the triangle.
		for(uint v=0; v<pMesh->mNumVertices; v++)
		{
			MNPoint3 pt = *(MNPoint3*)&pMesh->mVertices[v];
			outBounds = Union(outBounds, pt);
		}
	}

	// Don't forget to check the child nodes.
	for(uint i=0; i<pNode->mNumChildren; i++)
		CalculateBounds(pScene, pNode->mChildren[i], outBounds);
}

void AssimpScene::ReadLights(const aiScene* pScene)
{
	MNAssert(pScene);
	m_Lights.clear();

	for(uint i=0; i<pScene->mNumLights; i++)
	{
		aiLight* pLight = pScene->mLights[i];
		BasicLight light;

		if(pLight->mType == aiLightSource_POINT)
		{
			light.type = Light_Point;
		}
		else if(pLight->mType == aiLightSource_DIRECTIONAL)
		{
			light.type = Light_Directional;
		}
		else
		{
			MNMessage("Unsupported ASSIMP light type ignored.");
			continue; // Unsupported light type...
		}

		// Copy stuff, even if undefined.
		light.position = *(MNPoint3*)&pLight->mPosition;
		light.L_emit = *(MNVector3*)&pLight->mColorDiffuse;
		light.L_emit *= pLight->mAttenuationQuadratic;
		if(pLight->mType == aiLightSource_DIRECTIONAL)
			light.direction = Normalize(*(MNVector3*)&pLight->mDirection);
		else
			light.direction = MNVector3(1.f, 0.f, 0.f);

		m_Lights.push_back(light);
	}
}