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

#include "KernelDefs.h"
#include "MNCudaUtil.h"
#include "MNCudaMemPool.h"
#include "BasicScene.h"
#include "MNCudaPrimitives.h"

// DevIL includes to read in texture images.
#include <IL/il.h>

// Limit for texture size. Should suffice in almost all cases and avoid problems with
// larger textures (CUDA-related or memory related).
#define MAX_TEX_SIZE	32768

#define SET_TEX_FLAG(idxMat, idxTex, value) *((unsigned char*)&matProps.flags[idxMat] + idxTex) = value


////////////////////////////////////////////////////////////////////////////////////////////////////
// MaterialData implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool MaterialData::Initialize(BasicScene* pScene)
{
	MNAssert(pScene);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	numMaterials = std::min((uint)pScene->GetMaterialCount(), (uint)MAX_MATERIALS);

	for(uint i=0; i<numMaterials; i++)
	{
		const BasicMaterial& mat = pScene->GetMaterial(i);

		// Set light flags.
		matProps.flags[i].w = mat.isAreaLight;

		// Set colors.
		matProps.clrDiff[i] = *(float3*)&mat.colorDiffuse;
		matProps.clrSpec[i] = *(float3*)&mat.colorSpec;
		matProps.specExp[i] = mat.specExp;
		matProps.transAlpha[i] = mat.transAlpha;
		matProps.indexRefrac[i] = mat.indexRefrac;
	}

	// Move textures into CUDA arrays, if any.
	for(uint t=0; t<NUM_TEX_TYPES; t++)
	{
		std::vector<TextureHost> vecHostTextures;
		LoadTextures(pScene, t, &vecHostTextures);

		for(size_t i=0; i<vecHostTextures.size(); i++)
		{
			TextureHost& info = vecHostTextures[i];
			
			cudaArray* pArrayTex;
			cudaChannelFormatDesc channelDesc;
			if(t == Tex_Bump)
			{
				channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
				mncudaSafeCallNoSync(cudaMallocArray(&pArrayTex, &channelDesc, info.size.x, info.size.y));
				mncudaSafeCallNoSync(cudaMemcpyToArray(pArrayTex, 0, 0, info.h_texture, 
					info.size.x*info.size.y*sizeof(float), cudaMemcpyHostToDevice));
			}
			else
			{
				channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
				mncudaSafeCallNoSync(cudaMallocArray(&pArrayTex, &channelDesc, info.size.x, info.size.y));
				mncudaSafeCallNoSync(cudaMemcpyToArray(pArrayTex, 0, 0, info.h_texture, 
					info.size.x*info.size.y*sizeof(uchar4), cudaMemcpyHostToDevice));
			}

			vecTexArrays[t].push_back(pArrayTex);

			// Delete host array.
			SAFE_DELETE_ARRAY(info.h_texture);
		}
	}

	return true;
}

void MaterialData::LoadTextures(BasicScene* pScene, uint texType, 
								std::vector<TextureHost>* outHostTextures)
{
	outHostTextures->clear();
	for(uint i=0; i<numMaterials; i++)
	{
		const BasicMaterial& mat = pScene->GetMaterial(i);

		// Load texture image from file.
		if(!mat.hasTex[texType])
			SET_TEX_FLAG(i, texType, -1);
		else
		{
			TextureHost info;
			info.h_texture = LoadImageFromFile(mat.strTexPath[texType], texType == Tex_Bump, &info.size);
			if(!info.h_texture)
			{
				MNWarning("Failed to load texture image: %s.\n", mat.strTexPath[texType].c_str());
				SET_TEX_FLAG(i, texType, -1);
				continue;
			}

			outHostTextures->push_back(info);
			SET_TEX_FLAG(i, texType, outHostTextures->size() - 1);
		}
	}
}

/*static*/ void* MaterialData::LoadImageFromFile(const std::string& strImage, bool isBumpMap, uint2* outImgSize)
{
	ILuint handleImage, srcW, srcH;

	// Generate and bind *one* image to handle.
	ilGenImages(1, &handleImage);
	ilBindImage(handleImage);

	// Load from source.
	if(!ilLoadImage(strImage.c_str()))
	{
		// Failed to load image
		ilDeleteImages(1, &handleImage);
		return NULL;
	}

	// Read out image width/height.
	srcW = ilGetInteger(IL_IMAGE_WIDTH);
	srcH = ilGetInteger(IL_IMAGE_HEIGHT);
	//MNMessage("Texture: %d x %d (path: %s).\n", srcW, srcH, strImage.c_str());

	if(srcW > MAX_TEX_SIZE || srcH > MAX_TEX_SIZE)
	{
		MNWarning("Illegal texture size detected.");
		ilDeleteImages(1, &handleImage);
		return NULL;
	}

	// Ensure a compatible image size.
	outImgSize->x = std::min((uint)MAX_TEX_SIZE, srcW);
	outImgSize->y = std::min((uint)MAX_TEX_SIZE, srcH);

	void* h_texture;
	if(isBumpMap)
	{
		// Convert to one channel, floating point.
		if(!ilConvertImage(IL_LUMINANCE, IL_FLOAT))
		{
			// Failed...
			ilDeleteImages(1, &handleImage);
			return NULL;
		}

		h_texture = new float[outImgSize->x*outImgSize->y];

		unsigned char* dataIL = (unsigned char*)ilGetData();
		memcpy(h_texture, dataIL, outImgSize->x*outImgSize->y*sizeof(float));
	}
	else
	{
		// We need the image data in IL_RGBA format, IL_UNSIGNED_BYTE type.
		h_texture = new uchar4[outImgSize->x*outImgSize->y];
		ilCopyPixels(0, 0, 0, outImgSize->x, outImgSize->y, 1, 
			IL_RGBA, IL_UNSIGNED_BYTE, (void*)h_texture);
	}

	// We're done, delete the image and converted buffer.
	ilDeleteImages(1, &handleImage);

	return h_texture;
}

void MaterialData::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	for(uint type=0; type<NUM_TEX_TYPES; type++)
		for(size_t i=0; i<vecTexArrays[type].size(); i++)
			mncudaSafeCallNoSync(cudaFreeArray(vecTexArrays[type][i]));
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// TriangleData implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool TriangleData::Initialize(BasicScene* pScene)
{
	MNAssert(pScene);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Triangle count.
	numTris = pScene->GetNumTris();
	MNAssert(numTris > 0);

	// Scene bounds.
	MNBBox bounds = pScene->GetSceneBounds();
	aabbMin = *(float3*)&bounds.ptMin;
	aabbMax = *(float3*)&bounds.ptMax;

	// Temporary buffer, used to convert from (x, y, z), (x, y, z), ... to 
	// (x, x, ...), (y, y, ...), ...
	float* pTempBuf = new float[numTris];
	float4* pTempBuf4 = new float4[numTris];
	float* pDestCur = pTempBuf;

	// Triangle vertices.
	for(uint i=0; i<3; i++)
	{		
		// Convert to float4
		float4* pDest = pTempBuf4; 
		float3* pSrcCur = (float3*)pScene->GetVertices(i);
		for(uint j=0; j<numTris; j++)
			*pDest++ = make_float4(*pSrcCur++);

		mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_verts[i], numTris*sizeof(float4), "Scene data"));
		mncudaSafeCallNoSync(cudaMemcpy(d_verts[i], pTempBuf4, numTris*sizeof(float4), cudaMemcpyHostToDevice));
	}

	// Triangle normals.
	for(uint i=0; i<3; i++)
	{
		float4* pDest = pTempBuf4; 
		float3* pSrcCur = (float3*)pScene->GetNormals(i);
		for(uint j=0; j<numTris; j++)
			*pDest++ = make_float4(*pSrcCur++);

		mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_normals[i], numTris*sizeof(float4), "Scene data"));
		mncudaSafeCallNoSync(cudaMemcpy(d_normals[i], pTempBuf4, numTris*sizeof(float4), cudaMemcpyHostToDevice));
	}

	// Material indices.
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_idxMaterial, numTris*sizeof(uint), "Scene data"));
	mncudaSafeCallNoSync(cudaMemcpy(d_idxMaterial, pScene->GetMaterialIndices(), 
		numTris*sizeof(uint), cudaMemcpyHostToDevice));
	
	// Texture coords.
	float2* pTempBuf2 = new float2[numTris];
	for(uint v=0; v<3; v++)
	{
		float2* pDestCur2 = pTempBuf2; 
		MNVector3* pSrcCur = pScene->GetTextureCoords(v);
		for(uint j=0; j<numTris; j++)
		{
			// Just UV for now.
			(*pDestCur2).x = (*pSrcCur)[0];
			(*pDestCur2).y = (*pSrcCur)[1];
			pDestCur2++;
			pSrcCur++;
		}

		mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_texCoords[v], numTris*sizeof(float2), "Scene data"));
		mncudaSafeCallNoSync(cudaMemcpy(d_texCoords[v], pTempBuf2, 
			numTris*sizeof(float2), cudaMemcpyHostToDevice));
	}
	SAFE_DELETE_ARRAY(pTempBuf2);
	SAFE_DELETE_ARRAY(pTempBuf4);
	SAFE_DELETE_ARRAY(pTempBuf);

	// Initialize auxiliary vars.
	InitAuxillary();

	return true;
}

bool TriangleData::InitAuxillary()
{
	MNAssert(numTris > 0);

	return true;
}

void TriangleData::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();

	// Free allocated memory.
	for(uint i=0; i<3; i++)
	{
		mncudaSafeCallNoSync(pool.Release(d_verts[i]));
		mncudaSafeCallNoSync(pool.Release(d_normals[i]));
		mncudaSafeCallNoSync(pool.Release(d_texCoords[i]));
	}
	mncudaSafeCallNoSync(pool.Release(d_idxMaterial));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PhotonData implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool PhotonData::Initialize(uint _maxPhotons)
{
	MNAssert(_maxPhotons > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	numPhotons = 0;

	// Align max count for the use of textures. This is required for subcomponents.
	maxPhotons = MNCUDA_ALIGN_EX(_maxPhotons, pool.GetTextureAlignment()/sizeof(float));

	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_powers, maxPhotons*sizeof(float4), "Photon data"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_positions, maxPhotons*sizeof(float4), "Photon data"));

	return true;
}

void PhotonData::Clear()
{
	// Just reset count.
	numPhotons = 0;
}

void PhotonData::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_powers));
	mncudaSafeCallNoSync(pool.Release(d_positions));
	numPhotons = 0;
	maxPhotons = 0;
}

void PhotonData::CompactSrcAddr(uint* d_srcAddr, uint countNew)
{
	if(countNew == 0)
	{
		numPhotons = 0;
		return;
	}

	uint countOld = numPhotons;
	mncudaCompactInplace(d_positions, d_srcAddr, countOld, countNew);
	mncudaCompactInplace(d_powers, d_srcAddr, countOld, countNew);

	numPhotons = countNew;
}

uint PhotonData::Merge(const PhotonData& other, uint* d_isValid)
{
	if(other.numPhotons == 0)
		return 0;
	
	MNCudaMemory<uint> d_srcAddr(other.numPhotons);
	uint countNew = mncudaGenCompactAddresses(d_isValid, other.numPhotons, d_srcAddr);

	if(numPhotons + countNew > maxPhotons)
		MNFatal("Failed to merge photon lists. Too many photons.");
	if(countNew == 0)
		return 0;

	// Now move source data to destination data.
	mncudaSetFromAddress(d_positions + numPhotons, d_srcAddr, other.d_positions, countNew);
	mncudaSetFromAddress(d_powers + numPhotons, d_srcAddr, other.d_powers, countNew);

	// Update photon count.
	numPhotons += countNew;

	return countNew;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// ShadingPoints implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool ShadingPoints::Initialize(uint _maxPoints)
{
	MNAssert(_maxPoints > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	maxPoints = MNCUDA_ALIGN_EX(_maxPoints, pool.GetTextureAlignment()/sizeof(float));
	numPoints = 0;

	mncudaSafeCallNoSync(pool.Request((void**)&d_pixels, maxPoints*sizeof(uint), "ShadingPts"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_idxTris, maxPoints*sizeof(int), "ShadingPts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_ptInter, maxPoints*sizeof(float4), "ShadingPts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_normalsG, maxPoints*sizeof(float4), "ShadingPts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_normalsS, maxPoints*sizeof(float4), "ShadingPts"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_baryHit, maxPoints*sizeof(float2), "ShadingPts"));

	return true;
}

void ShadingPoints::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_pixels));
	mncudaSafeCallNoSync(pool.Release(d_idxTris));
	mncudaSafeCallNoSync(pool.Release(d_ptInter));
	mncudaSafeCallNoSync(pool.Release(d_normalsG));
	mncudaSafeCallNoSync(pool.Release(d_normalsS));
	mncudaSafeCallNoSync(pool.Release(d_baryHit));
	maxPoints = 0;
	numPoints = 0;
}

void ShadingPoints::CompactSrcAddr(uint* d_srcAddr, uint countNew)
{
	if(countNew == 0)
	{
		numPoints = 0;
		return;
	}
	if(countNew == numPoints)
		return;

	// Move source data to destination data inplace.
	mncudaCompactInplace(d_pixels, d_srcAddr, numPoints, countNew);
	mncudaCompactInplace(d_idxTris, d_srcAddr, numPoints, countNew);
	mncudaCompactInplace(d_normalsG, d_srcAddr, numPoints, countNew);
	mncudaCompactInplace(d_normalsS, d_srcAddr, numPoints, countNew);
	mncudaCompactInplace(d_ptInter, d_srcAddr, numPoints, countNew);
	mncudaCompactInplace(d_baryHit, d_srcAddr, numPoints, countNew);

	// Update count.
	numPoints = countNew;
}

void ShadingPoints::Add(const ShadingPoints& other)
{
	if(numPoints + other.numPoints > maxPoints)
		MNFatal("Failed to add shading point lists. Too many shading points.");

	// Move source data to destination data.
	mncudaSafeCallNoSync(cudaMemcpy(d_pixels + numPoints, other.d_pixels,
		other.numPoints*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_idxTris + numPoints, other.d_idxTris,
		other.numPoints*sizeof(int), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_normalsG + numPoints, other.d_normalsG,
		other.numPoints*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_normalsS + numPoints, other.d_normalsS,
		other.numPoints*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_ptInter + numPoints, other.d_ptInter,
		other.numPoints*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_baryHit + numPoints, other.d_baryHit,
		other.numPoints*sizeof(float2), cudaMemcpyDeviceToDevice));

	// Update point count.
	numPoints += other.numPoints;
}

uint ShadingPoints::Merge(const ShadingPoints& other, uint* d_isValid)
{
	if(other.numPoints == 0)
		return 0;

	MNCudaMemory<uint> d_srcAddr(other.numPoints);
	uint countNew = mncudaGenCompactAddresses(d_isValid, other.numPoints, d_srcAddr);

	if(numPoints + countNew > maxPoints)
		MNFatal("Failed to merge shading point lists. Too many shading points.");
	if(countNew == 0)
		return 0;

	// Now move source data to destination data.
	mncudaSetFromAddress(d_pixels + numPoints, d_srcAddr, other.d_pixels, countNew);
	mncudaSetFromAddress(d_idxTris + numPoints, d_srcAddr, other.d_idxTris, countNew);
	mncudaSetFromAddress(d_normalsG + numPoints, d_srcAddr, other.d_normalsG, countNew);
	mncudaSetFromAddress(d_normalsS + numPoints, d_srcAddr, other.d_normalsS, countNew);
	mncudaSetFromAddress(d_ptInter + numPoints, d_srcAddr, other.d_ptInter, countNew);
	mncudaSetFromAddress(d_baryHit + numPoints, d_srcAddr, other.d_baryHit, countNew);

	// Update point count.
	numPoints += countNew;

	return countNew;
}

void ShadingPoints::SetFrom(const ShadingPoints& other, uint* d_srcAddr, uint numSrcAddr)
{
	if(numSrcAddr == 0)
	{
		numPoints = 0;
		return;
	}

	mncudaSetFromAddress(d_pixels, d_srcAddr, other.d_pixels, numSrcAddr);
	mncudaSetFromAddress(d_idxTris, d_srcAddr, other.d_idxTris, numSrcAddr);
	mncudaSetFromAddress(d_normalsG, d_srcAddr, other.d_normalsG, numSrcAddr);
	mncudaSetFromAddress(d_normalsS, d_srcAddr, other.d_normalsS, numSrcAddr);
	mncudaSetFromAddress(d_ptInter, d_srcAddr, other.d_ptInter, numSrcAddr);
	mncudaSetFromAddress(d_baryHit, d_srcAddr, other.d_baryHit, numSrcAddr);

	// Update count.
	numPoints = numSrcAddr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PairList implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool PairList::Initialize(uint _maxPairs)
{
	MNAssert(_maxPairs > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	maxPairs = MNCUDA_ALIGN(_maxPairs);
	numPairs = 0;

	mncudaSafeCallNoSync(pool.Request((void**)&d_first, maxPairs*sizeof(uint), "Wang algorithms"));
	mncudaSafeCallNoSync(pool.Request((void**)&d_second, maxPairs*sizeof(uint), "Wang algorithms"));

	return true;
}

void PairList::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_first));
	mncudaSafeCallNoSync(pool.Release(d_second));

	numPairs = 0;
	maxPairs = 0;
}

void PairList::SortByFirst(uint firstValueMax, uint sortSegmentSize/* = 0*/, uint* d_outSrcAddr/* = NULL*/)
{
	if(sortSegmentSize != 0 && numPairs % sortSegmentSize != 0)
		MNFatal("Sorting pair list: Illegal sort segment size.");
	MNCudaPrimitives& cp = MNCudaPrimitives::GetInstance();
	
	uint numToSort = numPairs;
	if(sortSegmentSize > 0)
		numToSort = sortSegmentSize;

	if(numToSort == 0)
		return;

	if(!d_outSrcAddr)
	{
		// No need to generate source addresses. Just use d_second as values.
		// Note that this also sorts d_first (keys)!

		// Handle the case where the offset would be not aligned to avoid CUDPP sort problems.
		if(sortSegmentSize > 0 && MNCUDA_ALIGN(numToSort) != numToSort)
		{
			MNCudaMemory<uint> d_temp1(numToSort), d_temp2(numToSort);
			for(uint offset=0; offset<numPairs; offset+=numToSort)
			{
				mncudaSafeCallNoSync(cudaMemcpy(d_temp1, d_first+offset, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				mncudaSafeCallNoSync(cudaMemcpy(d_temp2, d_second+offset, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				cp.Sort(d_temp1, d_temp2, firstValueMax, numToSort);
				mncudaSafeCallNoSync(cudaMemcpy(d_first+offset, d_temp1, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				mncudaSafeCallNoSync(cudaMemcpy(d_second+offset, d_temp2, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
			}
		}
		else
		{
			for(uint offset=0; offset<numPairs; offset+=numToSort)
				cp.Sort(d_first+offset, d_second+offset, firstValueMax, numToSort);
		}
	}
	else
	{
		// Generate source addresses.
		// Note that this also sorts d_first!
		mncudaInitIdentity(d_outSrcAddr, numPairs);

		// Handle the case where the offset would be not aligned to avoid CUDPP sort problems.
		if(sortSegmentSize > 0 && MNCUDA_ALIGN(numToSort) != numToSort)
		{
			MNCudaMemory<uint> d_temp1(numToSort), d_temp2(numToSort);
			for(uint offset=0; offset<numPairs; offset+=numToSort)
			{
				mncudaSafeCallNoSync(cudaMemcpy(d_temp1, d_first+offset, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				mncudaSafeCallNoSync(cudaMemcpy(d_temp2, d_outSrcAddr+offset, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				cp.Sort(d_temp1, d_temp2, firstValueMax, numToSort);
				mncudaSafeCallNoSync(cudaMemcpy(d_first+offset, d_temp1, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
				mncudaSafeCallNoSync(cudaMemcpy(d_outSrcAddr+offset, d_temp2, numToSort*sizeof(uint), cudaMemcpyDeviceToDevice));
			}
		}
		else
		{
			for(uint offset=0; offset<numPairs; offset+=numToSort)
				cp.Sort(d_first+offset, d_outSrcAddr+offset, firstValueMax, numToSort);
		}

		// Now use set from address to sort d_second.
		MNCudaMemory<uint> d_temp(numPairs);
		mncudaSafeCallNoSync(cudaMemcpy(d_temp, d_second, numPairs*sizeof(uint), cudaMemcpyDeviceToDevice));
		mncudaSetFromAddress<uint>(d_second, d_outSrcAddr, d_temp, numPairs);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// QuadTreeSP implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool QuadTreeSP::Initialize(uint sizeScreen)
{
	MNAssert(sizeScreen > 0);
	if(!IsPowerOf2(sizeScreen))
		MNFatal("Shading point quad tree requires screen size to be power of 2.");

	// Compute node count. The number of nodes is 4^0 + 4^1 + ... + 4^(numLevels-1), that is
	// numNodes = 1 + sum(i=1..numLevels-1, 4^i). This is the geometric sum, so we have
	// numNodes = 1 + (4^(numLevels) - 4) / (4 - 1).
	numLevels = Log2Int((float)sizeScreen) + 1;
	numNodes = 1 + ((uint)pow(4.0f, (float)numLevels) - 4) / 3;
	maxNodes = MNCUDA_ALIGN(numNodes);

	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Request((void**)&d_positions, maxNodes*sizeof(float4), "Wang algorithms", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_normals, maxNodes*sizeof(float4), "Wang algorithms", 256));
	mncudaSafeCallNoSync(pool.Request((void**)&d_geoVars, maxNodes*sizeof(float), "Wang algorithms"));

	return true;
}

void QuadTreeSP::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_positions));
	mncudaSafeCallNoSync(pool.Release(d_normals));
	mncudaSafeCallNoSync(pool.Release(d_geoVars));

	numNodes = 0;
}

uint QuadTreeSP::GetLevel(uint idxNode) const
{
	MNAssert(idxNode < numNodes);

	uint curLevel = 0;
	uint numNodesCur;
	while(idxNode >= (numNodesCur = (uint)pow(4.0f, (float)curLevel)))
	{
		idxNode -= numNodesCur;
		curLevel++;
	}

	return curLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ClusterList implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
bool ClusterList::Initialize(uint _maxClusters)
{
	MNAssert(_maxClusters > 0);
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	maxClusters = MNCUDA_ALIGN_EX(_maxClusters, pool.GetTextureAlignment()/sizeof(float));
	numClusters = 0;

	// Add space for a potentially required virutal cluster.
	uint realMax = maxClusters + 1;

	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_positions, realMax*sizeof(float4), "Wang algorithms"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_normals, realMax*sizeof(float4), "Wang algorithms"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_idxShadingPt, realMax*sizeof(uint), "Wang algorithms"));
	mncudaSafeCallNoSync(pool.RequestTexture((void**)&d_geoVarMax, realMax*sizeof(float), "Wang algorithms"));

	return true;
}

void ClusterList::CompactSrcAddr(uint* d_srcAddr, uint countNew)
{
	if(countNew == 0)
	{
		numClusters = 0;
		return;
	}
	if(countNew == numClusters)
		return;

	// Move source data to destination data inplace.
	mncudaCompactInplace(d_positions, d_srcAddr, numClusters, countNew);
	mncudaCompactInplace(d_normals, d_srcAddr, numClusters, countNew);
	mncudaCompactInplace(d_idxShadingPt, d_srcAddr, numClusters, countNew);
	mncudaCompactInplace(d_geoVarMax, d_srcAddr, numClusters, countNew);

	// Update count.
	numClusters = countNew;
}

void ClusterList::Add(const ClusterList& other)
{
	if(numClusters + other.numClusters > maxClusters)
		MNFatal("Failed to add cluster lists. Too many clusters.");

	// Move source data to destination data.
	mncudaSafeCallNoSync(cudaMemcpy(d_positions + numClusters, other.d_positions,
		other.numClusters*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_normals + numClusters, other.d_normals,
		other.numClusters*sizeof(float4), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_idxShadingPt + numClusters, other.d_idxShadingPt,
		other.numClusters*sizeof(uint), cudaMemcpyDeviceToDevice));
	mncudaSafeCallNoSync(cudaMemcpy(d_geoVarMax + numClusters, other.d_geoVarMax,
		other.numClusters*sizeof(float), cudaMemcpyDeviceToDevice));

	// Update count.
	numClusters += other.numClusters;
}

uint ClusterList::Merge(const ClusterList& other, uint* d_isValid)
{
	if(other.numClusters == 0)
		return 0;

	MNCudaMemory<uint> d_srcAddr(other.numClusters);
	uint countNew = mncudaGenCompactAddresses(d_isValid, other.numClusters, d_srcAddr);

	if(numClusters + countNew > maxClusters)
		MNFatal("Failed to merge cluster lists. Too many clusters.");
	if(countNew == 0)
		return 0;

	// Now move source data to destination data.
	mncudaSetFromAddress(d_positions + numClusters, d_srcAddr, other.d_positions, countNew);
	mncudaSetFromAddress(d_normals + numClusters, d_srcAddr, other.d_normals, countNew);
	mncudaSetFromAddress(d_idxShadingPt + numClusters, d_srcAddr, other.d_idxShadingPt, countNew);
	mncudaSetFromAddress(d_geoVarMax + numClusters, d_srcAddr, other.d_geoVarMax, countNew);

	// Update photon count.
	numClusters += countNew;

	return countNew;
}

void ClusterList::Destroy()
{
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	mncudaSafeCallNoSync(pool.Release(d_positions));
	mncudaSafeCallNoSync(pool.Release(d_normals));
	mncudaSafeCallNoSync(pool.Release(d_idxShadingPt));
	mncudaSafeCallNoSync(pool.Release(d_geoVarMax));

	maxClusters = 0;
	numClusters = 0;
}