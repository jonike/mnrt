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

#include "BasicScene.h"

BasicScene::BasicScene(void)
{
	m_numTris = 0;

	m_idxMaterial = NULL;
	for(uint i=0; i<3; i++)
	{
		m_v[i] = NULL;
		m_n[i] = NULL;
		m_texCoords[i] = NULL;
	}
}

BasicScene::~BasicScene(void)
{
	Unload();
}

bool BasicScene::Load(const std::string& strFile)
{
	if(IsLoaded())
		return false;

	m_strSceneFile = strFile;

	return true;
}

void BasicScene::Unload()
{
	if(!m_numTris)
		return;

	SAFE_DELETE_ARRAY(m_idxMaterial);
	for(uint i=0; i<3; i++)
	{
		SAFE_DELETE_ARRAY(m_v[i]);
		SAFE_DELETE_ARRAY(m_n[i]);
		SAFE_DELETE_ARRAY(m_texCoords[i]);
	}

	m_numTris = 0;
}