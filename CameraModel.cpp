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

#include "CameraModel.h"
#include "Geometry/MNPlane.h"


CameraModel::CameraModel(void)
{
	LookAt(MNPoint3(0.f, 0.f, -10.f), MNPoint3(0.f, 0.f, 0.f), MNVector3(0.f, 1.f, 0.f));
	SetFOV(Radians(75.0f));
	SetScreenDimension(100, 100);
	SetClipDistances(0.3f, MN_INFINITY);

	m_bUseInitialUp = true;
}

CameraModel::~CameraModel(void)
{
}

void CameraModel::LookAt(MNPoint3 ptEye, MNPoint3 ptLookAt, MNVector3 vecUp)
{
	MNAssert(!(ptEye.x == ptLookAt.x && ptEye.y == ptLookAt.y && ptEye.z == ptLookAt.z));

	m_ptEye = ptEye;
	m_ptLookAt = ptLookAt;
	m_vecUp = Normalize(vecUp);
	m_vecInitialUp = m_vecUp;

	// Update the view direction.
	m_vecViewDir = Normalize(m_ptLookAt - m_ptEye);
}

void CameraModel::SetScreenDimension(uint screenW, uint screenH)
{
	m_screenW = screenW;
	m_screenH = screenH;

	// Update screen extent.
	float aspectRatio = float(m_screenW) / float(m_screenH);
	if(aspectRatio > 1.f) 
	{
		m_ScreenExtent[0] = -aspectRatio;
		m_ScreenExtent[1] =  aspectRatio;
		m_ScreenExtent[2] = -1.f;
		m_ScreenExtent[3] =  1.f;
	}
	else 
	{
		m_ScreenExtent[0] = -1.f;
		m_ScreenExtent[1] =  1.f;
		m_ScreenExtent[2] = -1.f / aspectRatio;
		m_ScreenExtent[3] =  1.f / aspectRatio;
	}
}

void CameraModel::SetFOV(float fov_rad)
{
	if(fov_rad <= 0 || fov_rad >= MN_PI)
	{
		MNError("Illegal FOV parameter.");
		return;
	}

	m_fFOV = fov_rad;
}

void CameraModel::SetClipDistances(float clipHither, float clipYon)
{
	if(clipHither <= 0 || clipHither >= clipYon)
	{
		MNError("Illegal clipping distances.");
		return;
	}

	m_fClipHither = clipHither;
	m_fClipYon = clipYon;
}

MNTransform CameraModel::GetCamera2World() const
{
	return MNTransform::LookAt(m_ptEye, m_ptLookAt, m_vecUp).GetInverse();
}

MNTransform CameraModel::GetRaster2Camera() const
{
	MNTransform cam2screen = Perspective(m_fFOV, m_fClipHither, m_fClipYon);

	// Compute projective camera screen transformations. We don't swap the y coordinates
	// here, opposed to what is done in "Physically based rendering". It is not required
	// since we don't have an bottom->up image space. See PBR p260.
	MNTransform screen2raster = 
		MNTransform::Scale(float(m_screenW), float(m_screenH), 1.f) *
		MNTransform::Scale(1.f / (m_ScreenExtent[1] - m_ScreenExtent[0]),
						   1.f / (m_ScreenExtent[3] - m_ScreenExtent[2]), 1.f) *
		MNTransform::Translate(MNVector3(-m_ScreenExtent[0], -m_ScreenExtent[2], 0.f));
	MNTransform raster2screen = screen2raster.GetInverse();

	// A := cam2screen.GetInverse() = screen to camera
	// B := raster2screen = raster to screen
	// x := raster space point
	// A(B(x)) = A(y) = z = (A*B)x
	// here y is a screen space point and z is a camera space point.
	return cam2screen.GetInverse() * raster2screen;
}

void CameraModel::RotateAroundAt(float angleLR_rad, float angleUD_rad)
{
	MNVector3 vecNew = m_ptEye - m_ptLookAt;

	// Rotate left/right around up vector.
	if(angleLR_rad != 0)
		vecNew = MNTransform::RotateVector(vecNew, m_vecUp, angleLR_rad);

	MNVector3 vecCross = Cross(vecNew, m_vecUp);

	// Rotate up/down around cross vector.
	if(angleUD_rad != 0)
	{
		vecNew = MNTransform::RotateVector(vecNew, vecCross, angleUD_rad);
		m_vecUp = MNTransform::RotateVector(m_vecUp, vecCross, angleUD_rad);
	}

	m_ptEye = m_ptLookAt + vecNew;

	// Update the view direction.
	m_vecViewDir = Normalize(m_ptLookAt - m_ptEye);
}

void CameraModel::RotateAroundFixedEye(float angleLR_rad, float angleUD_rad)
{
	// Rotate left/right around (initial) up vector.
	MNVector3 vecAxis = (m_bUseInitialUp ? m_vecInitialUp : m_vecUp);
	MNTransform transRotate = MNTransform::Rotate(vecAxis, angleLR_rad);
	transRotate = transRotate.GetInverse();
	TransformView(transRotate);

    // Perform up/down rotation around view_dir cross up.
    vecAxis = Cross(m_vecViewDir, m_vecUp);
    Normalize(vecAxis);
	transRotate = MNTransform::Rotate(vecAxis, angleUD_rad);
	transRotate = transRotate.GetInverse();
	TransformView(transRotate);
}

bool CameraModel::ProcessKey(int key_ASCII, float fTransFactor/* = 0.01f*/)
{
	MNVector3 vecTrans = MNVector3(0.f, 0.f, 0.f);

	MNVector3 vecStrafe = Cross(m_vecViewDir, m_vecUp);
    Normalize(vecStrafe);

	switch(key_ASCII)
	{
	case 115: // s
		vecTrans = -fTransFactor * m_vecViewDir;
		break;
	case 119: // w
		vecTrans = fTransFactor * m_vecViewDir;
		break;
	case 97: // a
		vecTrans = -fTransFactor * vecStrafe;
		break;
	case 100: // d
		vecTrans = fTransFactor * vecStrafe;
		break;
	default:
		break;
	}

	m_ptEye += vecTrans;
	m_ptLookAt += vecTrans;

	return (vecTrans.LengthSquared() > 0.f);
}

void CameraModel::TransformView(const MNTransform& trans)
{
	// Rotate vectors.
    m_vecUp = trans(m_vecUp);
	Normalize(m_vecUp);
    m_vecViewDir = trans(m_vecViewDir);
	Normalize(m_vecViewDir);
 
    // Get distance from eye to lookat.
    float distEye2Lookat = Distance(m_ptLookAt, m_ptEye);
    m_ptLookAt = m_ptEye + m_vecViewDir * distEye2Lookat;
}

MNTransform CameraModel::Perspective(float fov, float n, float f) const
{
	// Compute basic perspective matrix.
	float invDenom = 1.f / (f - n);
	MNMatrix4x4 matPersp(1.f, 0.f,        0.f,           0.f,
						 0.f, 1.f,        0.f,           0.f,
						 0.f, 0.f, f*invDenom, -f*n*invDenom,
						 0.f, 0.f,        1.f,           0.f);

	// Scale to canonical viewing volume.
	float invTanAng = 1.f / tanf(m_fFOV / 2.f);
	return MNTransform::Scale(invTanAng, invTanAng, 1.f) * MNTransform(matPersp);
}