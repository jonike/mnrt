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
/// \file	MNRT\CameraModel.h
///
/// \brief	Declares the CameraModel class. 
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_CAMERAMODEL_H__
#define __MN_CAMERAMODEL_H__

#pragma once

#include "Geometry/MNGeometry.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	CameraModel
///
/// \brief	A simple camera model for MNRT.
/// 		
/// 		Basically, a LookAt()-approach is taken. That is, the user specifies the camera's
/// 		orientation using eye and look-at points and up direction. Additionally, the field of
/// 		view angle and the near and far clipping distances can be set. The class supports two
/// 		rotation methods and WASD keyboard movement. To pass camera information to kernels,
/// 		MNTransform objects can be generated. They can be passed to kernels in form of 4x4
/// 		matrices. 
///
/// \author	Mathias Neumann
/// \date	31.01.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class CameraModel
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	CameraModel(void)
	///
	/// \brief	Default constructor.
	///
	///			Calls 
	///
	///			\code LookAt(MNPoint3(0.f, 0.f, -10.f), MNPoint3(0.f, 0.f, 0.f), MNVector3(0.f, 1.f, 0.f)); \endcode
	///
	///			and sets FOV to 75 degree and clip distances to \c 0.3f and ::MN_INFINITY.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	CameraModel(void);
	~CameraModel(void);

private:
	/// Eye position (world space).
	MNPoint3 m_ptEye;
	/// Look at position (world space).
	MNPoint3 m_ptLookAt;
	/// Up direction (normalized).
	MNVector3 m_vecUp;
	/// \brief	Initial up direction (normalized). 
	///
	///			Set when calling LookAt(). This direction can be used to fix the left
	///			and right rotation around this initial up direction. It confirms to the
	///			default behaviour of the camera in games.
	MNVector3 m_vecInitialUp;
	/// \brief	View direction (normalized).
	///
	///			Derived from eye and look-at position.
	MNVector3 m_vecViewDir;

	/// \brief	Screen dimension in pixels (screen space).
	///
	///			Used for raster to camera space transformation.
	uint m_screenW, m_screenH;
	///	\brief	"Normalized" screen extent.
	///
	///			Stores extent of screen in normalized coordinates, scaled by aspect ratio.
	///			See \ref lit_pharr "[Pharr and Humphreys 2004]" for details.
	float m_ScreenExtent[4];
	/// FOV angle (radians).
	float m_fFOV;
	/// Near clip distance (hither).
	float m_fClipHither;
	/// Far clip distance (yon).
	float m_fClipYon;

	/// If set, the initial up direction is used for left and right rotation.
	bool m_bUseInitialUp;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void LookAt(MNPoint3 ptEye, MNPoint3 ptLookAt, MNVector3 vecUp)
	///
	/// \brief	Sets the camera's position and orientation.	
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	///
	/// \param	ptEye		Eye position in world space.
	/// \param	ptLookAt	Look-at position in world space.
	/// \param	vecUp		The up direction. Can be unnormalized, but may not be zero.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void LookAt(MNPoint3 ptEye, MNPoint3 ptLookAt, MNVector3 vecUp);

	/// Returns the current eye position in world space.
	MNPoint3 GetEye() const { return m_ptEye; }
	/// Returns the current look-at position in world space.
	MNPoint3 GetLookAt() const { return m_ptLookAt; }
	/// Returns the current up direction (normalized).
	MNVector3 GetUp() const { return m_vecUp; }
	/// Returns the current viewing direction (normalized).
	MNVector3 GetViewDirection() const { return m_vecViewDir; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNTransform GetCamera2World() const
	///
	/// \brief	Computes and returns the camera space to world space transformation.
	///
	///			Camera space is the space with the camera's eye at the origin, the z axis as the
	///			viewing direction and the y axis as up direction.
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	/// \see	MNTransform::LookAt(), MNTransform::Inverse()
	///
	/// \return	The transformation.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNTransform GetCamera2World() const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNTransform GetRaster2Camera() const
	///
	/// \brief	Computes and returns the raster space to camera space transformation.
	///
	///			Raster space is a two-dimensional space where x and y coordinates range from 0 to
	///			the corresponding image resolution (screen dimension).
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	///
	/// \return	The transformation.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNTransform GetRaster2Camera() const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetScreenDimension(uint screenW, uint screenH)
	///
	/// \brief	Sets the screen's dimension or resolution.
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	///
	/// \param	screenW	Screen width in pixels. 
	/// \param	screenH	Screen height in pixels. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetScreenDimension(uint screenW, uint screenH);
	/// Returns the screen's width in pixels.
	uint GetScreenWidth() const { return m_screenW; }
	/// Returns the screen's height in pixels.
	uint GetScreenHeight() const { return m_screenH; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetFOV(float fov_rad)
	///
	/// \brief	Sets the field of view (FOV) angle.
	///
	/// \author	Mathias Neumann
	/// \date	March 2010
	///
	/// \param	fov_rad	FOV angle in radians. Has to be positive and smaller than ::MN_PI.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetFOV(float fov_rad);
	/// Returns the current field of view (FOV) angle.
	float GetFOV() const { return m_fFOV; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void SetClipDistances(float clipHither, float clipYon)
	///
	/// \brief	Sets clipping plane distances. 
	///
	///			The clipping planes define the z-axis range of space that will be visible in the
	///			generated image. All scene contents within this camera space range will be projected
	///			onto the image plane at z = \a clipHither.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	///
	/// \param	clipHither	Distance to near clipping plane.
	/// \param	clipYon		Distance to far clipping plane.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void SetClipDistances(float clipHither, float clipYon);
	/// Returns near (hither) clipping plane distance on camera space z-axis.
	float GetClipHither() const { return m_fClipHither; }
	/// Returns far (yon) clipping plane distance on camera space z-axis.
	float GetClipYon() const { return m_fClipYon; }

	/// Sets whether to use the initial up direction for left/right rotation within RotateAroundFixedEye().
	void SetUseInitialUp(bool bSet) { m_bUseInitialUp = bSet; }
	/// Gets whether the initial up direction is used for left/right rotation within RotateAroundFixedEye().
	bool GetUseInitialUp() const { return m_bUseInitialUp; }

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void RotateAroundAt(float angleLR_rad, float angleUD_rad)
	///
	/// \brief	Performs an orbital rotation around the look-at position. The look-at position is fixed.
	///
	/// \author	Mathias Neumann
	/// \date	Februrary 2010
	///
	/// \param	angleLR_rad	The left/right rotation angle in radians. 
	/// \param	angleUD_rad	The up/down rotation angle in radians. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void RotateAroundAt(float angleLR_rad, float angleUD_rad);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void RotateAroundFixedEye(float angleLR_rad, float angleUD_rad)
	///
	/// \brief	Rotates the camera with fixed exe position.
	///
	///			This corresponds to the rotation used in computer games, especially when
	///			rotation is performed around initial up direction (see SetUseInitialUp()).
	///
	/// \author	Mathias Neumann
	/// \date	Februrary 2010
	///
	/// \param	angleLR_rad	The left/right rotation angle in radians. 
	/// \param	angleUD_rad	The up/down rotation angle in radians. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void RotateAroundFixedEye(float angleLR_rad, float angleUD_rad);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool ProcessKey(int key_ASCII, float fTransFactor = 0.01f)
	///
	/// \brief	Processes keyboard input for WASD camera movement.
	///
	///			Very simple camera movement (i.e. eye and look-at position translation) using the
	///			following keys:
	///
	///			\li \c w: Move camera forward along viewing direction.
	///			\li \c a: Move camera to the left (strafing).
	///			\li \c s: Move camera backward along viewing direction.
	///			\li \c d: Move camera to the right (strafing).
	///
	///			All other keyboard inputs are ignored.
	///
	/// \author	Mathias Neumann
	/// \date	Februrary 2010
	///
	/// \param	key_ASCII		The ASCII key code to process. 
	/// \param	fTransFactor	The translation factor. Simple way to adjust camera translation speed.
	///
	/// \return \c true if the key was used to change the camera's parameters, else \c false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool ProcessKey(int key_ASCII, float fTransFactor = 0.01f);

private:
	void TransformView(const MNTransform& trans);
	MNTransform Perspective(float fov, float n, float f) const;
};


#endif //__MN_CAMERAMODEL_H__