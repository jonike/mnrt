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
/// \file	Geometry\MNTransform.h
///
/// \brief	Declares MNTransform class. 
/// \author	Mathias Neumann
/// \date	07.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_TRANSFORM_H__
#define __MN_TRANSFORM_H__

#pragma once

#include "MNPoint3.h"
#include "MNVector3.h"
#include "MNNormal3.h"
#include "MNRay.h"
#include "MNBBox.h"
#include "MNMatrix4x4.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNTransform
///
/// \brief	Class for geometric transformations.
/// 		
/// 		Holds a 4x4 matrix and it's inverse. It is designed so that the matrices cannot be
/// 		changed once initialized. Modelled after \ref lit_pharr "[Pharr and Humphreys 2004]". 
///
/// \author	Mathias Neumann
/// \date	07.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNTransform
{
public:
	/// Default constructor. Creates an identity transform.
	MNTransform(void);
	/// Constructs a transform from the given matrix entries.
	MNTransform(float mat[4][4]);
	/// Constructs a transform from the matrix.
	MNTransform(const MNMatrix4x4& mat);
	/// Constructs a transform from the matrix and inverse. For performance reasons it is
	/// not verified that \a matInv is the correct inverse of \a mat.
	MNTransform(const MNMatrix4x4& mat, const MNMatrix4x4& matInv);
	~MNTransform(void) {}

// Data members
private:
	// The matrix that defines the transform.
	MNMatrix4x4 m;
	// The inverse matrix that defines the inverse transform. Stored for
	// performance reasons as we need it to transform normals.
	MNMatrix4x4 mInv;

// Accessors
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	float GetMatrix(int i, int j) const
	///
	/// \brief	Gets a matrix element of the transformation matrix. 
	///
	/// \author	Mathias Neumann
	/// \date	05.04.2010
	///
	/// \param	i	Row index, where 0 <= i < 4. 
	/// \param	j	Column index, where 0 <= i < 4. 
	///
	/// \return	The element. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float GetMatrix(int i, int j) const { return m.m[i][j]; }

// Operators
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNTransform operator*(const MNTransform& t2) const
	///
	/// \brief	Generates a transform by combining this transform with \a t2 (new = \c this * \a t2).
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	t2	The second transform. 
	///
	/// \return	The result of the combination. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNTransform operator*(const MNTransform& t2) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	inline MNPoint3 operator()(const MNPoint3& pt) const
	///
	/// \brief	Transform operator for points.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	pt		The point to transform. 
	///
	/// \return	The transformed point.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNPoint3 operator()(const MNPoint3& pt) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	inline void operator()(const MNPoint3& pt, MNPoint3* ptTrans) const
	///
	/// \brief	Transform operator with parameter output for points.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param			pt		The point to transform. 
	/// \param [out]	ptTrans	The transformed point.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void operator()(const MNPoint3& pt, MNPoint3* ptTrans) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	inline MNVector3 operator()(const MNVector3& v) const
	///
	/// \brief	Transform operator for vectors.
	///
	/// \author	Mathias
	/// \date	07.02.2010
	///
	/// \param	v		The vector to transform. 
	///
	/// \return	The transformed vector.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNVector3 operator()(const MNVector3& v) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	inline void operator()(const MNVector3& v, MNVector3* vTrans) const
	///
	/// \brief	Transform operator with parameter output for vectors.
	///
	/// \author	Mathias
	/// \date	07.02.2010
	///
	/// \param			v		The vector to transform. 
	/// \param [out]	vTrans	The transformed vector.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void operator()(const MNVector3& v, MNVector3* vTrans) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNNormal3 operator()(const MNNormal3& n) const
	///
	/// \brief	Transforms the given normal.
	/// 		
	/// 		Transformation is done using the inverse transpose of the transformation matrix. This
	/// 		is done because for a tangent vector t: Dot(n, t) = 0. The transformed tangent vector
	/// 		is Mt for the transform M. We have to transform n using a matrix S, so that 
	///			Dot(Sn, Mt) = (Sn)^T(Mt) = (n^T)(S^T)Mt = 0. That's the case when S = (M^-1)^T. 
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	n	The normal to transform. 
	///
	/// \return	The transformed normal.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNNormal3 operator()(const MNNormal3& n) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void operator()(const MNNormal3& n, MNNormal3* nTrans) const
	///
	/// \brief	Transforms the given normal using the inverse transpose of the transformation matrix.
	/// 		Output is returned in \a nTrans. 
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param			n		The normal to transform. 
	/// \param [out]	nTrans	The transformed normal.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void operator()(const MNNormal3& n, MNNormal3* nTrans) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNRay operator()(const MNRay& r) const
	///
	/// \brief	Transforms the given ray and returns the result.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	r		The ray to transform. 
	///
	/// \return	The transformed ray.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNRay operator()(const MNRay& r) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void operator()(const MNRay& r, MNRay* rTrans) const
	///
	/// \brief	Transforms the given ray and returns the result in \a rTrans.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param			r		The ray to transform. 
	/// \param [out]	rTrans	The transformed ray. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void operator()(const MNRay& r, MNRay* rTrans) const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNBBox operator()(const MNBBox& box) const
	///
	/// \brief	Transforms the given bounding box.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	box		The AABB to transform. 
	///
	/// \return	The transformed AABB.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNBBox operator()(const MNBBox& box) const;

// Operations
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNTransform GetInverse() const
	///
	/// \brief	Returns the inverse transformation to this transformation. 
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \return	The inverse transform.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNTransform GetInverse() const
	{
		return MNTransform(mInv, m);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool SwapsHandedness() const
	///
	/// \brief	Checks if this transform swaps coordinate system handedness.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \return	Returns true if this transform swaps coordinate system handedness, else false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool SwapsHandedness() const;

// Factory methods
public:

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform Translate(const MNVector3& delta)
	///
	/// \brief	Constructs a translation transformation.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	delta	The translation delta as a vector. 
	///
	/// \return	The transform object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform Translate(const MNVector3& delta);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform Scale(float x, float y, float z)
	///
	/// \brief	Constructs a scaling transformation.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	x	The x axis scaling factor. 
	/// \param	y	The y axis scaling factor. 
	/// \param	z	The z axis scaling factor. 
	///
	/// \return	The transform object.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform Scale(float x, float y, float z);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform RotateX(float angle_rad)
	///
	/// \brief	Constructs a rotation around x axis transform.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	angle_rad	The rotation angle in radians. 
	///
	/// \return	The transform object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform RotateX(float angle_rad);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform RotateY(float angle_rad)
	///
	/// \brief	Constructs a rotation around y axis transform.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	angle_rad	The rotation angle in radians. 
	///
	/// \return	The transform object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform RotateY(float angle_rad);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform RotateZ(float angle_rad)
	///
	/// \brief	Constructs a rotation around z axis transform.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	angle_rad	The rotation angle in radians. 
	///
	/// \return	The transform object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform RotateZ(float angle_rad);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform Rotate(const MNVector3& axis, float angle_rad)
	///
	/// \brief	Constructs a rotation around an arbitrary axis transform.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	axis		The axis to rotate around.
	/// \param	angle_rad	The rotation angle in radians. 
	///
	/// \return	The transform object. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform Rotate(const MNVector3& axis, float angle_rad);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNVector3 RotateVector(const MNVector3& vecToRotate, const MNVector3& vecAxis,
	/// 	const float angle)
	///
	/// \brief	Rotates the given vector around the specified axis. 
	///
	///			This method simplifies the rotation by hiding the corresponding MNTransform object.
	///
	/// \author	Mathias Neumann
	/// \date	31.01.2010
	///
	/// \param	vecToRotate	The vector to rotate. 
	/// \param	vecAxis		The rotation axis. 
	/// \param	angle		The rotation angle in radians. 
	///
	/// \return	The rotated vector. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNVector3 RotateVector(const MNVector3& vecToRotate, const MNVector3& vecAxis, const float angle);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNTransform LookAt(const MNPoint3& eye, const MNPoint3& at, const MNVector3& up)
	///
	/// \brief	Generates a look-at transformation. It can be used to transform points from \em world
	///			to \em camera space.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	eye	The eye position. 
	/// \param	at	The look-at position where the camera looks at. 
	/// \param	up	The up vector that orients the camera along the viewing direction. 
	///
	/// \return	A transform from world to camera space. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNTransform LookAt(const MNPoint3& eye, const MNPoint3& at, const MNVector3& up);
};



inline MNPoint3 MNTransform::operator()(const MNPoint3& pt) const
{
	float x = pt.x, y = pt.y, z = pt.z;

	// The homogeneous representation for points is [x, y, z, 1]^T.
	float xp = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
	float yp = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
	float zp = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
	float wp = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];

	MNAssert(wp != 0);
	// Avoid division if possible.
	if(wp == 1.f)
		return MNPoint3(xp, yp, zp);
	else
		return MNPoint3(xp/wp, yp/wp, zp/wp);
}

inline void MNTransform::operator()(const MNPoint3& pt, MNPoint3* ptTrans) const
{
	// Read out to allow inplace transformation.
	float x = pt.x, y = pt.y, z = pt.z;

	// The homogeneous representation for points is [x, y, z, 1]^T.
	ptTrans->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
	ptTrans->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
	ptTrans->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
	float w    = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];

	MNAssert(w != 0);
	if(w != 1.f)
		*ptTrans /= w;
}

inline MNVector3 MNTransform::operator()(const MNVector3& v) const
{
	float x = v.x, y = v.y, z = v.z;

	// The homogeneous representation for vectors is [x, y, z, 0]^T. Therefore
	// there is no need to compute the w coordinate. This simplifies the
	// transform.
	return MNVector3(m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z,
					 m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z,
					 m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z);
}

inline void MNTransform::operator()(const MNVector3& v, MNVector3* vTrans) const
{
	// Read out to allow inplace transformation.
	float x = v.x, y = v.y, z = v.z;

	// The homogeneous representation for vectors is [x, y, z, 0]^T. Therefore
	// there is no need to compute the w coordinate. This simplifies the
	// transform.
	vTrans->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z;
	vTrans->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z;
	vTrans->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z;
}

inline MNNormal3 MNTransform::operator()(const MNNormal3& n) const
{
	float x = n.x, y = n.y, z = n.z;

	// Note the swapped indices (for transpose).
	return MNNormal3(mInv.m[0][0]*x + mInv.m[1][0]*y + mInv.m[2][0]*z,
					 mInv.m[0][1]*x + mInv.m[1][1]*y + mInv.m[2][1]*z,
					 mInv.m[0][2]*x + mInv.m[1][2]*y + mInv.m[2][2]*z);
}

inline void MNTransform::operator()(const MNNormal3& n, MNNormal3* nTrans) const
{
	// Read out to allow inplace transformation.
	float x = n.x, y = n.y, z = n.z;

	// Note the swapped indices (for transpose).
	nTrans->x = mInv.m[0][0]*x + mInv.m[1][0]*y + mInv.m[2][0]*z;
	nTrans->y = mInv.m[0][1]*x + mInv.m[1][1]*y + mInv.m[2][1]*z;
	nTrans->z = mInv.m[0][2]*x + mInv.m[1][2]*y + mInv.m[2][2]*z;
}

inline MNRay MNTransform::operator()(const MNRay& r) const
{
	MNRay ret;
	(*this)(r.o, &ret.o);
	(*this)(r.d, &ret.d);
	return ret;
}

inline void MNTransform::operator()(const MNRay& r, MNRay* rTrans) const
{
	(*this)(r.o, &rTrans->o);
	(*this)(r.d, &rTrans->d);
}


#endif // __MN_TRANSFORM_H__