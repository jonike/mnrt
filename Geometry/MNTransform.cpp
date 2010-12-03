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

#include "MNTransform.h"

MNTransform::MNTransform(void)
{
	// Note that both matrices can reference the same matrix since they
	// cannot be changed outside of this class.
	mInv = m;
}

MNTransform::MNTransform(float mat[4][4])
	: m(mat)
{ 
	mInv = m.Inverse();
}

MNTransform::MNTransform(const MNMatrix4x4& mat)
{
	m = mat;
	mInv = m.Inverse();
}

MNTransform::MNTransform(const MNMatrix4x4& mat, 
						 const MNMatrix4x4& matInv)
{
	m = mat;
	mInv = matInv;
}

MNTransform MNTransform::operator*(const MNTransform& t2) const
{
	MNMatrix4x4 mNew = m * t2.m;
	// (AB)^-1 = B^-1 A^-1
	MNMatrix4x4 mNewInv = t2.mInv * mInv;
	return MNTransform(mNew, mNewInv);
}

MNBBox MNTransform::operator()(const MNBBox& box) const
{
	const MNTransform& M = *this;

	// Transform origin and three axis vectors.
	MNPoint3 transOrigin = M(box.ptMin);
	MNVector3 transX = M(MNVector3(box.ptMax.x-box.ptMin.x, 0.f, 0.f));
	MNVector3 transY = M(MNVector3(0.f, box.ptMax.y-box.ptMin.y, 0.f));
	MNVector3 transZ = M(MNVector3(0.f, 0.f, box.ptMax.z-box.ptMin.z));

	MNBBox ret(transOrigin);
	ret = Union(ret, transOrigin + transX);
	ret = Union(ret, transOrigin + transY);
	ret = Union(ret, transOrigin + transZ);
	return ret;
}


bool MNTransform::SwapsHandedness() const
{
	// The handedness is changed only when the determinant of the upper 3x3 matrix
	// is negative.
	float det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
				 m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
				 m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));

	return det < 0.f;
}



/*static*/ MNTransform MNTransform::Translate(const MNVector3& delta)
{
	// Modifies points only.
	MNMatrix4x4 m(1, 0, 0, delta.x,
				  0, 1, 0, delta.y,
				  0, 0, 1, delta.z,
				  0, 0, 0,       1);

	MNMatrix4x4 mInv(1, 0, 0, -delta.x,
					 0, 1, 0, -delta.y,
					 0, 0, 1, -delta.z,
					 0, 0, 0,        1);

	return MNTransform(m, mInv);
}

/*static*/ MNTransform MNTransform::Scale(float x, float y, float z)
{
	// Modifies both points and vectors.
	MNMatrix4x4 m(x, 0, 0, 0,
				  0, y, 0, 0,
				  0, 0, z, 0,
				  0, 0, 0, 1);

	MNMatrix4x4 mInv(1.f/x,     0,     0, 0,
					 0,		1.f/y,     0, 0,
					 0,			0, 1.f/z, 0,
					 0,			0,     0, 1);

	return MNTransform(m, mInv);
}

/*static*/ MNTransform MNTransform::RotateX(float angle_rad)
{
	float fSin = sinf(angle_rad);
	float fCos = cosf(angle_rad);
	MNMatrix4x4 m(1,    0,     0, 0,
				  0, fCos, -fSin, 0,
				  0, fSin,  fCos, 0,
				  0,    0,     0, 1);

	// As rotation matrix, m is orthogonal. Therefore transpose(m) = m_inv.
	return MNTransform(m, m.Transpose());
}

/*static*/ MNTransform MNTransform::RotateY(float angle_rad)
{
	float fSin = sinf(angle_rad);
	float fCos = cosf(angle_rad);
	MNMatrix4x4 m(fCos, 0, fSin, 0,
					 0, 1,    0, 0,
				 -fSin, 0, fCos, 0,
				     0, 0,    0, 1);

	// As rotation matrix, m is orthogonal. Therefore transpose(m) = m_inv.
	return MNTransform(m, m.Transpose());
}

/*static*/ MNTransform MNTransform::RotateZ(float angle_rad)
{
	float fSin = sinf(angle_rad);
	float fCos = cosf(angle_rad);
	MNMatrix4x4 m(fCos, -fSin, 0, 0,
				  fSin,  fCos, 0, 0,
				     0,     0, 1, 0,
					 0,     0, 0, 1);

	// As rotation matrix, m is orthogonal. Therefore transpose(m) = m_inv.
	return MNTransform(m, m.Transpose());
}

/*static*/ MNTransform MNTransform::Rotate(const MNVector3& axis, 
										   float angle_rad)
{
	MNVector3 a = Normalize(axis);
	float fSin = sinf(angle_rad);
	float fCos = cosf(angle_rad);
	
	// Check "Physically based Rendering", Matt Pharr and Greg Humphreys, 2004, page 73,
	// for the details.
	MNMatrix4x4 mat;
	mat.m[0][0] = a.x * a.x + (1.f - a.x * a.x) * fCos;
	mat.m[0][1] = a.x * a.y * (1.f - fCos) - a.z * fSin;
	mat.m[0][2] = a.x * a.z * (1.f - fCos) + a.y * fSin;
	mat.m[0][3] = 0;

	mat.m[1][0] = a.x * a.y * (1.f - fCos) + a.z * fSin;
	mat.m[1][1] = a.y * a.y + (1.f - a.y * a.y) * fCos;
	mat.m[1][2] = a.y * a.z * (1.f - fCos) - a.x * fSin;
	mat.m[1][3] = 0;

	mat.m[2][0] = a.x * a.z * (1.f - fCos) - a.y * fSin;
	mat.m[2][1] = a.y * a.z * (1.f - fCos) + a.x * fSin;
	mat.m[2][2] = a.z * a.z + (1.f - a.z * a.z) * fCos;
	mat.m[2][3] = 0;

	mat.m[3][0] = 0;
	mat.m[3][1] = 0;
	mat.m[3][2] = 0;
	mat.m[3][3] = 1;

	// As rotation matrix, m is orthogonal. Therefore transpose(m) = m_inv.
	return MNTransform(mat, mat.Transpose());
}

/*static*/ MNVector3 MNTransform::RotateVector(const MNVector3& vecToRotate, 
											   const MNVector3& vecAxis, 
											   const float angle)
{
	MNTransform tRotate = MNTransform::Rotate(vecAxis, angle);
	return tRotate(vecToRotate);
}

/*static*/ MNTransform MNTransform::LookAt(const MNPoint3& eye, 
										   const MNPoint3& at, 
										   const MNVector3& up)
{
	MNMatrix4x4 cam2world;

	// Forth column determines how the camera space origin [0, 0, 0, 1]^T is mapped in
	// world space. We have to map it to eye.
	cam2world.m[0][3] = eye.x;
	cam2world.m[1][3] = eye.y;
	cam2world.m[2][3] = eye.z;
	cam2world.m[3][3] = 1;

	// 3rd column (z axis) maps to normalized viewing direction.
	MNVector3 dir = Normalize(at - eye);
	// 1st column (x axis) maps to right axis.
	MNVector3 right = Cross(dir, Normalize(up));
	// 2nd column (y axis) maps to recomputes up vector.
	MNVector3 newUp = Cross(right, dir);

	cam2world.m[0][0] = right.x;
	cam2world.m[1][0] = right.y;
	cam2world.m[2][0] = right.z;
	cam2world.m[3][0] = 0;

	cam2world.m[0][1] = newUp.x;
	cam2world.m[1][1] = newUp.y;
	cam2world.m[2][1] = newUp.z;
	cam2world.m[3][1] = 0;

	cam2world.m[0][2] = dir.x;
	cam2world.m[1][2] = dir.y;
	cam2world.m[2][2] = dir.z;
	cam2world.m[3][2] = 0;

	// Compute the inverse, that is the world 2 cam transformation.
	MNMatrix4x4 world2cam = cam2world.Inverse();

	return MNTransform(world2cam, cam2world);
}