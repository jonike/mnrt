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

#include "MNMatrix4x4.h"
#include "../MNUtilities.h"


MNMatrix4x4::MNMatrix4x4(void)
{
	// Initialize with identity matrix.
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] = ((i == j)?1.f:0.f);
}

MNMatrix4x4::MNMatrix4x4(float mat[4][4])
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] = mat[i][j];
}

MNMatrix4x4::MNMatrix4x4(float m00, float m01, float m02, float m03,
						 float m10, float m11, float m12, float m13,
						 float m20, float m21, float m22, float m23,
						 float m30, float m31, float m32, float m33)
{
	m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
	m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
	m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
	m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
}

// Add
MNMatrix4x4 MNMatrix4x4::operator+(const MNMatrix4x4& mat) const
{
	MNMatrix4x4 mNew(*this);
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			mNew.m[i][j] += mat.m[i][j];
	return mNew;
}

MNMatrix4x4& MNMatrix4x4::operator+=(const MNMatrix4x4& mat)
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] += mat.m[i][j];
	return *this;
}

// Subtract
MNMatrix4x4 MNMatrix4x4::operator-(const MNMatrix4x4& mat) const
{
	MNMatrix4x4 mNew(*this);
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			mNew.m[i][j] -= mat.m[i][j];
	return mNew;
}

MNMatrix4x4& MNMatrix4x4::operator-=(const MNMatrix4x4& mat)
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] -= mat.m[i][j];
	return *this;
}

// Scaling
MNMatrix4x4 MNMatrix4x4::operator*(float f) const
{
	MNMatrix4x4 mNew(*this);
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			mNew.m[i][j] *= f;
	return mNew;
}

MNMatrix4x4& MNMatrix4x4::operator*=(float f)
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] *= f;
	return *this;
}

// Multiplication
MNMatrix4x4 MNMatrix4x4::operator*(const MNMatrix4x4& mat) const
{
	float res[4][4];
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			res[i][j] = m[i][0] * mat.m[0][j] +
						m[i][1] * mat.m[1][j] +
						m[i][2] * mat.m[2][j] +
						m[i][3] * mat.m[3][j];
	return MNMatrix4x4(res);
}

MNMatrix4x4& MNMatrix4x4::operator*=(const MNMatrix4x4& mat)
{
	float res[4][4];
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			res[i][j] = m[i][0] * mat.m[0][j] +
						m[i][1] * mat.m[1][j] +
						m[i][2] * mat.m[2][j] +
						m[i][3] * mat.m[3][j];

	// Now assign.
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] = res[i][j];
	return *this;
}

// Transpose
MNMatrix4x4 MNMatrix4x4::Transpose() const
{
	float res[4][4];
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			res[i][j] = m[j][i];
	return MNMatrix4x4(res);
}

// Inverse
MNMatrix4x4 MNMatrix4x4::Inverse() const
{
	// Temporary array for pairs.
	float tmp[12];
	// Transpose source matrix.
	float src[16];
	// Determinant.
	float det;
	// Inverse matrix target.
	float dst[16];

	// Transpose matrix.
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++) 
			src[4*j + i] = m[i][j];

	/* calculate pairs for first 8 elements (cofactors) */
	tmp[0] = src[10] * src[15];
	tmp[1] = src[11] * src[14];
	tmp[2] = src[9] * src[15];
	tmp[3] = src[11] * src[13];
	tmp[4] = src[9] * src[14];
	tmp[5] = src[10] * src[13];
	tmp[6] = src[8] * src[15];
	tmp[7] = src[11] * src[12];
	tmp[8] = src[8] * src[14];
	tmp[9] = src[10] * src[12];
	tmp[10] = src[8] * src[13];
	tmp[11] = src[9] * src[12];

	/* calculate first 8 elements (cofactors) */
	dst[0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
	dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
	dst[1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
	dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
	dst[2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
	dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
	dst[3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
	dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
	dst[4] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
	dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
	dst[5] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
	dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
	dst[6] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
	dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
	dst[7] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
	dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

	/* calculate pairs for second 8 elements (cofactors) */
	tmp[0] = src[2]*src[7];
	tmp[1] = src[3]*src[6];
	tmp[2] = src[1]*src[7];
	tmp[3] = src[3]*src[5];
	tmp[4] = src[1]*src[6];
	tmp[5] = src[2]*src[5];
	tmp[6] = src[0]*src[7];
	tmp[7] = src[3]*src[4];
	tmp[8] = src[0]*src[6];
	tmp[9] = src[2]*src[4];
	tmp[10] = src[0]*src[5];
	tmp[11] = src[1]*src[4];

	/* calculate second 8 elements (cofactors) */
	dst[8] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
	dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
	dst[9] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
	dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
	dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
	dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
	dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
	dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
	dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
	dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
	dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
	dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
	dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
	dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
	dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
	dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

	/* calculate determinant */
	det = src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];
	if(det == 0.f)
		MNFatal("Trying to invert singular matrix.");

	/* calculate matrix inverse */
	MNMatrix4x4 mInv;
	det = 1/det;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			mInv.m[i][j] = dst[4*i + j] * det;

	return mInv;
}