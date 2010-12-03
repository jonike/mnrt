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
/// \file	Geometry\MNMatrix4x4.h
///
/// \brief	Declares the MNMatrix4x4 class. 
/// \author	Mathias Neumann
/// \date	07.02.2010
/// \ingroup	Geometry
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_MATRIX4X4_H__
#define __MN_MATRIX4X4_H__

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNMatrix4x4
///
/// \brief	4x4 floating point matrix representation.
/// 		
/// 		Access entries of the matrix using the public #m array.
///
/// \author	Mathias Neumann
/// \date	07.02.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNMatrix4x4
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNMatrix4x4(void)
	///
	/// \brief	Default constructor. Creates 4x4 identity matrix.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNMatrix4x4(void);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNMatrix4x4(float mat[4][4])
	///
	/// \brief	Constructor that initializes matrix with given values from float array.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	mat	Starting values of matrix entries.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNMatrix4x4(float mat[4][4]);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNMatrix4x4(float m00, float m01, float m02, float m03, float m10, float m11, float m12,
	/// 	float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32,
	/// 	float m33)
	///
	/// \brief	Constructor that initializes the matrix from given parameters. \a m13 for example
	/// 		describes the value in the first row and third column. 
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \param	m00	Value for entry (0, 0). 
	/// \param	m01	Value for entry (0, 1). 
	/// \param	m02	Value for entry (0, 2). 
	/// \param	m03	Value for entry (0, 3). 
	/// \param	m10	Value for entry (1, 0). 
	/// \param	m11	Value for entry (1, 1). 
	/// \param	m12	Value for entry (1, 2). 
	/// \param	m13	Value for entry (1, 3). 
	/// \param	m20	Value for entry (2, 0). 
	/// \param	m21	Value for entry (2, 1). 
	/// \param	m22	Value for entry (2, 2). 
	/// \param	m23	Value for entry (2, 3). 
	/// \param	m30	Value for entry (3, 0). 
	/// \param	m31	Value for entry (3, 1). 
	/// \param	m32	Value for entry (3, 2). 
	/// \param	m33	Value for entry (3, 3). 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNMatrix4x4(float m00, float m01, float m02, float m03,
				float m10, float m11, float m12, float m13,
				float m20, float m21, float m22, float m23,
				float m30, float m31, float m32, float m33);
	~MNMatrix4x4(void) {}

// Data members
public:
	/// Entries of the matrix. Public for convenience.
	float m[4][4];

// Operators
public:
	/// Matix addition operator.
	MNMatrix4x4 operator+(const MNMatrix4x4& mat) const;
	/// Matrix addition assignment operator.
	MNMatrix4x4& operator+=(const MNMatrix4x4& mat);

	/// Matrix subtraction operator.
	MNMatrix4x4 operator-(const MNMatrix4x4& mat) const;
	/// Matrix subtraction assignment operator.
	MNMatrix4x4& operator-=(const MNMatrix4x4& mat);

	/// Matrix scaling operator.
	MNMatrix4x4 operator*(float f) const;
	/// Matrix scaling assignment operator.
	MNMatrix4x4& operator*=(float f);

	/// Matrix multiplication operator.
	MNMatrix4x4 operator*(const MNMatrix4x4& mat) const;
	/// Matrix multiplication assignment operator.
	MNMatrix4x4& operator*=(const MNMatrix4x4& mat);

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNMatrix4x4 Transpose() const
	///
	/// \brief	Transposes this matrix and returns the result.
	///
	///	\warning This is no inplace operation.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \return	The transposed matrix. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Transpose
	MNMatrix4x4 Transpose() const;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	MNMatrix4x4 Inverse() const
	///
	/// \brief	Computes the inverse of this matrix. 
	///
	///			Implemented using Cramer's rule. Check
	///			ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf. Assumes that the matrix
	///			is invertible.
	///
	///	\warning This is no inplace operation.
	///
	/// \author	Mathias Neumann
	/// \date	07.02.2010
	///
	/// \return	The inverse. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	MNMatrix4x4 Inverse() const;
};

#endif // __MN_MATRIX4X4_H__