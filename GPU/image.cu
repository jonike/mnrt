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
/// \file	GPU\image.cu
///
/// \brief	Contains image conversion kernels.
/// \author	Mathias Neumann
/// \date	21.08.2010
/// \ingroup	globalillum
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mncudautil_dev.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_ConvertToRGBA8(float4* d_inRadiance, uint numPixel,
/// 	uchar4* d_outScreenBuffer)
///
/// \brief	Converts radiance values (\c float4) to RGBA8 format.
///
///			Resulting buffer can be displayed using OpenGL.
///
/// \author	Mathias Neumann
/// \date	27.06.2010
///
/// \param [in]		d_inRadiance		Radiance pixel buffer (R, G, B, unused).
/// \param	numPixel					Number of pixels. 
/// \param [out]	d_outScreenBuffer	Conversion target pixel buffer (RGBA8 format).
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_ConvertToRGBA8(float4* d_inRadiance, uint numPixel, 
									  uchar4* d_outScreenBuffer)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numPixel)
	{
		float4 L = d_inRadiance[tid];

		// Write converted data. Ensure coalesced access by writing uchar4 in one step.
		uchar4 pix;
		pix.x = (uchar)fminf(255.f, 255.f * L.x);
		pix.y = (uchar)fminf(255.f, 255.f * L.y);
		pix.z = (uchar)fminf(255.f, 255.f * L.z);
		d_outScreenBuffer[tid] = pix;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_TestDiscrepancy(uint numSamples, uint screenW, uint screenH,
/// 	uchar4* d_outScreenBuffer)
///
/// \brief	Simple test of halton sequence generation.
/// 		
/// 		More precisely, the output of ::dev_RadicalInverse() is tested. This is done by
/// 		generating \a numSamples samples of a 2D halton sequence and plotting them on the
/// 		given screen buffer. Due to possible write conflicts, the result might not be exact. 
///
/// \author	Mathias Neumann
/// \date	08.07.2010
///
/// \param	numSamples					Number of sequence members to generate. 
/// \param	screenW						Screen width in pixels. 
/// \param	screenH						Screen height in pixels. 
/// \param [in,out]	d_outScreenBuffer	Screen buffer (accumulator). 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_TestDiscrepancy(uint numSamples,
									   uint screenW, uint screenH, uchar4* d_outScreenBuffer)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numSamples)
	{
		float rnd1 = dev_RadicalInverse(tid+1, 5);
		float rnd2 = dev_RadicalInverse(tid+1, 7);

		uint myX = rnd1*screenW;
		uint myY = rnd2*screenH;
		uint myPixel = screenW*myY + myX;

		uchar4 pix = d_outScreenBuffer[myPixel];
		pix.x = min(255, pix.x + 32);
		pix.y = min(255, pix.y + 32);
		pix.z = min(255, pix.z + 32);
		// Warning: Write conflicts possible!
		d_outScreenBuffer[myPixel] = pix;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	__global__ void kernel_GenerateErrorImage(uchar4* d_ioImage, uchar4* d_inReference,
/// 	uint numPixels, float fScale)
///
/// \brief	Generates an error image from current image and reference image.
/// 		
/// 		The relative error is calculated and its absolute value is displayed, where a
/// 		relative error of zero leads to black pixel value. Error scaling is possible by
/// 		providing an error scale factor. 
///
/// \author	Mathias Neumann
/// \date	21.08.2010
///
/// \param [in,out]	d_ioImage		The buffer containing the current image. Will be updated to
///									contain the error image.
/// \param [in]		d_inReference	Reference image pixel buffer of same size as \a d_ioImage. 
/// \param	numPixels				Number of pixels in both buffers. 
/// \param	fScale					The error scale factor. Will be multiplied with computed
/// 								pixel values (per channel) to amplify the resulting color. 
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_GenerateErrorImage(uchar4* d_ioImage, uchar4* d_inReference, 
										  uint numPixels, float fScale)
{
	uint idxPixel = blockIdx.x * blockDim.x + threadIdx.x;

	if(idxPixel < numPixels)
	{
		// Read pixel values.
		uchar4 clr = d_ioImage[idxPixel];
		uchar4 clrRef = d_inReference[idxPixel];
		float4 fclr = make_float4(float(clr.x) / 255.f, float(clr.y) / 255.f, 
								  float(clr.z) / 255.f, float(clr.w) / 255.f);
		float4 fclrRef = make_float4(float(clrRef.x) / 255.f, float(clrRef.y) / 255.f, 
								     float(clrRef.z) / 255.f, float(clrRef.w) / 255.f);
		float4 absErr = fclr - fclrRef;
		float4 relErr = absErr / fclrRef;

		// Write error.
		uchar4 err;
		err.x = (uchar)fminf(255.f, fmaxf(0.f, fScale * 255.f * fabsf(absErr.x)));
		err.y = (uchar)fminf(255.f, fmaxf(0.f, fScale * 255.f * fabsf(absErr.y)));
		err.z = (uchar)fminf(255.f, fmaxf(0.f, fScale * 255.f * fabsf(absErr.z)));
		err.w = (uchar)fminf(255.f, fmaxf(0.f, fScale * 255.f * fabsf(absErr.w)));
		d_ioImage[idxPixel] = err;
	}
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
//@{

/// Wraps kernel_ConvertToRGBA8() kernel call.
extern "C"
void KernelIMGConvertToRGBA8(float4* d_inRadiance, uint numPixel, 
							 uchar4* d_outScreenBuffer)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numPixel, blockSize.x), 1, 1);

	kernel_ConvertToRGBA8<<<gridSize, blockSize>>>(d_inRadiance, numPixel, d_outScreenBuffer);
	MNCUDA_CHECKERROR;
}

/// Calls kernel_TestDiscrepancy() to test discrepancy of ::dev_RadicalInverse().
extern "C" 
void KernelIMGTestDiscrepancy(uint numSamples, uint screenW, uint screenH, uchar4* d_outScreenBuffer)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numSamples, blockSize.x), 1, 1);

	mncudaSafeCallNoSync(cudaMemset(d_outScreenBuffer, 0, screenW*screenH*sizeof(uchar4)));

	kernel_TestDiscrepancy<<<gridSize, blockSize>>>(numSamples, screenW, screenH, d_outScreenBuffer);
	MNCUDA_CHECKERROR;
}

/// Wraps kernel_GenerateErrorImage() kernel call.
extern "C"
void KernelIMGGenerateErrorImage(uchar4* d_ioImage, uchar4* d_inReference, uint numPixels, float fScale)
{
	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(MNCUDA_DIVUP(numPixels, blockSize.x), 1, 1);

	kernel_GenerateErrorImage<<<gridSize, blockSize>>>(d_ioImage, d_inReference, numPixels, fScale);
	MNCUDA_CHECKERROR;
}

//@}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////