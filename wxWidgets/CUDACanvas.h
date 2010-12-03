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
/// \file	wxWidgets\CUDACanvas.h
///
/// \brief	Declares the CUDACanvas class. 
/// \author	Mathias Neumann
/// \date	03.10.2010
/// \ingroup	UI
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_CUDACANVAS_H__
#define __MN_CUDACANVAS_H__

#pragma once

#include <vector_types.h>
#include "wx/wx.h"
#include "wx/glcanvas.h"

class MainFrame;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	CUDACanvas
///
/// \brief	An OpenGL canvas that uses CUDA to render the image.
///
///			This class contains the important OpenGL and CUDA initialization.
///
/// \author	Mathias Neumann
/// \date	03.10.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class CUDACanvas : public wxGLCanvas
{
public:
	/// Camera modes supported.
	enum CameraMode
	{
		/// Rotation around look-at position. No keyboard movement.
		CamMode_RotAroundLookAt,
		/// WASD keyboard movement and mouse rotation with fixed eye position.
		CamMode_WASD
	};

public:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \fn	CUDACanvas(MainFrame* pMainFrame, wxWindowID id = wxID_ANY,
    /// 	const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize)
    ///
    /// \brief	Constructs the CUDA canvas.
	///
	///			Does \em not initialize OpenGL and CUDA. This has to be done when the canvas is
	///			visible. See InitGLandCUDA().
    ///
    /// \author	Mathias Neumann
    /// \date	03.10.2010
    ///
    /// \param [in]		pMainFrame	Pointer to the main frame of MNRT. May not be \c NULL. 
    /// \param	id					The identifier of the canvas.
    /// \param	pos					Position of canvas. 
    /// \param	size				Size of canvas. 
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    CUDACanvas(MainFrame* pMainFrame, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize);
    virtual ~CUDACanvas();

// Attributes
private:
    bool   m_bInited;
	wxSize m_szScreen;
	MainFrame* m_pMainFrame;
	// Last mouse position.
	wxPoint m_nMouseLast;
	bool m_bMouseDragged;
	// Camera mode.
	CameraMode m_CamMode;

	// Video buffer object.
	GLuint m_glVBO;
	// Texture, used to enable filtering.
	GLuint m_glTexture;
	// CUDA VBO resource.
	struct cudaGraphicsResource *m_cudaVBORes;


	// FPS measurement timer.
	unsigned int m_TimerFPS;
	// Last "real" frame timer.
	unsigned int m_TimerLastFrame;
	// Last frames per second.
	float m_fLastFPS;

// Methods
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void InitGLandCUDA()
	///
	/// \brief	Initialises OpenGL and CUDA.
	///
	///			When you forget to call this, it is called within the \c OnPaint method.
	/// 		
	/// 		\warning	This has to be called when this canvas is visible! 
	///
	/// \author	Mathias Neumann
	/// \date	03.10.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void InitGLandCUDA();
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void DestroyGLandCUDA()
	///
	/// \brief	Destroys OpenGL and CUDA. 
	///
	/// \author	Mathias Neumann
	/// \date	03.10.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void DestroyGLandCUDA();

	/// Sets current camera mode.
	void SetCameraMode(CameraMode mode) { m_CamMode = mode; }
	/// Gets current camera mode.
	CameraMode GetCameraMode() const { return m_CamMode; }

	/// Returns the current FPS (frames per second).
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void GetTimings(float* fps, float* timeLastFrame_s)
	///
	/// \brief	Computes and returns timings. 
	///
	/// \author	Mathias Neumann
	/// \date	29.10.2010
	///
	/// \param [out]	fps				Will receive the current FPS. 
	/// \param [out]	timeLastFrame_s	Will receive the time spent for the last frame (seconds).
	///									Updated only for "real" frames, i.e. frames where the screen
	///									buffer was updated.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void GetTimings(float* fps, float* timeLastFrame_s);

	/// Returns whether we are initialized, hence whether CUDA is ready.
	bool IsCUDAInited() const { return m_bInited; }
	/// Returns current OpenGL screen buffer contents.
	void GetScreenBuffer(uchar4* data);
	/// Returns current image in given buffer.
	void GetCurrentImage(uchar4* data);
	/// Returns the size of the screen buffer.
	wxSize GetScreenSize() const { return m_szScreen; }

private:
	void OnPaint(wxPaintEvent& event);
    void OnEraseBackground(wxEraseEvent& event);
	void OnMouseEvent(wxMouseEvent& event);
	void OnKeyEvent(wxKeyEvent& event);

    void Render();
	void UpdateTimings();

	// Use this to turn off VSync under Windows. It has no effect on other OS. 
	void SetEnableVSync(bool bEnable);
    
DECLARE_EVENT_TABLE()
};

#endif // __MN_CUDACANVAS_H__