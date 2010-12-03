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

#include <GL/glew.h>	// Has to be included before "wx/glcanvas.h".
#include "CUDACanvas.h"
#include "MainFrame.h"
#include "../RTCore.h"
#include "../MNCudaMemPool.h"
#include "../CameraModel.h"
#include "../SceneConfig.h"
#include "../MNStatContainer.h"
#include "../MNCudaPrimitives.h"
#include <cutil_inline.h>
#include <cuda_gl_interop.h>


// Device context attribute list for wxGLCanvas
// "Note that all the attributes need to be set in the attribList. Just setting the 
//  interesting values of GL_SAMPLE_BUFFERS do not work causing glewInit() to fail."
// See: http://www.comp.nus.edu.sg/~ashwinna/docs/wxWidgetsInstallation.html
int f_devAttribs[] = {WX_GL_RGBA, 
					  WX_GL_DOUBLEBUFFER, 
					  WX_GL_DEPTH_SIZE, 0,
					  0, 0};

BEGIN_EVENT_TABLE(CUDACanvas, wxGLCanvas)
    EVT_PAINT(CUDACanvas::OnPaint)
    EVT_ERASE_BACKGROUND(CUDACanvas::OnEraseBackground)
	EVT_MOUSE_EVENTS(CUDACanvas::OnMouseEvent)
	EVT_CHAR(CUDACanvas::OnKeyEvent)
END_EVENT_TABLE()


CUDACanvas::CUDACanvas(MainFrame* pMainFrame, wxWindowID id, const wxPoint& pos, const wxSize& size)
    : wxGLCanvas(pMainFrame, (wxGLCanvas*)NULL, id, pos, size, 
				wxFULL_REPAINT_ON_RESIZE, _T("CUDA GLCanvas"), f_devAttribs)
{
	m_bInited = false;
	m_pMainFrame = pMainFrame;
	m_szScreen = size;
	m_CamMode = CamMode_WASD;
	m_bMouseDragged = false;

	m_glVBO = 0;
	m_glTexture = 0;

	m_TimerFPS = 0;
	m_TimerLastFrame = 0;
	m_fLastFPS = 0.f;

	SetBackgroundColour(wxColor("black"));
}

CUDACanvas::~CUDACanvas(void)
{
	DestroyGLandCUDA();
}

void CUDACanvas::OnPaint(wxPaintEvent& event)
{
	wxPaintDC dc(this);

	if(!m_bInited)
		InitGLandCUDA();

	Render();
}

void CUDACanvas::OnEraseBackground(wxEraseEvent& event)
{
	// Do nothing when initialized, to avoid flickering.
}

void CUDACanvas::InitGLandCUDA()
{
	if(m_bInited)
		return;
	if (!GetContext()) 
		MNFatal("Failed to get OpenGL context!");
    SetCurrent(); // Requires the window to be visible!

	// Required for VBO stuff.
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0"))
		MNFatal("Required OpenGL extensions GL_VERSION_2_0 missing.");

    // Disable VSync. Works only on Windows.
	SetEnableVSync(false);

	// Note that we do not call cudaSetDevice() since we call cudaGLSetGLDevice.
	int nDev = m_pMainFrame->GetCUDADeviceID();
	mncudaSafeCallNoSync(cudaGLSetGLDevice(nDev));

	// Initialize memory pool.
	MNCudaMemPool& pool = MNCudaMemPool::GetInstance();
	pool.Initialize(256*1024*1024, 256*1024);

	mncudaCheckErrorCUtil(cutCreateTimer(&m_TimerFPS));
	mncudaCheckErrorCUtil(cutCreateTimer(&m_TimerLastFrame));

	// Create video buffer object.
	// NOTE: For CUDA toolkit 2.3 I used a pixel buffer object here. This is no more required for toolkit 3.0.
    glGenBuffers(1, &m_glVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
	glBufferData(GL_ARRAY_BUFFER, m_szScreen.x*m_szScreen.y*sizeof(GLubyte)*4, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register this buffer object with CUDA
	mncudaSafeCallNoSync(cudaGraphicsGLRegisterBuffer(&m_cudaVBORes, m_glVBO, cudaGraphicsMapFlagsNone));

	// Create texture for filtering purposes.
    glGenTextures(1, &m_glTexture);
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_szScreen.x, m_szScreen.y, 
		0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);


	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


	MNMessage("OpenGL and CUDA initialized. Using device %d.", nDev);

	m_bInited = true;
}

void CUDACanvas::DestroyGLandCUDA()
{
	if(!m_bInited)
		return;

	mncudaCheckErrorCUtil(cutDeleteTimer(m_TimerFPS));
	mncudaCheckErrorCUtil(cutDeleteTimer(m_TimerLastFrame));

	mncudaSafeCallNoSync(cudaGraphicsUnregisterResource(m_cudaVBORes));

	glDeleteBuffers(1, &m_glVBO);
	glDeleteTextures(1, &m_glTexture);
	m_glVBO = 0;
	m_glTexture = 0;

	// Kill primitive plans just before CUDA exit to avoid errors.
	MNCudaPrimitives::GetInstance().DestoryPlans();

	mncudaSafeCallNoSync(cudaThreadExit());

	m_bInited = false;
}

void CUDACanvas::Render()
{
	bool coreWantsUpdate = m_pMainFrame->WantUpdate();

	mncudaCheckErrorCUtil(cutStartTimer(m_TimerFPS));

	// Update last frame timer only if core wants an update.
	if(coreWantsUpdate)
	{
		mncudaCheckErrorCUtil(cutResetTimer(m_TimerLastFrame));
		mncudaCheckErrorCUtil(cutStartTimer(m_TimerLastFrame));
	}

	if (!GetContext()) 
		MNFatal("Failed to get OpenGL context!");

    wxPaintDC dc(this);
    SetCurrent();

	glClear(GL_COLOR_BUFFER_BIT); 

	if(!coreWantsUpdate)
		glBindTexture(GL_TEXTURE_2D, m_glTexture);
	else
	{
		// Map VBO to get CUDA device pointer.
		size_t num_bytes;
		uchar4* d_buffer;
		mncudaSafeCallNoSync(cudaGraphicsMapResources(1, &m_cudaVBORes, 0));
		mncudaSafeCallNoSync(cudaGraphicsResourceGetMappedPointer((void **)&d_buffer, 
			&num_bytes, m_cudaVBORes));

		m_pMainFrame->Render(d_buffer);
		mncudaSafeCallNoSync(cudaGetLastError());

		// This performs synchronization for the given stream.
		mncudaSafeCallNoSync(cudaGraphicsUnmapResources(1, &m_cudaVBORes, 0));

		// Load texture from VBO.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glVBO);
		glBindTexture(GL_TEXTURE_2D, m_glTexture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_szScreen.x, m_szScreen.y, 
			GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	// Draw texture to screen.
	glEnable(GL_TEXTURE_2D);
    
	// Define a quad that covers the whole screen.
	glBegin(GL_QUADS);
    glTexCoord2f(0, 0);	glVertex2f(0, 0);
    glTexCoord2f(1, 0);	glVertex2f(1, 0);
    glTexCoord2f(1, 1);	glVertex2f(1, 1);
    glTexCoord2f(0, 1);	glVertex2f(0, 1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

	glFlush();
    SwapBuffers();

	mncudaCheckErrorCUtil(cutStopTimer(m_TimerFPS));
	if(coreWantsUpdate)
		mncudaCheckErrorCUtil(cutStopTimer(m_TimerLastFrame));
}

void CUDACanvas::GetScreenBuffer(uchar4* data)
{
	if(!GetContext()) 
		MNFatal("Failed to get OpenGL context!");

	glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, m_szScreen.x*m_szScreen.y*sizeof(uchar4), data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void CUDACanvas::GetCurrentImage(uchar4* data)
{
	if(!GetContext()) 
		MNFatal("Failed to get OpenGL context!");

	glBindTexture(GL_TEXTURE_2D, m_glTexture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void CUDACanvas::GetTimings(float* fps, float* timeLastFrame_s)
{
	MNAssert(fps && timeLastFrame_s);

	if(!m_bInited)
	{
		*timeLastFrame_s = FLT_MAX;
		*fps = FLT_MAX;
	}
	else
	{
		*timeLastFrame_s = cutGetTimerValue(m_TimerLastFrame) / 1000.f;

		float avgTimer_s = cutGetAverageTimerValue(m_TimerFPS) / 1000.f;
		if(avgTimer_s == 0.f)
			*fps = FLT_MAX;
		else
		{
			*fps = 1.f / avgTimer_s;
			mncudaCheckErrorCUtil(cutResetTimer(m_TimerFPS));
		}
	}

	m_fLastFPS = *fps;
}

void CUDACanvas::OnMouseEvent(wxMouseEvent& event)
{
	if(!m_pMainFrame->WantUpdate())
		return;

	if(event.LeftDown())
	{
		// Start mouse movement.
		m_nMouseLast = event.GetPosition();
		m_bMouseDragged = true;
		event.Skip(); // This ensures the window gets the focus.
	}
	else if(event.LeftUp())
	{
		m_bMouseDragged = false;
	}
	else if(event.Dragging() && event.LeftIsDown() && m_bMouseDragged)
	{
		CameraModel* pCam = m_pMainFrame->GetCamera();
		if(!pCam)
			return;

		wxPoint newPos = event.GetPosition();

		if(m_CamMode == CamMode_RotAroundLookAt)
			pCam->RotateAroundAt(
				0.01f*(newPos.x-m_nMouseLast.x), 0.01f*(newPos.y-m_nMouseLast.y));
		else
			pCam->RotateAroundFixedEye(
				0.01f*(newPos.x-m_nMouseLast.x), 0.01f*(newPos.y-m_nMouseLast.y));

		m_nMouseLast = newPos;
		Refresh();
	}
}

void CUDACanvas::OnKeyEvent(wxKeyEvent& event)
{
	CameraModel* pCam = m_pMainFrame->GetCamera();
	if(!pCam)
		return;

	if(event.GetEventType() == wxEVT_CHAR)
	{
		if(event.GetKeyCode() == 99) // c = print camera position
		{
			MNMessage("Camera - Eye:    %8.3f, %8.3f, %8.3f", pCam->GetEye().x, pCam->GetEye().y, pCam->GetEye().z);
			MNMessage("Camera - LookAt: %8.3f, %8.3f, %8.3f", pCam->GetLookAt().x, pCam->GetLookAt().y, pCam->GetLookAt().z);
			MNMessage("Camera - Up:     %8.3f, %8.3f, %8.3f", pCam->GetUp().x, pCam->GetUp().y, pCam->GetUp().z);
		}
		else if(m_CamMode == CamMode_WASD && m_pMainFrame->WantUpdate())
		{
			// Adjust translation factor corresponding to scene extent.
			SceneConfig* pSC = m_pMainFrame->GetSceneConfig();
			MNBBox bounds = pSC->GetSceneBounds();
			MNVector3 vDiagonal = bounds.ptMax - bounds.ptMin;
			float fSize = vDiagonal.Length();
			float fTransFactor = 0.001f * fSize;

			// Scale by FPS if available. Restrict FPS since we only get
			// some k key events per second.
			if(m_fLastFPS > 0.f)
				fTransFactor *= 25.f / std::min(10.f, m_fLastFPS);

			if(pCam->ProcessKey(event.GetKeyCode(), fTransFactor))
				Refresh();
		}
	}
}


#ifdef _WIN32
	// See http://www.devmaster.net/forums/showthread.php?t=443
	typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);
	PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;

	void CUDACanvas::SetEnableVSync(bool bEnable)
	{
		const char *extensions = (const char*)glGetString( GL_EXTENSIONS );

		if(strstr( extensions, "WGL_EXT_swap_control" ) == 0)
		{
			MNWarning("Disabling vertical synchronization not supported.");
			return;
		}
		else
		{
			wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress( "wglSwapIntervalEXT" );

			int swapInterval = (bEnable ? 1 : 0);
			if(wglSwapIntervalEXT)
				wglSwapIntervalEXT(swapInterval);
		}
	}
#else // _WIN32
	void CUDACanvas::SetEnableVSync(bool bEnable)
	{
		MNWarning("Disabling vertical synchronization not supported.");
	}
#endif // _WIN32