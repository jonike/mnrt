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
/// \file	MNRT\RTCore.h
///
/// \brief	Declares the RTCore class. 
/// \author	Mathias Neumann
/// \date	30.01.2010
/// \ingroup	core
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \mainpage	MNRT Source Code Documentation
/// 
/// \section	sec_intro		Introduction
///
///	The application MNRT implements several techniques to realize fast global illumination for 
///	dynamic scenes on Graphics Processing Units (GPUs). It was developed during the creation of 
/// my Diplomarbeit (German master thesis equivalent):
///
///  -	<b>Neumann, Mathias <br />
///		"GPU-basierte globale Beleuchtung mit CUDA in Echtzeit" <br />
///		Diplomarbeit, FernUniversit&auml;t in Hagen, 2010</b>
///
/// The basic ideas of the implementation are described within \ref lit_wang "[Wang et al. 2009]". 
/// Right now MNRT is very experimental. Therefore it might contain several errors. Furthermore
/// MNRT does not show all features of the system described in my thesis. For example, spherical
/// harmonics are not used to handle glossy materials, yet. They might be added in the future.
///
/// I released MNRT (including source code) to the public as I believe that I might not be able to
/// concentrate on improving the application in the next time. Hence I want to allow others to check
/// what I've done so far. Maybe someone has some ideas on improving MNRT. I'd be glad to hear about
/// them.
///
/// Currently neither an English nor a German version of my thesis are available online. Please
/// check the <a href="http://www.maneumann.com/MNRT/Help/MNRT_Documentation.html">documentation of 
/// MNRT</a> for some background on MNRT. It's available in English.
///
///
/// \section	sec_overview	Overview
///
///	MNRT is not a fully functional application for end users. This section shows features and problems
/// of the current version of MNRT, specifically the source code of MNRT.
///
/// \par Features
///
/// -	Almost everything (except e.g. scene and image loading) is implemented on the GPU using CUDA.
/// -	GPU-based kd-tree implementation, following \ref lit_zhou "[Zhou et al. 2008]". Currently both
///		triangle and point kd-trees are supported (see KDTreeGPU).
///	-	GPU-based ray tracing implementation, including kd-tree traversal algorithm for intersection search
///		(see raytracing.cu).
/// -	GPU-based photon Mapping including final gathering to compute indirect illumination. This includes
///		algorithms for photon tracing and density estimation. 
/// -	Iterative, histogram-based kNN-search implementation, following \ref lit_zhou "[Zhou et al. 2008]".
/// -	Accelerated final gathering step by adaptively selecting samples and subsequent interpolation
///		(for indirect illumination).
/// -	Accelerated density estimation using an illumination cut through the kd-tree of the photon map.
///		Gathering k nearest photons is replaced by a spatial interpolation using RBF.
/// -	CUDA implementations of parallel reduction and parallel segmented reduction (see MNCudaUtil.h).
/// -	CUDA API overhead elimination using custom GPU memory pool which allocates big chunks of memory
///		and hands segments of memory out to requesters (see MNCudaMemPool).
/// -	Random number generation using NVIDIA's Mersenne Twister SDK sample.
///
/// \par Limitations
///
///	-	MNRT does not yet perform at frame rates required for computer games. Even the rates
///		\ref lit_wang "[Wang et al. 2009]" reported cannot be reached.
///	-	Currently only diffuse materials (BRDFs) are allowed. I already added some support for specular
///		BSDFs (reflection, transmission), but it is currently disabled.
/// -	Missing support for spherical harmonics. Using irradiance values instead of radiance fields.
///		Furthermore there is no support to read glossy materials from scene descriptions, yet.
///	-	Input scenes may only consist of triangles.
/// -	Dynamic geometry, lights and materials are not implemented. Camera position however can 
///		be varied. The former would require an animation system. To simulate dynamic geometry,
///		there is an option to rebuild kd-trees for objects and photon maps every frame.
/// -	MNRT does not benefit from multiple GPUs, yet.
/// -	Right now, the CPU cores are left unemployed and all work is done on the GPU. Moving some
///		work to CPU cores might be a possible way to increase performance.
///
/// \par Source Code Quality
///
/// -	As MNRT is my first GPU-based application, it might not be the state-of-the-art of GPU-based 
///		programming.
///	-	In no case does MNRT represent a high quality software product of good stability and 
///		extensibility.
/// -	Seeming limitations of the CUDA API reduced the coherency of some components,
///		e.g. limited scope of CUDA texture references (see raytracing.cu).
/// -	Due to limited optimization, one might reach significant improvement for some parts
///		of MNRT.
///
///
/// \section	sec_usage		Usage Hints
///
///	I used Microsoft Visual Studio for the development of MNRT. Hence MNRT is currently only available
/// for Windows. But as I used almost no Windows specific code, compiling MNRT for other operating
/// systems should be possible, assumed that the target system has support for CUDA. Please check
/// the \ref page_usagehints page for more details.
/// 
///
/// \section	sec_license		Copyright and License
///
/// MNRT is copyrighted by Mathias Neumann and released under a 3-clause BSD license. Check the
/// \ref page_license page for details. Thus reuse of my work in other projects is encouraged. 
/// I'd be happy if you'd <a href="http://www.maneumann.com">tell me</a> of such reuse. To cite my
/// work, you can use my thesis reference, see section \ref sec_intro.
///
///
/// \section	sec_literature	Literature
///
/// <dl>
///     <dt>[Jensen 2001]</dt><dd></a> \anchor lit_jensen
///         Jensen, H. W.<br />
///         <em>Realistic Image Synthesis Using Photon Mapping</em><br />
///         A K Peters, 2001
///     </dd>
///     <dt>[Pharr and Humphreys 2004]</dt><dd></a> \anchor lit_pharr
///         Pharr, M. and Humphreys, G.<br />
///         <em>Physically Based Rendering: From Theory to Implementation</em><br />
///         Morgan Kaufmann Publishers Inc., 2004
///     </dd>
///     <dt>[Veach 1997]</dt><dd></a> \anchor lit_veach
///         Veach, E.<br />
///         <em>Robust Monte Carlo Methods for Light Transport Simulation</em><br />
///         PhD thesis, Stanford University, 1997
///     </dd>
///     <dt>[Wang et al. 2009]</dt><dd> \anchor lit_wang
///         Wang, R.; Zhou, K.; Pan, M. & Bao, H.<br />
///         <em>An efficient GPU-based approach for interactive global illumination</em><br />
///         SIGGRAPH '09: ACM SIGGRAPH 2009 papers, ACM, 2009, 1-8
///     </dd>
///     <dt>[Ward et al. 1988]</dt><dd></a> \anchor lit_ward
///         Ward, G. J.; Rubinstein, F. M. & Clear, R. D.<br />
///         <em>A Ray Tracing Solution for Diffuse Interreflection</em><br />
///         SIGGRAPH '88: Proceedings of the 15th annual conference on Computer graphics and interactive techniques, ACM, 1988, 85-92
///     </dd>
///     <dt>[Zhou et al. 2008]</dt><dd></a> \anchor lit_zhou
///         Zhou, K.; Hou, Q.; Wang, R. & Guo, B.<br />
///         <em>Real-time KD-tree construction on graphics hardware</em><br />
///         SIGGRAPH Asia '08: ACM SIGGRAPH Asia 2008 papers, ACM, 2008, 1-11
///     </dd>
/// </dl>
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \page		page_usagehints	Usage Hints
///
///	I used Microsoft Visual Studio 2008 for the development of MNRT. Hence MNRT is currently only
/// available for Windows. But as I used almost no Windows specific code, compiling MNRT for other
/// operating systems should be possible, assumed that the target system has support for CUDA. All used
/// third party libraries are platform independent, e.g. the wxWidgets library for the user interface.
///	 
/// \par Target Platform
///		In the early phases of development, I compiled MNRT for the \b x64 platform, too. For
///		convenience I dropped x64 and sticked with \b Win32, mainly to avoid additional problems
///		(e.g. with libraries). I haven't tested compiling for x64 for some time.
///
/// \par Preprocessor definitions
///		Most of the preprocessor definitions currently used result from including the GUI library
///		wxWidgets, specifically the Windows version of it. The wiki on wxWidgets, the wxWiki, lists all
///		<a href="http://wiki.wxwidgets.org/Microsoft_Visual_C%2B%2B_Guide">required definitions for 
///		wxWidgets</a>.
///
/// \par Runtime Library
///		I compile MNRT using the Multi-Threaded (<code>/MT</code>) runtime library option, see "C/C++ ->
///		Code Generation" properties. For debug mode of course, this should be Multi-threaded Debug 
///		(<code>/MTd</code>). The same has to be used for the correspoonding options in the CUDA build
///		rule and all used libraries. Ignoring this will lead to linker problems.
///
/// \par Required Libraries
///		To compile MNRT, you'd have to compile several third party libraries. I decided not to include
///		compiled versions of these libraries, primarily to save space. The following list shows all
///		used libraries. It contains links to sites where you can download each library. Furthermore
///		I added the library versions I used for compilation.\n\n
///		Please note that all these libraries have to be compiled using the Multi-Threaded (<code>/MT</code>) 
///		runtime library option for release and Multi-threaded Debug (<code>/MTd</code>) for debug mode. See
///		"C/C++ -> Code Generation" property page.
///		\n\n
///		 - <a href="http://www.nvidia.com/object/tesla_software.html">CUDA and CUDA SDK</a> (used version 3.1)
///		 - OpenGL Utility Toolkit and Extension Wrangler Library (used version included in CUDA SDK)
///		 - <a href="http://assimp.sourceforge.net/">Open Asset Import Libary</a> (used version 1.1)
///		 - <a href="http://code.google.com/p/cudpp/">CUDA Data Parallel Primitives Library</a> (used version 1.1.1)
///		 - <a href="http://openil.sourceforge.net/">Developer's Image Library</a> (used version 1.7.8)
///		 - <a href="http://www.wxwidgets.org/">wxWidgets</a> GUI library (used version 2.8.11), see the <a href="http://wiki.wxwidgets.org/Microsoft_Visual_C%2B%2B_Guide">wxWiki page</a> for Visual Studio configuration
///		 - <a href="http://wxpropgrid.sourceforge.net/cgi-bin/index">wxPropertyGrid</a> control library  (used version 1.4.13)\n
///
/// \par Assumed Environment Variables
///		Currently the properties of MNRT assume the existance of some environment variables that describe
///		the paths to some libraries. You have the option to use them or to replace them with concrete
///		paths.
///		\n\n
///     <table width="60%" class="doxtable" align="center">
///			<tr><th>Env. Variable</th><th>Description</th></tr>
///         <tr><td><code>ASSIMP_PATH</code></td><td>Path to ASSIMP SDK root directory.</td></tr>
///         <tr><td><code>DEV_IL_PATH</code></td><td>Path to DevIL SDK root directory.</td></tr>
///         <tr><td><code>CUDA_INC_PATH</code></td><td>CUDA library include path.</td></tr>
///         <tr><td><code>CUDA_LIB_PATH32</code></td><td>CUDA library path for 32-bit libraries. I added this as I installed the x64 version of the CUDA toolkit.</td></tr>
///         <tr><td><code>NVSDKCOMPUTE_ROOT</code></td><td>CUDA SDK root directory path.</td></tr>
///         <tr><td><code>WXWIN</code></td><td>wxWidgets root directory path.</td></tr>
///		</table>
///
/// \par Additional Dependencies
///		The bulk of dependencies are required for the GUI components created with wxWidgets. Again,
///		I refer to the wxWiki for the <a href="http://wiki.wxwidgets.org/Microsoft_Visual_C%2B%2B_Guide">
///		dependancies required for wxWidgets</a>. Debug mode versions of some the libraries will be
///		available by including an additional "d" or "D", as shown by <b style="color: #44B;">[d]</b>
///		or <b style="color: #44B;">[D]</b>.
///		\n\n
///		- <b>cuda.lib</b>: CUDA library.
///		- <b>cudart.lib</b>: CUDA Runtime library.
///		- <b>cutil32<b style="color: #44B;">[D]</b>.lib</b>: CUDA SDK utility library.
///		- <b>glut32.lib</b>: <a href="http://www.opengl.org/resources/libraries/glut/">OpenGL Utility Toolkit</a>. For CUDA-OpenGL-interoperability.
///		- <b>glew32.lib</b>: <a href="http://glew.sourceforge.net/credits.html">OpenGL Extension Wrangler Library</a>. For CUDA-OpenGL-interoperability.
///		- <b>assimp.lib</b>: <a href="http://assimp.sourceforge.net/">Open Asset Import Libary</a>.
///		- <b>cudpp32<b style="color: #44B;">[d]</b>.lib</b>: <a href="http://code.google.com/p/cudpp/">CUDA Data Parallel Primitives Library</a>.
///		- <b>DevIL.lib</b>: <a href="http://openil.sourceforge.net/">Developer's Image Library</a>.
///		- <b>wxmsw28<b style="color: #44B;">[d]</b>_core.lib</b>: Required for wxWidgets.
///		- <b>wxbase28<b style="color: #44B;">[d]</b>.lib</b>: Required for wxWidgets.
///		- <b>wxmsw28<b style="color: #44B;">[d]</b>_html.lib</b>: HTML control support (wxWidgets).
///		- <b>wxmsw28<b style="color: #44B;">[d]</b>_gl.lib</b>: OpenGL canvas support (wxWidgets).
///		- <b>wxmsw28<b style="color: #44B;">[d]</b>_adv.lib</b>: Advanced wxWidgets components.
///		- <b>comctl32.lib</b>: Required for wxWidgets.
///		- <b>rpcrt4.lib</b>: Required for wxWidgets.
///		- <b>winmm.lib</b>: Required for wxWidgets.
///		- <b>advapi32.lib</b>: Required for wxWidgets.
///		- <b>wsock32.lib</b>: Required for wxWidgets.
///		- <b>wxcode_msw28<b style="color: #44B;">[d]</b>_propgrid.lib</b>: <a href="http://wxpropgrid.sourceforge.net/cgi-bin/index">wxPropertyGrid</a> control library.
///
/// \par printf Commands
///		As I switched to the wxWidgets GUI, the valuable command line window seemed to be lost. I
///		was not able find any simple way of passing the \c stdout to some wxWidgets component. However,
///		Fermi GPUs allow \c printf inside kernels, a very useful feature. To retain the command line
///		window, I switched to the Console (<code>/SUBSYSTEM:CONSOLE</code>) sub system (see Linker options).
///		Furthermore I had to replace the <code>IMPLEMENT_APP()</code> macro with a <code>IMPLEMENT_APP_CONSOLE()</code>
///		macro to get a valid <code>main()</code>. Else there there was the linker error
///		\code LIBCMT.lib(crt0.obj) : error LNK2001: unresolved external symbol _main \endcode
///		Of course, for a version without command line window, the former macro has to be used in
///		conjunction with Windows sub system linker option (<code>/SUBSYSTEM:WINDOWS</code>).
///
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \page		page_license	Copyright and License
///
/// MNRT is copyrighted by Mathias Neumann and released under the subsequent BSD license. Thus reuse
/// of my work in other projects is encouraged. I'd be happy if you'd
/// <a href="http://www.maneumann.com">tell me</a> of such reuse. To cite my work, you can use my 
/// thesis reference, see section \ref sec_intro.
///
///	In addition to this license there are the licenses of all used software libraries, see section
/// \ref sec_libraries.
///
/// <div style="margin-left: 25%; width: 50%; padding: 1em;" class="memproto">
///		MNRT License
/// </div>
/// <div style="margin-left: 25%; width: 50%; padding: 1em;" class="memdoc">
///     <strong>Copyright &copy; 2010 Mathias Neumann, www.maneumann.com</strong>.<br />
///		<strong>All rights reserved</strong>.<br />
///		<br />
///		Redistribution and use in source and binary forms, with or without modification, are 
///	    permitted provided that the following conditions are met:<br />
///    <ul>
///        <li>Redistributions of source code must retain the above copyright notice, this list of conditions 
///            and the following disclaimer.</li>
///        <li>Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
///            and the following disclaimer in the documentation and/or other materials provided with the 
///            distribution.</li>
///        <li>Neither the name Mathias Neumann, nor the names of contributors may be 
///            used to endorse or promote products derived from this software without specific prior written 
///            permission.</li>
///    </ul>
///    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
///    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
///    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
///    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
///    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
///    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
///    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
///    WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/// </div>
///
/// \section	sec_libraries	Used Software Libraries
///
/// <center>
///     <table width="60%" class="doxtable">
///			<tr><th>Library</th><th>License</th><th>Utilization</th></tr>
///         <tr>
///             <td><a href="http://assimp.sourceforge.net/">Open Asset Import Libary</a><br />
///                 (ASSIMP)</td>
///             <td>
///                 <a href="http://assimp.sourceforge.net/main_license.html">BSD license</a>
///             </td>
///             <td>
///                 Scene loading.
///             </td>
///         </tr>
///         <tr>
///             <td><a href="http://code.google.com/p/cudpp/">CUDA Data Parallel Primitives Library</a><br />
///                 (CUDPP)</td>
///             <td>
///                 <a href="http://www.gpgpu.org/static/developer/cudpp/rel/cudpp_1.1/html/license.html">BSD license</a>
///             </td>
///             <td>
///                 GPU-based implementation of parallel primitives Scan, Segmented Scan, Compact and Sort.
///             </td>
///         </tr>
///         <tr>
///             <td><a href="http://openil.sourceforge.net/">Developer's Image Library</a><br />
///                 (DevIL)</td>
///             <td>
///                 <a href="http://openil.sourceforge.net/license.php">LGPL License</a>
///             </td>
///             <td>
///                 Loading of texture images.
///             </td>
///         </tr>
///         <tr>
///             <td><a href="http://www.wxwidgets.org/">wxWidgets</a><br />
///             </td>
///             <td>
///                 <a href="http://www.wxwidgets.org/about/newlicen.htm">wxWindows Licence</a>
///             </td>
///             <td>
///                 Realization of GUI in a plattform independent way. Additionally, the
///                 <a href="http://wxpropgrid.sourceforge.net/cgi-bin/index">wxPropertyGrid</a> 
///                 control for wxWidgets was used. It is licensed under the wxWindows License, too.
///             </td>
///         </tr>
///         <tr>
///             <td><a href="http://glew.sourceforge.net/credits.html">OpenGL Extension Wrangler Library</a>
///             </td>
///             <td>
///                 <a href="http://glew.sourceforge.net/glew.txt">BSD license</a>
///             </td>
///             <td>
///                 Display of results.
///             </td>
///         </tr>
///     </table>
/// </center>
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \page		page_changelog	Change Log
///
/// \par	MNRT Version 1.0
///		- Initial release.
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup	core	Application Core
/// 
/// \brief	Core components of MNRT.
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __MN_RTCORE_H__
#define __MN_RTCORE_H__

#pragma once

#include <vector_types.h>
#include "KernelDefs.h"

class RayChunk;
class RayPool;
class KDTreeTri;
class KDTreePoint;
class PhotonMap;
class ProgressListener;
class MNRTSettings;
class SceneConfig;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	RTCore
///
/// \brief	Core class of MNRT.
///
///			Manages and coordinates all components of the global illumination algorithm used in
///			MNRT. 
///
/// \todo	Currently the class is somewhat overloaded. Improve the coherency by extracting algorithms
///			like photon map construction or compaction methods into other classes.
///
/// \author	Mathias Neumann
/// \date	30.01.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class RTCore
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	RTCore(MNRTSettings* pSettings)
	///
	/// \brief	Constructor. Pass MNRT settings object.
	///
	/// \author	Mathias Neumann
	/// \date	30.01.2010
	///
	/// \param [in]	pSettings	The settings to use.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	RTCore(MNRTSettings* pSettings);
	/// Destructor. Calls Destroy().
	~RTCore(void);

private:
	bool m_bInited;

	// Ray pool which holds the rays to process in host memory.
	RayPool* m_pRayPool;
	// Shading point structures.
	ShadingPoints m_spShade;
	ShadingPoints m_spClusters;
	ShadingPoints m_spFinalGather;
	// Cluster list. Used to store clusters of last frame.
	ClusterList m_clusterList;

	// Scene configuration.
	SceneConfig* m_pSC;
	// Triangle data.
	TriangleData m_Tris;
	// Material data.
	MaterialData m_Materials;
	// kd-tree acceleration structure for triangles. Used to speed up intersection
	// search when ray tracing.
	KDTreeTri* m_pKDTree;

	// Global photon map. Represented as kd-tree for points.
	// Stores
	// - Direct photons (at first intersection after emitting).
	// - Indirect photons (photons that scattered nonspecular right before the hit).
	PhotonMap* m_pPMGlobal;
	// Caustics photon map. Represented as kd-tree for points.
	// Stores
	// - Caustics photons (photons that followed a path of specular scattering events).
	PhotonMap* m_pPMCaustics;

	// Settings
	MNRTSettings* m_pSettings;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool Initialize(uint screenSize, SceneConfig* pSceneConfig)
	///
	/// \brief	Initializes this object for given parameters.
	/// 		
	/// 		This method does not build object kd-tree for triangle intersection search and photon
	/// 		maps for photon mapping. This is done on demand, e.g. for each new frame to support
	/// 		dynamic scenes in theory. In practice MNRT cannot handle dynamic geometry yet. The
	/// 		support for animations is missing within the corresponding data structures, e.g.
	/// 		BasicScene or TriangleData. 
	///
	/// \author	Mathias Neumann
	/// \date	30.01.2010
	///
	/// \param	screenSize				Size of the screen in pixels (width = height). Has to be
	/// 								power of two. 
	/// \param [in]		pSceneConfig	Describes the scene to load. 
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool Initialize(uint screenSize, SceneConfig* pSceneConfig);
	/// Destroys data structures and releases device memory.
	void Destroy();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool RenderScene(uchar4* d_screenBuffer)
	///
	/// \brief	Renders the scene to the given buffer.
	/// 		
	/// 		Object kd-tree and photon map construction are included. 
	///
	/// \author	Mathias Neumann
	/// \date	February 2010
	///
	/// \param [in,out]	d_screenBuffer	Target buffer. Has to contain the same number of pixels
	///									as the current camera object (see SceneConfig).
	///
	/// \return	\c true if it succeeds, \c false if it fails. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool RenderScene(uchar4* d_screenBuffer);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	bool BenchmarkKDTreeCon(uint numWarmup, uint numRuns, float* outTotal_s, float* outAvg_s,
	/// 	ProgressListener* pListener = NULL)
	///
	/// \brief	Runs a benchmark for the kd-tree construction.
	///
	///			This method is merely an aid to improve the kd-tree construction algorithm but a
	///			perfect benachmark suite for CUDA performance of GPUs. Right now it only tests
	///			the triangle kd-tree construction for the scene currently loaded.
	///
	/// \author	Mathias Neumann
	/// \date	21.10.2010
	///
	/// \param	numWarmup			Number of warmup constructions. Excluded from time measurement.
	/// \param	numRuns				Number of runs constructions.
	/// \param [out]	outTotal_s	Total time in seconds.
	/// \param [out]	outAvg_s	Average time (per construction) in seconds.
	/// \param [in]	pListener		Progress listener. Might be \c NULL.
	///
	/// \return	\c true if benchmark was not aborted, else \c false.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool BenchmarkKDTreeCon(uint numWarmup, uint numRuns, float* outTotal_s, float* outAvg_s,
		ProgressListener* pListener = NULL);

private:
	// Compacts rays *and* shading points. Returns number of rays traced.
	uint TraceRays(RayChunk* pRayChunk, ShadingPoints* pSP, const std::string& strCategory, 
		uint* d_outSrcAddr = NULL);

	// Rebuilds object kd-tree. Returns true if successful.
	bool RebuildObjectKDTree();

	// Rebuilds photons maps if required. Returns true if successful.
	bool RebuildPhotonMaps();
	void SpawnLightPhotons(PhotonData& outPhotons, uint photonOffset, uint numToSpawn);
	// Rebuilds photon map kd-trees. Requires photon lists. Returns true if successful.
	bool BuildPhotonMapKDTrees();

	uint CompactPhotonTrace(PhotonData& photonsTrace, int* d_triHitIndices, 
						float2* d_hitBaryCoords, float4* d_hitDiffClrs, float4* d_hitSpecClrs, 
						uint* d_isValid);
	uint CompactClusters(ClusterList& clusters, uint* d_isValid);

	bool RenderToBuffer(float4* d_outLuminance);

	bool DoFinalGathering(const ShadingPoints& spHits, float4* d_ioRadiance);
	void FullFinalGathering(const ShadingPoints& shadingPts, 
		uint numFGSamplesX, uint numFGSamplesY,
		float4* d_outIrradiance, float* d_outMeanReciDists = NULL);
	bool SelectiveFinalGathering(const ShadingPoints& spHits, 
		const ShadingPoints& spClusters, float4* d_outIrradiance);

// Wang algorithms
private:
	void UpdateIrradianceSamples(const ShadingPoints& spHits);
	void SelectIrradianceSamples(const ShadingPoints& spHits, uint numSamplesToSelect, 
		ClusterList& outClusters, PairList& outCluster2SP, ShadingPoints& outSPClusters, 
		float4* d_outScreenBuffer);
	void PerformAdaptiveSeeding(const ShadingPoints& spHits, uint numSamplesToSelect, 
		ClusterList& outClusters, float4* d_outScreenBuffer);
	void ClassifyShadingPoints(const ShadingPoints& spHits, const ClusterList& clusters, 
		PairList& outCluster2SP, float* d_outGeoVarsSorted);
	void PerformKMeansClustering(const ShadingPoints& spHits, ClusterList& ioClusters,
		PairList& outCluster2SP);
};


#endif //__MN_RTCORE_H__