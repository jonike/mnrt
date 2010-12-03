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

#include "MNBBox.h"


// Union of AABB and point.
MNBBox Union(const MNBBox& box, const MNPoint3& p)
{
	MNBBox ret = box;
	ret.ptMin.x = std::min(box.ptMin.x, p.x);
	ret.ptMin.y = std::min(box.ptMin.y, p.y);
	ret.ptMin.z = std::min(box.ptMin.z, p.z);
	ret.ptMax.x = std::max(box.ptMax.x, p.x);
	ret.ptMax.y = std::max(box.ptMax.y, p.y);
	ret.ptMax.z = std::max(box.ptMax.z, p.z);
	return ret;
}

// Union of two AABBs.
MNBBox Union(const MNBBox& box1, const MNBBox& box2)
{
	MNBBox ret = box1;
	ret.ptMin.x = std::min(box1.ptMin.x, box2.ptMin.x);
	ret.ptMin.y = std::min(box1.ptMin.y, box2.ptMin.y);
	ret.ptMin.z = std::min(box1.ptMin.z, box2.ptMin.z);
	ret.ptMax.x = std::max(box1.ptMax.x, box2.ptMax.x);
	ret.ptMax.y = std::max(box1.ptMax.y, box2.ptMax.y);
	ret.ptMax.z = std::max(box1.ptMax.z, box2.ptMax.z);
	return ret;
}