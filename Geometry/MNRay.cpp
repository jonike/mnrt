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

#include "MNRay.h"
#include "../MNUtilities.h"

MNRay::MNRay(const MNPoint3& origin, const MNVector3& direction)
{
	MNAssert(direction.Length() != 0);

	o = origin;
    d = Normalize(direction);
}

MNRay::~MNRay(void)
{
}

bool MNRay::IsOnRay(MNPoint3 pt)
{
	float lambda = 0;
    if(d.x != 0)
    {
        float temp = (pt.x - o.x) /  d.x;
        
        // Allow a small difference due to computation errors.
        if(temp < 0 || (lambda != 0 && abs(temp - lambda) > 1E-12))
            return false;
        else
            lambda = temp;
    }
    if(d.y != 0)
    {
        float temp = (pt.y - o.y) /  d.y;
        
        // Allow a small difference due to computation errors.
        if(temp < 0 || (lambda != 0 && abs(temp - lambda) > 1E-12))
            return false;
        else
            lambda = temp;
    }
    if(d.z != 0)
    {
        float temp = (pt.z - o.z) /  d.z;
        
        // Allow a small difference due to computation errors.
        if(temp < 0 || (lambda != 0 && abs(temp - lambda) > 1E-12))
            return false;
        else
            lambda = temp;
    }
    
    return lambda > 0;
}

MNPoint3 MNRay::GetPoint(float lambda)
{
	return o + lambda*d;
}