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

#include "MNCudaPrimitives.h"
#include "MNUtilities.h"
#include "MNCudaUtil.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
// MNCudaPrimitives implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
MNCudaPrimitives::MNCudaPrimitives(void)
{
	CreatePlans(1024*1024);
}

MNCudaPrimitives::~MNCudaPrimitives(void)
{
	DestoryPlans();
}

MNCudaPrimitives& MNCudaPrimitives::GetInstance()
{
	static MNCudaPrimitives cp;
	return cp;
}


void MNCudaPrimitives::CreatePlans(size_t maxElements)
{
	m_mapPrims.clear();

	// Create scan plan.
	CUDPPConfiguration configScan;
    configScan.op = CUDPP_ADD;
	configScan.datatype = CUDPP_UINT;
    configScan.algorithm = CUDPP_SCAN;
    configScan.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_EXCLUSIVE;
	CreatePlan(MNPrim_ScanAddE, configScan, maxElements);

	configScan.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_INCLUSIVE;
	CreatePlan(MNPrim_ScanAddI, configScan, maxElements);

	// Create compact plan.
	CUDPPConfiguration configCompact;
	configCompact.datatype = CUDPP_UINT;
    configCompact.algorithm = CUDPP_COMPACT;
	configCompact.options = CUDPP_OPTION_FORWARD;
    CreatePlan(MNPrim_Compact, configCompact, maxElements);

	// Create segmented scan plan (inclusive).
	CUDPPConfiguration configSegScan;
    configSegScan.op = CUDPP_ADD;
	configSegScan.datatype = CUDPP_UINT;
    configSegScan.algorithm = CUDPP_SEGMENTED_SCAN;
    configSegScan.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_INCLUSIVE;
    CreatePlan(MNPrim_SegScanAddI, configSegScan, maxElements);

	// Create segmented scan plan (exclusive).
    configSegScan.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_EXCLUSIVE;
    CreatePlan(MNPrim_SegScanAddE, configSegScan, maxElements);

	// Create key-value-pair sort plan.
	CUDPPConfiguration configSort;
	configSort.datatype = CUDPP_UINT;
	configSort.algorithm = CUDPP_SORT_RADIX;
	configSort.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    CreatePlan(MNPrim_SortKeyValue, configSort, maxElements);
}

void MNCudaPrimitives::CreatePlan(MNPrimitiveType type, const CUDPPConfiguration& config, 
										 size_t maxElemCount)
{
	// Check if plan is already created.
	if(m_mapPrims.find(type) != m_mapPrims.end())
	{	
		// Already in map. So recreate.
		CUDAPrimitive* pPrim = m_mapPrims[type];

		if(CUDPP_SUCCESS != cudppDestroyPlan(pPrim->plan))
			MNFatal("Failed to destroy CUDPP plan (op: %d).", type);
		if(CUDPP_SUCCESS != cudppPlan(&pPrim->plan, config, maxElemCount, 1, 0))
			MNFatal("Failed to create CUDPP plan (op: %d).", type);
	}
	else
	{
		CUDPPHandle plan;
		if(CUDPP_SUCCESS != cudppPlan(&plan, config, maxElemCount, 1, 0))
			MNFatal("Failed to create CUDPP plan (op: %d).", type);

		CUDAPrimitive* pPrim = new CUDAPrimitive(config, plan, maxElemCount);
		m_mapPrims[type] = pPrim;
	}
}

void MNCudaPrimitives::DestoryPlans()
{
	std::map<MNPrimitiveType, CUDAPrimitive*>::iterator iter;
	for(iter=m_mapPrims.begin(); iter!=m_mapPrims.end(); ++iter) 
	{
		CUDAPrimitive* pPrim = iter->second;
		if(CUDPP_SUCCESS != cudppDestroyPlan(pPrim->plan))
			MNFatal("Failed to destroy CUDPP plan (op: %d).", iter->first);

		SAFE_DELETE(pPrim);
    }

	m_mapPrims.clear();
}

bool MNCudaPrimitives::CheckPlanSize(MNPrimitiveType type, size_t requiredMaxElems)
{
	MNAssert(m_mapPrims.find(type) != m_mapPrims.end());

	// Check if the current plan is large enough.
	CUDAPrimitive* pPrim = m_mapPrims[type];
	if(pPrim->maxElemCount >= requiredMaxElems)
		return false;

	// Determine new size.
	size_t newMax = max(requiredMaxElems, 2*pPrim->maxElemCount);

	// Recreate...
	MNMessage("Recreating CUDPP plan (op: %d, old size: %d; new size: %d).\n", type, pPrim->maxElemCount, newMax); 
	CreatePlan(type, pPrim->config, newMax);
	return true;
}


void MNCudaPrimitives::Scan(const void* d_in, size_t numElems, bool bInclusive, void* d_out)
{
	MNAssert(numElems > 0 && d_in && d_out);

	if(bInclusive)
	{
		CheckPlanSize(MNPrim_ScanAddI, numElems);
		CUDAPrimitive* pPrim = m_mapPrims[MNPrim_ScanAddI];
		if(CUDPP_SUCCESS != cudppScan(pPrim->plan, d_out, d_in, numElems))
			MNFatal("Failed to run CUDPP inclusive scan.");
	}
	else
	{
		CheckPlanSize(MNPrim_ScanAddE, numElems);
		CUDAPrimitive* pPrim = m_mapPrims[MNPrim_ScanAddE];
		if(CUDPP_SUCCESS != cudppScan(pPrim->plan, d_out, d_in, numElems))
			MNFatal("Failed to run CUDPP exclusive scan.");
	}
}

void MNCudaPrimitives::Compact(const void* d_in, const unsigned* d_isValid, size_t numElems, 
							   void* d_outCompacted, size_t *d_outNewCount)
{
	MNAssert(numElems > 0 && d_in && d_isValid && d_outCompacted && d_outNewCount);
	CheckPlanSize(MNPrim_Compact, numElems);

	CUDAPrimitive* pPrim = m_mapPrims[MNPrim_Compact];
	if(CUDPP_SUCCESS != cudppCompact(pPrim->plan, d_outCompacted, 
				d_outNewCount, d_in, d_isValid, numElems))
		MNFatal("Failed to run CUDPP compact.");
}

void MNCudaPrimitives::SegmentedScan(const void* d_in, const uint* d_flags, size_t numElems,
									 bool bInclusive, void *d_out)
{
	MNAssert(numElems > 0 && d_in && d_flags && d_out);
	
	if(bInclusive)
	{
		CheckPlanSize(MNPrim_SegScanAddI, numElems);
		CUDAPrimitive* pPrim = m_mapPrims[MNPrim_SegScanAddI];
		if(CUDPP_SUCCESS != cudppSegmentedScan(pPrim->plan, d_out, 
					d_in, d_flags, numElems))
			MNFatal("Failed to run CUDPP segmented scan (I).");
	}
	else
	{
		CheckPlanSize(MNPrim_SegScanAddE, numElems);
		CUDAPrimitive* pPrim = m_mapPrims[MNPrim_SegScanAddE];
		if(CUDPP_SUCCESS != cudppSegmentedScan(pPrim->plan, d_out, 
					d_in, d_flags, numElems))
			MNFatal("Failed to run CUDPP segmented scan (E).");
	}
}

void MNCudaPrimitives::Sort(void* d_ioKeys, void* d_ioValues, uint keyValueMax, size_t numElems)
{
	MNAssert(numElems > 0 && d_ioKeys && d_ioValues && keyValueMax > 0);
	if(numElems > 32*1024*1024)
		MNWarning("Large CUDPP sorts might not work. This one has %d elements...", numElems);
	CheckPlanSize(MNPrim_SortKeyValue, numElems);

	// Compute number of least significant bits from keyValueMax.
	int keyBits = Log2Int(float(keyValueMax) + 1.f);

	CUDAPrimitive* pPrim = m_mapPrims[MNPrim_SortKeyValue];
	if(CUDPP_SUCCESS != cudppSort(pPrim->plan, d_ioKeys, d_ioValues, keyBits, numElems))
		MNFatal("Failed to run CUDPP sort.");
}