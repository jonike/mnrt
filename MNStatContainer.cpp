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

#include "MNStatContainer.h"
#include "MNCudaUtil.h"
#include <string>
#include <ctime>
#include <cutil_inline.h>

using namespace std;

typedef pair<string, string> StatKeyType;
typedef map<pair<string, string>, StatEntry*> StatMap;

////////////////////////////////////////////////////////////////////////////////////////////////////
// MNStatContainer implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
MNStatContainer::MNStatContainer(void)
{
	m_bTimersEnabled = false;
}

MNStatContainer::~MNStatContainer(void)
{
	StatMap::iterator iter;
	for(iter=m_Counters.begin(); iter!=m_Counters.end(); ++iter)
	{
		StatEntry* pCtr = iter->second;
		SAFE_DELETE(pCtr);
	}
}

MNStatContainer& MNStatContainer::GetInstance()
{
	static MNStatContainer sc;
	return sc;
}

void MNStatContainer::AddEntry(StatEntry* pEntry)
{
	MNAssert(pEntry);
	if(!pEntry)
		return;

	// Construct pair.
	StatKeyType key = std::make_pair(pEntry->GetCategory(), pEntry->GetName());

	// Check if contained.
	StatMap::iterator iter = m_Counters.find(key);
	if(iter != m_Counters.end())
		MNFatal("Duplicate stat entry detected.");

	// Add new counter.
	m_Counters[key] = pEntry;
}

StatEntry* MNStatContainer::FindEntry(const std::string& strCategory, const std::string& strName)
{
	// Construct pair.
	StatKeyType key = std::make_pair(strCategory, strName);

	StatMap::iterator iter = m_Counters.find(key);

	if(iter != m_Counters.end())
		return iter->second;
	else
		return NULL;
}

void MNStatContainer::Reset()
{
	StatMap::iterator iter;
	for(iter=m_Counters.begin(); iter!=m_Counters.end(); ++iter)
	{
		StatEntry* pCtr = iter->second;
		pCtr->Reset();
	}
}

void MNStatContainer::SetTimersEnabled(bool b)
{
	m_bTimersEnabled = b;
}

void MNStatContainer::Print(FILE* fileTarget)
{
	// Get current time.
	time_t now = time(NULL);
	char strTime[64];
	ctime_s(strTime, 64, &now);

	fprintf(fileTarget, "------------------------------------------------------------\n");
	fprintf(fileTarget, "Statistics %s", strTime);
	
	// Print counters.
	StatMap::iterator iter;
	string curCat;
	for(iter=m_Counters.begin(); iter!=m_Counters.end(); ++iter)
	{
		StatEntry* pCtr = iter->second;

		// Check if category changed.
		if(curCat != pCtr->GetCategory())
		{
			// Print category...
			curCat = pCtr->GetCategory();
			fprintf(fileTarget, "%s\n", curCat.c_str());
		}

		pCtr->Print(fileTarget);
	}

	fprintf(fileTarget, "------------------------------------------------------------\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// StatEntry implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
StatEntry::StatEntry(const std::string& strCategory, const std::string& strName)
{
	m_strCategory = strCategory;
	m_strName = strName;
}

StatEntry::~StatEntry()
{
}

void StatEntry::PrintHeading(FILE* fileTarget)
{
	PrintHeading(fileTarget, "");
}

void StatEntry::PrintHeading(FILE* fileTarget, const std::string strExtra)
{
	MNAssert(fileTarget);
	const int valColumn = 45;

	std::string strHeading = "---> " + GetName() + strExtra;
	fprintf(fileTarget, "%s:", strHeading.c_str());

	// Offset value to get clear layout.
	int offset = valColumn - (int)strHeading.length();
	while(offset-- > 0)
		fputc(' ', fileTarget);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// StatCounter implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
StatCounter::StatCounter(const std::string& strCategory, const std::string& strName)
	: StatEntry(strCategory, strName)
{
	m_nCounter = 0;
}

StatCounter::~StatCounter()
{
}

/*static*/ StatCounter& StatCounter::Create(const std::string& strCategory, const std::string& strName)
{
	// Check if allready created.
	MNStatContainer& cont = MNStatContainer::GetInstance();
	StatCounter* pCounter = (StatCounter*)cont.FindEntry(strCategory, strName);
	if(!pCounter)
	{
		pCounter = new StatCounter(strCategory, strName);
		cont.AddEntry(pCounter);
	}

	return *pCounter;
}

/*virtual*/ void StatCounter::Print(FILE* fileTarget)
{
	PrintHeading(fileTarget);

	// Now write value.
	if(m_nCounter > 1e6)
		fprintf(fileTarget, "%10.3lf M", (double)m_nCounter / 1e6);
	else if(m_nCounter > 1e3)
		fprintf(fileTarget, "%10.3lf K", (double)m_nCounter / 1e3);
	else
		fprintf(fileTarget, "%6.0lf", (double)m_nCounter);

	fprintf(fileTarget, "\n");
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// StatRatio implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
StatRatio::StatRatio(const std::string& strCategory, const std::string& strName)
	: StatEntry(strCategory, strName)
{
	m_nCounter = 0;
	m_nSteps = 0;
}

StatRatio::~StatRatio()
{
}

/*static*/ StatRatio& StatRatio::Create(const std::string& strCategory, const std::string& strName)
{
	// Check if allready created.
	MNStatContainer& cont = MNStatContainer::GetInstance();
	StatRatio* pCounter = (StatRatio*)cont.FindEntry(strCategory, strName);
	if(!pCounter)
	{
		pCounter = new StatRatio(strCategory, strName);
		cont.AddEntry(pCounter);
	}

	return *pCounter;
}

/*virtual*/ void StatRatio::Print(FILE* fileTarget)
{
	PrintHeading(fileTarget);

	// Now write value.
	if(m_nSteps == 0)
		fprintf(fileTarget, "   N/A");
	else
	{
		float ratio = GetRatio();
		if(ratio > 1e6f)
			fprintf(fileTarget, "%10.3f M", ratio / 1e6f);
		else if(ratio > 1e3f)
			fprintf(fileTarget, "%10.3f K", ratio / 1e3f);
		else
			fprintf(fileTarget, "%10.3f", ratio);
	}
	fprintf(fileTarget, "\n");
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// StatTimer implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
StatTimer::StatTimer(const std::string& strCategory, const std::string& strName, bool hasSteps/* = false*/)
	: StatEntry(strCategory, strName)
{
	mncudaCheckErrorCUtil(cutCreateTimer(&m_Timer));
	m_hasSteps = hasSteps;
}

StatTimer::~StatTimer()
{
	mncudaCheckErrorCUtil(cutDeleteTimer(m_Timer));
}

/*static*/ StatTimer& StatTimer::Create(const std::string& strCategory, const std::string& strName, bool hasSteps/* = false*/)
{
	// Check if allready created.
	MNStatContainer& cont = MNStatContainer::GetInstance();
	StatTimer* pCounter = (StatTimer*)cont.FindEntry(strCategory, strName);
	if(!pCounter)
	{
		pCounter = new StatTimer(strCategory, strName, hasSteps);
		cont.AddEntry(pCounter);
	}

	return *pCounter;
}

float StatTimer::GetTotal() const
{
	return cutGetTimerValue(m_Timer);
}

float StatTimer::GetAverage() const
{
	return cutGetAverageTimerValue(m_Timer);
}

cudaError_t StatTimer::Start(bool cudaSynchronize/* = false*/)
{
	cudaError_t err = cudaSuccess;

	MNStatContainer& cont = MNStatContainer::GetInstance();
	if(cont.GetTimersEnabled())
	{
		if(cudaSynchronize)
			err = cudaThreadSynchronize();
		mncudaCheckErrorCUtil(cutStartTimer(m_Timer));
	}

	return err;
}

cudaError_t StatTimer::Stop(bool cudaSynchronize/* = false*/)
{
	cudaError_t err = cudaSuccess;

	MNStatContainer& cont = MNStatContainer::GetInstance();
	if(cont.GetTimersEnabled())
	{
		if(cudaSynchronize)
			err = cudaThreadSynchronize();
		mncudaCheckErrorCUtil(cutStopTimer(m_Timer));
	}

	return err;
}

/*virtual*/ void StatTimer::Reset()
{
	mncudaCheckErrorCUtil(cutResetTimer(m_Timer));
}

/*virtual*/ void StatTimer::Print(FILE* fileTarget)
{
	PrintHeading(fileTarget);

	// Nothing to print if timers disabled.
	MNStatContainer& cont = MNStatContainer::GetInstance();
	if(!cont.GetTimersEnabled())
	{
		fprintf(fileTarget, "    Disabled\n");
		return;
	}

	// Write total.
	float total_ms = GetTotal();
	if(total_ms > 1e3f)
		fprintf(fileTarget, "%10.3f s", total_ms / 1e3f);
	else
		fprintf(fileTarget, "%10.3f ms", total_ms);
	fprintf(fileTarget, "\n");

	// Write average if available.
	if(m_hasSteps)
	{
		float avg_ms = GetAverage();

		PrintHeading(fileTarget, " (avg.)");
		if(avg_ms > 1e3f)
			fprintf(fileTarget, "%10.3f s", avg_ms / 1e3f);
		else
			fprintf(fileTarget, "%10.3f ms", avg_ms);
		fprintf(fileTarget, "\n");

		PrintHeading(fileTarget, " (freq.)");
		if(avg_ms <= 0)
			fprintf(fileTarget, "     N/A");
		else
		{
			float freq = 1e3f / avg_ms;
			if(freq > 1e6f)
				fprintf(fileTarget, "%10.3f M/s", freq / 1e6f);
			else if(freq > 1e3f)
				fprintf(fileTarget, "%10.3f K/s", freq / 1e3f);
			else
				fprintf(fileTarget, "%10.3f 1/s", freq);
		}
		fprintf(fileTarget, "\n");
	}
}