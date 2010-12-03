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
/// \file	MNRT\MNStatContainer.h
///
/// \brief	Declares the MNStatContainer class and several StatEntry classes.
/// \author	Mathias Neumann
/// \date	14.04.2010
/// \ingroup	cpuutil
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MN_STATCONTAINER_H__
#define __MN_STATCONTAINER_H__

#pragma once

#include <map>
#include <limits>

// Forward decl.
#ifndef DOXYGEN_IGNORE
class StatEntry;
enum cudaError;
typedef cudaError cudaError_t;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	MNStatContainer
///
/// \brief	Statistics container class for MNRT.
///
///			Inspired by \ref lit_pharr "[Pharr and Humphreys 2004]". This
///			container holds a set of StatEntry objects, ordered by both category name and
///			entry name. There is a simple way to print out the status of the container.
///
///			Class is designed as singleton and might need optimizations for when used from
///			multiple CPU-threads.
///
/// \author	Mathias Neumann
/// \date	14.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class MNStatContainer
{
	// Singleton. Hide constructors.
private:
	MNStatContainer(void);
	MNStatContainer(const MNStatContainer& other);
public:
	~MNStatContainer(void);

// Attributes
private:
	// Holds the stat counters.
	std::map<std::pair<std::string, std::string>, StatEntry*> m_Counters;
	// Whether timers are enabled.
	bool m_bTimersEnabled;

// Class
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static MNStatContainer& GetInstance()
	///
	/// \brief	Returns the only MNStatContainer instance.
	/// 		
	/// \warning Not thread-safe! 
	///
	/// \author	Mathias Neumann
	/// \date	19.03.2010
	///
	/// \return	The instance. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static MNStatContainer& GetInstance();

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void AddEntry(StatEntry* pEntry)
	///
	/// \brief	Adds a stat entry to the container.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param [in]	pEntry	The entry to add.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void AddEntry(StatEntry* pEntry);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	StatEntry* FindEntry(const std::string& strCategory, const std::string& strName)
	///
	/// \brief	Searches for a stat entry. 
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param	strCategory	Category of stat entry.
	/// \param	strName		Name of stat entry.
	///
	/// \return A pointer to the stat entry or \c NULL, if nothing found.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	StatEntry* FindEntry(const std::string& strCategory, const std::string& strName);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Reset()
	///
	/// \brief	Resets all stat entries using StatEntry::Reset().
	///
	/// \author	Mathias Neumann
	/// \date	02.11.2010
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Reset();

	/// Sets whether timers should take times. Can be used to avoid timing overhead.
	void SetTimersEnabled(bool b);
	/// Gets whether timers are enabled.
	bool GetTimersEnabled() const { return m_bTimersEnabled; }

// Reporting
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Print(FILE* fileTarget)
	///
	/// \brief	Prints all statistics to given file.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param [in]	fileTarget	The target file.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Print(FILE* fileTarget);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	StatEntry
///
/// \brief	Abstract entry base class for MNStatContainer entries.
///
///			This class manages the basic properties of an entry, that is the stat category and it's
///			name. Subclasses should provide factory methods that allow creation of stat entries
///			\e and register the entry with the MNStatContainer instance.
///
/// \author	Mathias Neumann
/// \date	14.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class StatEntry
{
protected:
	/// Hidden constructor. Use factory methods.
	StatEntry(const std::string& strCategory, const std::string& strName);
	/// Hidden constructor. Use factory methods.
	StatEntry(const StatEntry& other);
public:
	~StatEntry();

// Attributes
private:
	// Meta data
	std::string m_strCategory, m_strName;

// Accessors
public:
	/// Returns entry's category name.
	const std::string& GetCategory() { return m_strCategory; }
	/// Returns entry's name.
	const std::string& GetName() { return m_strName; }

// Operations
public:
	/// Resets this entry to it's initial state.
	virtual void Reset() = 0;
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	virtual void Print(FILE* fileTarget) = 0
	///
	/// \brief	Prints this entry to the given file.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param [in]	fileTarget	The target file.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void Print(FILE* fileTarget) = 0;

// Implementation
protected:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void PrintHeading(FILE* fileTarget)
	///
	/// \brief	Prints entry heading to given file.
	///
	///			The entry heading is basically the entry's name in a special format to ensure
	///			alignment and improve visual quality. Subclasses should call this method within
	///			their Print() implementation.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param [in]	fileTarget	The target file.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void PrintHeading(FILE* fileTarget);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void PrintHeading(FILE* fileTarget, const std::string strExtra)
	///
	/// \brief	Prints entry heading to given file, including an extra string.
	///
	///			The extra string is appended to the default entry heading. It can be used within
	///			subclasses to print out multiple, distinguishable stat lines.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param [in]	fileTarget	The target file.
	/// \param	strExtra		The extra string. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void PrintHeading(FILE* fileTarget, const std::string strExtra);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	StatCounter
///
/// \brief	Counter entry for MNStatContainer. 
///
/// \author	Mathias Neumann
/// \date	14.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class StatCounter : public StatEntry
{
protected:
	/// Hidden constructor. Use factory methods.
	StatCounter(const std::string& strCategory, const std::string& strName);
	/// Hidden constructor. Use factory methods.
	StatCounter(const StatCounter& other);
public:
	~StatCounter();

// Attributes
private:
	// Counter
	long long m_nCounter;

// Factory
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static StatCounter& Create(const std::string& strCategory, const std::string& strName)
	///
	/// \brief	Creates a StatCounter object and registers it with the MNStatContainer instance.
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param	strCategory	Entry's category name.
	/// \param	strName		Entry's name.
	///
	/// \return	The created counter. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static StatCounter& Create(const std::string& strCategory, const std::string& strName);

// Accessors
public:
	/// Returns the current counter state.
	long long GetCounter() const { return m_nCounter; }
	/// Operator to get the current counter state.
	operator long long() { return m_nCounter; }
	/// Operator to get the current counter state as \c double.
	operator double() { return (double)m_nCounter; }

// Operations
public:
	/// Increments the counter by the given value \a val.
	void Increment(long long val = 1) { m_nCounter += val; }
	/// Ensures the counter's value is smaller or equal \a val.
	void Min(long long val) { if(val < m_nCounter) m_nCounter = val; }
	/// Ensures the counter's value is larger or equal \a val.
	void Max(long long val) { if(val > m_nCounter) m_nCounter = val; }
	/// Operator to add a given value \a val to the counter.
	void operator+=(long long val) { m_nCounter += val; }
	/// Increment operator. Increments counter by one.
	void operator++() { ++m_nCounter; }

	virtual void Reset() { m_nCounter = 0; }
	virtual void Print(FILE* fileTarget);
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	StatRatio
///
/// \brief	Ratio entry for MNStatContainer. 
///
///			This entry has a basic counter and a step counter. The ratio is defined as the ratio of
///			basic counter to step counter. If the step counter is zero, the ratio is \c FLT_MAX.
///
/// \author	Mathias Neumann
/// \date	14.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class StatRatio : public StatEntry
{
protected:
	/// Hidden constructor. Use factory methods.
	StatRatio(const std::string& strCategory, const std::string& strName);
	/// Hidden constructor. Use factory methods.
	StatRatio(const StatRatio& other);
public:
	~StatRatio();

// Attributes
private:
	// Counter
	size_t m_nCounter;
	// Steps
	size_t m_nSteps;

// Factory
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static StatRatio& Create(const std::string& strCategory, const std::string& strName)
	///
	/// \brief	Creates a StatRatio object and registers it with the MNStatContainer instance. 
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param	strCategory	Entry's category name. 
	/// \param	strName		Entry's name. 
	///
	/// \return	The created ratio entry. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static StatRatio& Create(const std::string& strCategory, const std::string& strName);

// Accessors
public:
	/// Returns the current ratio of base counter to step counter.
	float GetRatio() const 
	{ 
		if(m_nSteps)
			return float(m_nCounter) / float(m_nSteps); 
		else
			return FLT_MAX;
	}
	/// Operator to get the current ratio.
	operator float() { return GetRatio(); }

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	void Increment(size_t count = 1, size_t steps = 0)
	///
	/// \brief	Increments one or both counters. 
	///
	/// \author	Mathias Neumann
	/// \date	14.04.2010
	///
	/// \param	count	Increment for base counter.
	/// \param	steps	Increment for step counter.
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void Increment(size_t count = 1, size_t steps = 0) { m_nCounter += count; m_nSteps += steps; }

	virtual void Reset() { m_nCounter = 0; m_nSteps = 0; }
	virtual void Print(FILE* fileTarget);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \class	StatTimer
///
/// \brief	Timer entry for MNStatContainer.  
///
///			Implemented using CUDA GPU timers. Provides means to measure timings for GPU
///			execution by allowing synchronization before starting and stopping the timer.
///
/// \author	Mathias Neumann
/// \date	15.04.2010
////////////////////////////////////////////////////////////////////////////////////////////////////
class StatTimer : public StatEntry
{
protected:
	/// Hidden constructor. Use factory methods.
	StatTimer(const std::string& strCategory, const std::string& strName, bool hasSteps = false);
	/// Hidden constructor. Use factory methods.
	StatTimer(const StatTimer& other);
public:
	~StatTimer();

// Attributes
private:
	// CUDA timer identifier.
	unsigned int m_Timer;
	// Whether we have steps and can display an average.
	bool m_hasSteps;

// Factory
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	static StatTimer& Create(const std::string& strCategory, const std::string& strName,
	/// 	bool hasSteps = false)
	///
	/// \brief	Creates a StatTimer object and registers it with the MNStatContainer instance. 
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	///
	/// \param	strCategory	Entry's category name. 
	/// \param	strName		Entry's name. 
	/// \param	hasSteps	Whether the timer has steps and can display an average.
	///
	/// \return	The created timer entry. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static StatTimer& Create(const std::string& strCategory, const std::string& strName, bool hasSteps = false);

// Accessors
public:
	/// Returns total time in milliseconds.
	float GetTotal() const;
	/// Returns average time in milliseconds. Only valid for timer entries with steps.
	float GetAverage() const;

// Operations
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Start(bool cudaSynchronize = false)
	///
	/// \brief	Starts the timer. 
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	///
	/// \param	cudaSynchronize	Pass \c true to synchronize CUDA threads before starting the timer. 
	///
	/// \return	Error code returned from \c cudaThreadSynchronize(), or \c cudaSuccess. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Start(bool cudaSynchronize = false);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \fn	cudaError_t Stop(bool cudaSynchronize = false)
	///
	/// \brief	Stops the timer. 
	///
	/// \author	Mathias Neumann
	/// \date	15.04.2010
	///
	/// \param	cudaSynchronize	Pass \c true to synchronize CUDA threads before stopping the timer. 
	///
	/// \return	Error code returned from \c cudaThreadSynchronize(), or \c cudaSuccess. 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaError_t Stop(bool cudaSynchronize = false);

	virtual void Reset();
	virtual void Print(FILE* fileTarget);
};



#endif // __MN_STATCONTAINER_H__