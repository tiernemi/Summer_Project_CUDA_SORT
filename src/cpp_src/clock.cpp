/*
 * =====================================================================================
 *
 *       Filename:  clock.cpp
 *
 *    Description:  Clock used for timing functions.
 *
 *        Version:  1.0
 *        Created:  09/06/16 12:00:23
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Custom Headers //
#include "../../inc/clock.hpp"

/* 
 * ===  MEMBER FUNCTION : CLOCK  =======================================================
 *         Name:  startClock
 *  Description:  Starts clock based on system time.
 * =====================================================================================
 */

void Clock::start() {
	startTime = std::chrono::system_clock::now() ;
}

/* 
 * ===  MEMBER FUNCTION : CLOCK  =======================================================
 *         Name:  stopClock
 *  Description:  Stops clock.
 * =====================================================================================
 */

void Clock::stop() {
	endTime = std::chrono::system_clock::now() ;
}

/* 
 * ===  MEMBER FUNCTION : CLOCK  =======================================================
 *         Name:  getDuration
 *      Returns:  Elapsed time.
 *  Description:  Gets the elapsed time from the stop to start of clock.
 * =====================================================================================
 */

float Clock::getDuration() {
	std::chrono::duration<float> elapsed_seconds = endTime - startTime ;
	return elapsed_seconds.count() ;
}


