#ifndef CLOCK_HPP_YNTIKD6P
#define CLOCK_HPP_YNTIKD6P

/*
 * =====================================================================================
 *
 *       Filename:  clock.hpp
 *
 *    Description:  Clock object used for timing.
 *
 *        Version:  1.0
 *        Created:  09/06/16 11:58:07
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <chrono>

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Clock
 *       Fields:  std::chrono::time_point<std::chrono::system_clock> start - Start time.
 *                std::chrono::time_point<std::chrono::system_clock> end - End time.
 *  Description:  Clock object used for recording execution times.
 * =====================================================================================
 */

class Clock {
 public:
	Clock() { ; } ;
	void start() ;
	void stop() ;
	float getDuration() ;
 private:
	std::chrono::time_point<std::chrono::system_clock> startTime ; 
	std::chrono::time_point<std::chrono::system_clock> endTime ; 
} ;		/* -----  end of class Clock  ----- */

#endif /* end of include guard: CLOCK_HPP_YNTIKD6P */
