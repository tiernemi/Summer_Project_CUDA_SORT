/*
 * =====================================================================================
 *
 *       Filename:  test_funcs.cpp
 *
 *    Description:  Helper functions for testing and profiling code.
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:52:41
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <vector>
#include <chrono>
#include <map>
#include "../../inc/test_funcs.hpp"

static std::chrono::time_point<std::chrono::system_clock> start, end ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkSorted
 *    Arguments:  std::vector<Datatype> & list - List of potentially sorted data.
 *      Returns:  True if sorted (ascending), false otherwise.
 *  Description:  Checks if the list is sorted in ascendign order.
 * =====================================================================================
 */

template <typename DataType>
bool checkSorted(std::vector<DataType> & list) {
	DataType prev = *list.begin() ;
	for (auto i = list.begin()+1 ; i != list.end() ; ++i) {
		if (prev > i) {
			return false ;
		}
		prev = *i ;
	}
	return true ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkSorted
 *    Arguments:  std::map<Datatype> & list - List of potentially sorted data.
 *      Returns:  True if sorted (ascending), false otherwise.
 *  Description:  Checks if the list is sorted in ascendign order.
 * =====================================================================================
 */

template <typename DataType>
bool checkSorted(std::unordered_map<int,DataType> & list) {
	DataType prev = list.begin()->second ;
	for (auto i = list.begin()+1 ; i != list.end() ; ++i) {
		if (prev > i->second) {
			return false ;
		}
		prev = i->second ;
	}
	return true ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  startClock
 *  Description:  Starts clock based on system time.
 * =====================================================================================
 */

void startClock() {
	start = std::chrono::system_clock::now() ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  stopClock
 *  Description:  Stops clock.
 * =====================================================================================
 */

void stopClock() {
	end = std::chrono::system_clock::now() ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  getDuration
 *      Returns:  Elapsed time.
 *  Description:  Gets the elapsed time from the stop to start of clock.
 * =====================================================================================
 */

float getDuration() {
	std::chrono::duration<float> elapsed_seconds = end - start ;
	return elapsed_seconds.count() ;
}

bool checkSorted(std::unordered_map<int,float> & list) ;
bool checkSorted(std::vector<float> & list) ;
