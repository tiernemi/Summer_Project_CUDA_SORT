#ifndef TEST_FUNCS_HPP_RE69JCLG
#define TEST_FUNCS_HPP_RE69JCLG

/*
 * =====================================================================================
 *
 *       Filename:  test_funcs.hpp
 *
 *    Description:  Header file for test functions.
 *
 *        Version:  1.0
 *        Created:  07/06/16 17:19:00
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <unordered_map>

template <typename DataType>
bool checkSorted(std::vector<DataType> & list) ;

template <typename DataType>
bool checkSorted(std::unordered_map<int,DataType> & list) ;

void startClock() ;

void stopClock() ;

float getDuration() ;

#endif /* end of include guard: TEST_FUNCS_HPP_RE69JCLG */
