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

namespace Tests {

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkSorted
 *    Arguments:  std::vector<Datatype> & list - List of potentially sorted data.
 *      Returns:  True if sorted (ascending), false otherwise.
 *  Description:  Checks if the list is sorted in ascendign order.
 * =====================================================================================
 */

template <typename DataType>
bool checkSorted(std::vector<std::pair<int,DataType>> & list) {
	DataType prev = (*list.begin()).second ;
	for (auto i = list.begin()+1 ; i != list.end() ; ++i) {
		if (prev > (*i).second) {
			return false ;
		}
		prev = (*i).second ;
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
	DataType prev = (*list.begin()).second ;
	for (auto i = list.begin()+1 ; i != list.end() ; ++i) {
		if (prev > (*i).second) {
			return false ;
		}
		prev = (*i).second ;
	}
	return true ;
}


}

#endif /* end of include guard: TEST_FUNCS_HPP_RE69JCLG */
