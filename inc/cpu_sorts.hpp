#ifndef CPU_SORTS_HPP_ZSROWPJV
#define CPU_SORTS_HPP_ZSROWPJV

/*
 * =====================================================================================
 *
 *       Filename:  cpu_sorts.hpp
 *
 *    Description:  Header file for cpu sorts.
 *
 *        Version:  1.0
 *        Created:  09/06/16 10:48:20
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <unordered_map>
#include <tuple>

namespace CPUSorts {
	void sortVecSTD(std::vector<std::pair<int,float>> & distances) ;
	void sortMapSTD(std::unordered_map<int,float> & distances) ;
}

#endif /* end of include guard: CPU_SORTS_HPP_ZSROWPJV */
