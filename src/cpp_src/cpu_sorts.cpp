/*
 * =====================================================================================
 *
 *       Filename:  cpu_sorts.cpp
 *
 *    Description:  Namespace file containing implementations of cpu sorts.
 *
 *        Version:  1.0
 *        Created:  09/06/16 09:24:46
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <tuple>

// Custom Headers //
#include "../../inc/triangle.hpp"

namespace CPUSorts {

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2)  {
	return (el1.second < el2.second) ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sortVecSTD
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void sortVecSTD(std::vector<std::pair<int,float>> & distances) {
	std::sort(distances.begin(), distances.end(), comparVec) ;
}		/* -----  end of function cpuQsortSTL  ----- */

}
