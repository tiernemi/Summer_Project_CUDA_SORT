/*
 * =====================================================================================
 *
 *       Filename:  stl_sort.cpp
 *
 *    Description:  Implementaion of stl sort on cpu.
 *
 *        Version:  1.0
 *        Created:  13/06/16 09:53:32
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <algorithm>
#include <iostream>

// Custom Headers //
#include "../../inc/stl_sort.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"


namespace CPUSorts {

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2) ;

/* 
 * ===  MEMBER FUNCTION CLASS : STLSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void STLSort::sortDistances(std::vector<std::pair<int,float>> & distances) {
	std::sort(distances.begin(), distances.end(), comparVec) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : STLSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void STLSort::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	Clock clock ;
	clock.start() ;
	std::sort(distances.begin(), distances.end(), comparVec) ;
	clock.stop() ;
	sortTime = clock.getDuration() ;
}		/* -----  end of member function function  ----- */

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2)  {
		return (el1.second < el2.second) ;
}

}
