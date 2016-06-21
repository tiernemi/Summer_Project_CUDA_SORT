/*
 * =====================================================================================
 *
 *       Filename:  bubble_sort.cpp
 *
 *    Description:  Implementation of cpu bubble sort.
 *
 *        Version:  1.0
 *        Created:  21/06/16 12:06:20
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */


#include <vector>
#include <iostream>
#include <tuple>

// Custom Headers //
#include "../../inc/bubble_sort.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

namespace CPUSorts {

static void cocktailSort(std::vector<std::pair<int,float>> & distances) ;

/* 
 * ===  MEMBER FUNCTION CLASS : BubbleSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use bubble sort.
 * =====================================================================================
 */

void BubbleSort::sortDistances(std::vector<std::pair<int,float>> & distances) {
	cocktailSort(distances) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BubbleSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use bubble sort.
 * =====================================================================================
 */

void BubbleSort::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	Clock clock ;
	clock.start() ;
	cocktailSort(distances) ;
	clock.stop() ;
	sortTime = clock.getDuration() ;
}		/* -----  end of member function function  ----- */


static void cocktailSort(std::vector<std::pair<int,float>> & distances) {
	int lastSwapLoc = distances.size() ;
	while( lastSwapLoc != 0) {
		// Only need to iterate as far as the last swap location. //
		int upperLoc = lastSwapLoc ;
		lastSwapLoc = 0 ;
		for (int i = 1 ; i < upperLoc ; ++i) {
			if (distances[i-1].second > distances[i].second) {
				std::swap(distances[i-1],distances[i]) ;
				lastSwapLoc = i ;
			}
		}
	}
}

}
