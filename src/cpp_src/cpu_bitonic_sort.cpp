/*
 * =====================================================================================
 *
 *       Filename:  cpu_bitonic_sort.cpp
 *
 *    Description:  Source file for normalised bitonic sort.
 *
 *        Version:  1.0
 *        Created:  10/06/16 15:20:03
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>

// Custom Headers //

namespace CPUSorts {

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  MergeUp
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

static void mergeUp(std::vector<std::pair<int,float>>::iterator iter, int length) {
	int mid = length/2 ;
	// outer. //
	for (int i = 1, j = mid-1 ; i < length ; i+=2, --j) {
		if ((*(iter+j)).second > (*(iter+j+i)).second) {
			std::swap((*(iter+j)),(*(iter+j+i))) ;
		}
	}
	mid /= 2 ;

	// inner. //
	while (mid > 0) {
		for (int i = 0 ; i < length ; i+=mid*2 ) {
			for (int j = i , k = 0 ; k < mid ; ++k , ++j) {
				if ((*(iter+j)).second > (*(iter+j+mid)).second) {
					std::swap((*(iter+j)),(*(iter+j+mid))) ;
				}
			}
		}
		mid /= 2 ;
	}

}		/* -----  end of function MergeUp  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cpu_bitonic_sort
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

static void bitonicBuildNorm(std::vector<std::pair<int,float>> & distances) {
	for (unsigned int i = 2 ; i <= distances.size() ; i*=2) {
		for (unsigned int j = 0 ; j < distances.size() ; j += i) {
			mergeUp(distances.begin()+j, i) ;
		}
	}
}		/* -----  end of function bitonic_build  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cpuBitonicSort
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Uses normalised
 *                bitonic sort algorithm.
 * =====================================================================================
 */

void cpuBitonicSort(std::vector<std::pair<int,float>> & distances) {
	bitonicBuildNorm(distances) ;
}		/* -----  end of function cpuBitonicSort  ----- */

}


