#ifndef IMPL_RADIX_SORT_POLICY_HPP_UUQ1HY9D
#define IMPL_RADIX_SORT_POLICY_HPP_UUQ1HY9D

/*
 * =====================================================================================
 *
 *       Filename:  impl_radix_sort_policy.hpp
 *
 *    Description:  Header for project implementation of sort.
 *
 *        Version:  1.0
 *        Created:  11/08/16 10:58:05
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "centroid.hpp"
#include "camera.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  ImplRadixSort
 *  Description:  Implementation of 3-kernel radix sort from Duane Merrill's paper.
 * =====================================================================================
 */

class ImplRadixSort {
 protected:
	 std::pair<float*,int*> allocate(const std::vector<Centroid> & centroids) ;
	 void sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) ;
	 void benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
			int * centroidIDs, float * centroidPos, std::vector<float> & times) ;
	 void deAllocate(float * centroidPos, int * centroidIDs) ;
} ;		/* -----  end of class ImplRadixSort  ----- */

#endif /* end of include guard: IMPL_RADIX_SORT_POLICY_HPP_UUQ1HY9D */
