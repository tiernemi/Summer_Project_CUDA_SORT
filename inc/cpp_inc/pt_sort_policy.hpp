#ifndef PT_SORT_POLICY_HPP_UUQ1HY9D
#define PT_SORT_POLICY_HPP_UUQ1HY9D

/*
 * =====================================================================================
 *
 *       Filename:  pt_sort_policy.hpp
 *
 *    Description:  Header for pt radix sort.
 *
 *        Version:  1.0
 *        Created:  11/08/16 10:58:05
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "centroid.hpp"
#include "camera.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  PTSort
 *  Description:  Pierre Terdiman's radix sort policy. Uses CPU.
 * =====================================================================================
 */

class PTSort {
 protected:
	 std::pair<float*,int*> allocate(const std::vector<Centroid> & centroids) ;
	 void sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) ;
	 void benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
			int * centroidIDs, float * centroidPos, std::vector<float> & times) ;
	 void deAllocate(float * centroidPos, int * centroidIDs) ;
} ;		/* -----  end of class PTSort  ----- */

#endif /* end of include guard: PT_SORT_POLICY_HPP_UUQ1HY9D */
