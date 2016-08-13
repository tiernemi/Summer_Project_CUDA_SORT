#ifndef STL_SORT_POLICY_HPP_UUQ1HY9D
#define STL_SORT_POLICY_HPP_UUQ1HY9D

/*
 * =====================================================================================
 *
 *       Filename:  stl_sort_policy.hpp
 *
 *    Description:  Header for stl_sort policy.
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
 *         Name:  STLSort
 *  Description:  STL sort policy. Uses CPU and stl sort.
 * =====================================================================================
 */

class STLSort {
 protected:
	 std::pair<float*,int*> allocate(const std::vector<Centroid> & centroids) ;
	 void sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) ;
	 void benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
			int * centroidIDs, float * centroidPos, std::vector<float>  & times) ;
	 void deAllocate(float * centroidPos, int * centroidIDs) ;
} ;		/* -----  end of class STLSort  ----- */

#endif /* end of include guard: STL_SORT_POLICY_HPP_UUQ1HY9D */
