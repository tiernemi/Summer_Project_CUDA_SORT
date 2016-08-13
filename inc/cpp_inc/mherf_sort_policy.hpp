#ifndef MHERF_SORT_POLICY_HPP_UUQ1HY9D
#define MHERF_SORT_POLICY_HPP_UUQ1HY9D

/*
 * =====================================================================================
 *
 *       Filename:  mherf_sort_policy.hpp
 *
 *    Description:  Header for mherf sort.
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
 *         Name:  MHERFSort
 *  Description:  Michael Herf's radix sort policy implemented on CPU.
 * =====================================================================================
 */

class MHerfSort {
 protected:
	 std::pair<float*,int*> allocate(const std::vector<Centroid> & centroids) ;
	 void sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) ;
	 void benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
			int * centroidIDs, float * centroidPos, std::vector<float> & times) ;
	 void deAllocate(float * centroidPos, int * centroidIDs) ;
} ;		/* -----  end of class MHERFSort  ----- */

#endif /* end of include guard: MHERF_SORT_POLICY_HPP_UUQ1HY9D */
