
#ifndef THRUST_SORT_POLICY_HPP_UUQ1HY9D
#define THRUST_SORT_POLICY_HPP_UUQ1HY9D

/*
 * =====================================================================================
 *
 *       Filename:  thrust_sort_policy.hpp
 *
 *    Description:  Thrust sorting policy. Members implemented using CUDA.
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
 *         Name:  ThrustSort
 *  Description:  Thrust sorting policy. Members implemented using CUDA.
 * =====================================================================================
 */

class ThrustSort {
 protected:
	 std::pair<float*,int*> allocate(const std::vector<Centroid> & centroids) ;
	 void sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) ;
	 void benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
			int * centroidIDs, float * centroidPos, std::vector<float> & times) ;
	 void deAllocate(float * centroidPos, int * centroidIDs) ;
} ;		/* -----  end of class ThrustSort  ----- */

#endif /* end of include guard: THRUST_SORT_POLICY_HPP_UUQ1HY9D */
