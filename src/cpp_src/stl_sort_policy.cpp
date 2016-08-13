/*
 * =====================================================================================
 *
 *       Filename:  mherf_sort_policy.cpp
 *
 *    Description:  Source file for stl sort policy.
 *
 *        Version:  1.0
 *        Created:  11/08/16 11:01:19
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <algorithm>
#include <iostream>

#include "../../inc/cpp_inc/stl_sort_policy.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/clock.hpp"

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2)  {
		return (el1.second < el2.second) ;
}

/* 
 * ===  MEMBER FUNCTION : STLSort  === =================================================
 *         Name:  allocate
 *    Arguments:  const std::vector<Centroid> & centroids - Centroid data to allocate.
 *      Returns:  Pair of pointers to centroid position data and ids.
 *  Description:  Allocates position and id data on the CPU.
 * =====================================================================================
 */

std::pair<float*,int*> STLSort::allocate(const std::vector<Centroid> & centroids) {
	std::pair<float*,int*> ptrs ;
	ptrs.first = new float[3*centroids.size()] ;
	ptrs.second = new int[centroids.size()] ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		const float * coords = centroids[i].getCoords() ;
		ptrs.first[3*i] = coords[0] ;
		ptrs.first[3*i+1] = coords[1] ;
		ptrs.first[3*i+2] = coords[2] ;
		ptrs.second[i] = centroids[i].getID() ;
	}
	return ptrs ;
}

/* 
 * ===  MEMBER FUNCTION : STLSort  === ======================================================
 *         Name:  sort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using stl sort.
 * =====================================================================================
 */

void STLSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) {
	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;
	std::vector<std::pair<int,float>> distVec(centroidIDsVec.size()) ;
	for (int i = 0 ; i < centroidIDsVec.size() ; ++i) {
		distVec[i].first = centroidIDs[i] ;
		distVec[i].second = dists[i] ;
	}
	std::sort(distVec.begin(), distVec.end(), comparVec) ;
	for (int i = 0 ; i < centroidIDsVec.size() ; ++i) {
		centroidIDsVec[i] = distVec[i].first ;
	}
	delete [] dists ;
}

/* 
 * ===  MEMBER FUNCTION : STLSort  === ======================================================
 *         Name:  benchSort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *                std::vector<float> & times - Vector used to store timings.
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using stl sort. This version also benchmarks.
 * =====================================================================================
 */

void STLSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos, std::vector<float> & times) {

	Clock clock ;
	clock.start() ;
	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;
	clock.stop() ;
	float transformTime = clock.getDuration() ;

	std::vector<std::pair<int,float>> distVec(centroidIDsVec.size()) ;
	for (int i = 0 ; i < centroidIDsVec.size() ; ++i) {
		distVec[i].first = centroidIDs[i] ;
		distVec[i].second = dists[i] ;
	}

	clock.start() ;
	std::sort(distVec.begin(), distVec.end(), comparVec) ;
	clock.stop() ;
	float sortTime = clock.getDuration() ;
	
	clock.start() ;
	for (int i = 0 ; i < centroidIDsVec.size() ; ++i) {
		centroidIDsVec[i] = distVec[i].first ;
	}
	delete [] dists ;
	clock.stop() ;
	float copyTime = clock.getDuration() ;

	times.push_back(sortTime) ;
	times.push_back(sortTime+transformTime) ;
	times.push_back(sortTime+transformTime+copyTime) ;
}

/* 
 * ===  MEMBER FUNCTION : STLSort  === =================================================
 *         Name:  deAllocate
 *    Arguments:  float * centroidPos - Centroid position location.
 *                int * centroidIDs - Centroid ids location.
 *  Description:  Frees data sotred at pointers.
 * =====================================================================================
 */

void STLSort::deAllocate(float * centroidPos, int * centroidIDs) {
	delete [] centroidPos ;
	delete [] centroidIDs ;
}



