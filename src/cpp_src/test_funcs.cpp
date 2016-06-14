/*
 * =====================================================================================
 *
 *       Filename:  test_funcs.cpp
 *
 *    Description:  Helper functions for testing and profiling code.
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:52:41
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <vector>
#include <chrono>
#include <map>
#include <random>
#include <iostream>

// Custom Headers //
#include "../../inc/test_funcs.hpp"
#include "../../inc/stl_sort.hpp"
#include "../../inc/transforms.hpp"

namespace Tests {

static unsigned long int merge_inv(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
		int left, int right) ;
static unsigned long int merge(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
			int left, int mid, int right) ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcPercentSorted
 *    Arguments:  const std::vector<Triangle> & triangles - Vector of triangles
 *                const Camera & camera - Camera
 *      Returns:  The sorting score of triangles relative to camera, 0 sorted 1 
 *                not sorted.
 *  Description:  Uses mergesort to count the number of inversions in dataset. This
 *                is normalised by NCr2 to generate a sorting score. 0 is highly sorted
 *                while 1 is sorted the other direction. 0.5 is random.
 *                Uses algorithm from http://geeksforgeeks.org
 * =====================================================================================
 */

float calcPercentSorted(const std::vector<Triangle> & triangles, const Camera & camera) {
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	Transforms::transformToDistVec(distances,triangles,camera) ;
	std::vector<std::pair<int,float>> temp = distances ;
	unsigned long int numInvers = merge_inv(distances,temp,0,distances.size()-1) ;
	return numInvers*(2.f/float((distances.size()-1)*(distances.size()))) ;

}		/* -----  end of function calcPercentSorted  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  makePercentSorted:w
 *
 *    Arguments:  
 *      Returns: RETURN TO THIS 
 *  Description:  
 * =====================================================================================
 */

void makePercentSorted(std::vector<Triangle> & triangles, Camera & camera, float percent, 
		std::mt19937 & gen) {

	CPUSorts::STLSort stlsorter ;
	stlsorter.sortTriangles(triangles,camera) ;
	std::vector<std::pair<int,float>> sortedDistances(triangles.size()) ;
	Transforms::transformToDistVec(sortedDistances,triangles,camera) ;
	std::vector<int> unvistedIndices(sortedDistances.size()) ;

	int numInvers = 0 ;
	int reqInvers = std::round(percent*(((sortedDistances.size()-1)*(sortedDistances.size()))/2)) ;

	for (unsigned int i = 0 ; i < unvistedIndices.size() ; ++i) {
		unvistedIndices[i] = i ;
	}
	std::cout << reqInvers << std::endl;
	while (numInvers < reqInvers) {
		std::uniform_int_distribution<int> indices(0,unvistedIndices.size()) ;
		int index1 = indices(gen) ;
		unvistedIndices.erase(unvistedIndices.begin()+index1) ;
		indices = std::uniform_int_distribution<int>(0,unvistedIndices.size()) ;
		int index2 = indices(gen) ;
		unvistedIndices.erase(unvistedIndices.begin()+index2) ;
		std::swap(sortedDistances[index1],sortedDistances[index2]) ;
		++numInvers ;
		std::cout << numInvers << std::endl;
	}

	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		temp[i] = triangles[sortedDistances[i].first] ;
	}
}		/* -----  end of function makePercentSorted  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  merge_inv
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances.
 *                std::vector<std::pair<int,float>> & temp - Temp vector.
 *                int left - Left index.
 *                int right - Right index.
 *      Returns:  Number of inversions.
 *  Description:  Recursively called merge sort routine. Counts number of inversions due
 *                to merging step.
 * =====================================================================================
 */

static unsigned long int merge_inv(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
		int left, int right)  {
	unsigned long int inv_count = 0 ;
	int mid ;
	if (right > left) {
		mid = (right + left)/2 ;
		inv_count = merge_inv(distances,temp,left,mid) ;
		inv_count += merge_inv(distances,temp,mid+1,right) ;
		inv_count += merge(distances,temp, left, mid+1, right) ;
	}
	return inv_count ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  merge
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances.
 *                std::vector<std::pair<int,float>> & temp - Temp vector.
 *                int left - Left index of distances.
 *                int mid - Middle index of distances.
 *                int right - Right index of distances.
 *      Returns:  Number of inversions.
 *  Description:  Merges from left index to mid and mid+1 to right index and sorts
 *                during this merge. The number of inversions is also counted.
 * =====================================================================================
 */

static unsigned long int merge(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
			int left, int mid, int right) {
	int i,j,k ;
	unsigned long int inv_count = 0 ;

	i = left ;
	j = mid ;
	k = left ;

	while ((i <= mid-1) && (j <= right)) {
		if (distances[i].second <= distances[j].second) {
			temp[k++] = distances[i++] ;
		} else {
			temp[k++] = distances[j++] ;
			inv_count += (mid - i) ;
		}
	}
	while (i <= mid -1) {
		temp[k++] = distances[i++] ;
	}
	while (j <= right) {
		temp[k++] = distances[j++] ;
	}
	for (int i = left ; i <= right ; ++i) {
		distances[i] = temp[i] ;
	}
	return inv_count ;
}

}
