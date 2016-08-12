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
#include <algorithm>
#include <iostream>

// Custom Headers //
#include "../../inc/cpp_inc/test_funcs.hpp"
#include "../../inc/cpp_inc/transforms.hpp"

namespace Tests {

static unsigned long int merge_inv(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
		int left, int right) ;
static unsigned long int merge(std::vector<std::pair<int,float>> & distances, std::vector<std::pair<int,float>> & temp,
			int left, int mid, int right) ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcPercentSorted
 *    Arguments:  const std::vector<Centroid> & centroids - Vector of centroids
 *                const Camera & camera - Camera
 *      Returns:  The sorting score of centroids relative to camera, 0 sorted 1 
 *                not sorted.
 *  Description:  Uses mergesort to count the number of inversions in dataset. This
 *                is normalised by NCr2 to generate a sorting score. 0 is highly sorted
 *                in ascending order while 1 is sorted the other direction. 0.5 is random.
 *                Uses algorithm from http://geeksforgeeks.org
 * =====================================================================================
 */

float calcPercentSorted(const std::vector<Centroid> & centroids, const Camera & camera) {
	std::vector<std::pair<int,float>> distances(centroids.size()) ;
	Transforms::transformToDistVec(distances,centroids,camera) ;
	std::vector<std::pair<int,float>> temp = distances ;
	unsigned long int numInvers = merge_inv(distances,temp,0,distances.size()-1) ;
	return numInvers*(2.f/float((distances.size()-1)*(distances.size()))) ;

}		/* -----  end of function calcPercentSorted  ----- */

/*  
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  makePercentSorted
 *    Arguments:  std::vector<Centroid> & centroids - Centroids which will be inversed
 *                until the desired sort score is reached.
 *                Camera & camera - Camera to be sorted relative to.
 *                float score - Desired sort score 0 is sorted ascending, 0.5 is random and
 *                1.0 is sorted descending.
 *                std::mt19937 & gen - Mersenne twister RNG.
 *  Description:  Chooses random pairs of indices and inverts them if the total number of
 *                inversions in the array increases or decreases depending on the approach
 *                regime. If the score is > 0.5 its more efficient to work from an inverted
 *                array while if it's < 0.5 it's more efficient to start from a sorted
 *                array. The algorithm is is stochastic in nature and thus only approximately
 *                gives the desired sort score. This should "randomly sort" array.
 * =====================================================================================
 

void makePercentSorted(std::vector<Centroid> & centroids, Camera & camera, float score, 
		std::mt19937 & gen) {
	// Presort the distances. //
//	CentroidSorter<STLSort> stlSorter(centroids) ;
	std::vector<std::pair<int,float>> sortedDistances(centroids.size()) ;
	Transforms::transformToDistVec(sortedDistances,centroids,camera) ;
	// Index selection distribution. //
	std::uniform_int_distribution<int> indices(0,centroids.size()-1) ;
	// This is the number of inverses required for a given score. //
	unsigned long int targetNumIvers = score*((centroids.size())*(centroids.size()-1))/2 ;
	unsigned long int numInvers = 0 ;
	// If >0.5 approach from reverse sorted array. //
	if (score > 0.5) {
		std::reverse(sortedDistances.begin(), sortedDistances.end()) ;
		numInvers = ((centroids.size())*(centroids.size()-1))/2 ;
		while (numInvers > targetNumIvers) {
			int indexCand1 = indices(gen) ;
			int indexCand2 = indices(gen) ;
			while (indexCand1 == indexCand2) {
				indexCand2 = indices(gen) ;
			}
			int index1 = std::min(indexCand1,indexCand2) ;
			int index2 = std::max(indexCand1,indexCand2) ;

			// Swapped values. //
			float newLeftVal = sortedDistances[index2].second ;
			float newRightVal = sortedDistances[index1].second ;

			// Calculate the change in the number of inverses. Left<-Right //
			int deltaInver = 0 ;
			int numGtTowardsLeft = 0 ;
			int numLtTowardsLeft = 0 ;
			for (int i = index2-1 ; i > index1 ; --i) {
				if (newRightVal < sortedDistances[i].second) {
					++numGtTowardsLeft ;
				}
			}
			if (newRightVal < newLeftVal) {
				++numGtTowardsLeft ;
			}
			numLtTowardsLeft = (index2 - index1) - numGtTowardsLeft ;
			
			// Calculate the change in the number of inverses. Left->Right //
			int numGtTowardsRight = 0 ;
			int numLtTowardsRight = 0 ;
			for (int i = index1+1 ; i < index2 ; ++i) {
				if (newLeftVal < sortedDistances[i].second) {
					++numGtTowardsRight ;
				}
			}
			numLtTowardsRight = (index2 - index1) - numGtTowardsRight ;

			deltaInver = numGtTowardsLeft - numLtTowardsLeft + numLtTowardsRight - numGtTowardsRight ;
			// Only accept decreasing swaps. //
			if (deltaInver < 0) {
				numInvers += deltaInver ;
				std::swap(sortedDistances[index1],sortedDistances[index2]) ;
			}
		}
	} else {
		while (numInvers < targetNumIvers) {
			int indexCand1 = indices(gen) ;
			int indexCand2 = indices(gen) ;
			while (indexCand1 == indexCand2) {
				indexCand2 = indices(gen) ;
			}
			int index1 = std::min(indexCand1,indexCand2) ;
			int index2 = std::max(indexCand1,indexCand2) ;
			// Swapped values. //
			float newLeftVal = sortedDistances[index2].second ;
			float newRightVal = sortedDistances[index1].second ;

			int deltaInver = 0 ;
			int numGtTowardsLeft = 0 ;
			int numLtTowardsLeft = 0 ;
			// Calculate the change in the number of inverses. Left<-Right //
			for (int i = index2-1 ; i > index1 ; --i) {
				if (newRightVal < sortedDistances[i].second) {
					++numGtTowardsLeft ;
				}
			}
			if (newRightVal < newLeftVal) {
				++numGtTowardsLeft ;
			}
			numLtTowardsLeft = (index2 - index1) - numGtTowardsLeft ;
			
			// Calculate the change in the number of inverses. Left->Right //
			int numGtTowardsRight = 0 ;
			int numLtTowardsRight = 0 ;
			for (int i = index1+1 ; i < index2 ; ++i) {
				if (newLeftVal < sortedDistances[i].second) {
					++numGtTowardsRight ;
				}
			}
			numLtTowardsRight = (index2 - index1) - numGtTowardsRight ;

			deltaInver = numGtTowardsLeft - numLtTowardsLeft + numLtTowardsRight - numGtTowardsRight ;
			// Only accept increasing swaps. //
			if (deltaInver > 0) {
				numInvers += deltaInver ;
				std::swap(sortedDistances[index1],sortedDistances[index2]) ;
			}
		}
	}

	std::vector<Centroid> temp = centroids ;
	for (unsigned int i = 0 ; i < centroids.size() ; ++i) {
		temp[i] = centroids[sortedDistances[i].first] ;
	}

	centroids = temp ;
}		 -----  end of function makePercentSorted  ----- */

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
