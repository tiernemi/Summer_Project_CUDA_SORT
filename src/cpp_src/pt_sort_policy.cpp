/*
 * =====================================================================================
 *
 *       Filename:  pt_sort_policy.cpp
 *
 *    Description:  Source file for histogram sort written by Pierre Terdiman.
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

#include "stdlib.h"
#include "string.h"
#include "../../inc/cpp_inc/pt_sort_policy.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/clock.hpp"
#include <iostream>


typedef unsigned char ubyte ;
static const unsigned int HIST_SIZE = 1024 ;
static const unsigned int OFF_SIZE = 256 ;
static bool createHistogram(const float * buffer, unsigned int * indices1, unsigned int size, 
		unsigned int * byteHistogram) ;
static bool performPassCheck(int histIndex, unsigned int * & count, unsigned int size, const float * input,
		unsigned int * byteHistogram) ;

/* 
 * ===  MEMBER FUNCTION : PTSort  ======================================================
 *         Name:  allocate
 *    Arguments:  const std::vector<Centroid> & centroids - Centroid data to allocate.
 *      Returns:  Pair of pointers to centroid position data and ids.
 *  Description:  Allocates position and id data on the CPU.
 * =====================================================================================
 */

std::pair<float*,int*> PTSort::allocate(const std::vector<Centroid> & centroids) {
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
 * ===  MEMBER FUNCTION : PTSort  ======================================================
 *         Name:  sort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using a histogram radix sort written by Pierre Terdiman.
 * =====================================================================================
 */

void PTSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) {
	const int size = centroidIDsVec.size() ;

	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;

	// Arrays storing index postions. //
	unsigned int * indices1 = new unsigned int[size] ;
	unsigned int * indices2 = new unsigned int[size] ;
	for (int i = 0 ; i < size ; ++i) {
		indices2[i] = i ;
		indices1[i] = i ;
	}
	// Offsets is an array of sorting offsets created by summing histograms. //
	unsigned int offsets[OFF_SIZE] ;
	// Histogram stores the number of occurrences of a bit pattern (radix) //
	unsigned int byteHistogram[OFF_SIZE*4] ;
	// Initialise to zero. //
	memset(byteHistogram,0,sizeof(unsigned int)*HIST_SIZE) ;
	memset(offsets,0,sizeof(unsigned int)*OFF_SIZE) ;

	// Type cast to unsigned int in order to perform bit ops  //
	bool isSorted = createHistogram(dists,indices1,size,byteHistogram) ;
	if (isSorted) {
		for (int i = 0 ; i < size ; ++i) {
			centroidIDsVec[i] = centroidIDs[indices1[i]] ;
		}
		return ;
	}

	// For each histogram perform the sort. //
	for (int i = 0 ; i < 4 ; ++i) {
		unsigned int * count = NULL ;
		// Only perform a pass if everything has not already been sorted. //
		bool passState = performPassCheck(i,count,size,dists,byteHistogram) ;
		if(passState) {
			// Generate write offsets. //
			offsets[0] = 0 ;
			for (int j = 1 ; j < OFF_SIZE ; ++j) {
				offsets[j] = offsets[j-1] + count[j-1] ;
			}

			// Overwrite indices. //
			ubyte * inputBytes = (ubyte*) dists ;
			unsigned int * indices = indices1 ;
			unsigned int * indicesEnd = &indices1[size] ;
			inputBytes += i ;
			while (indices != indicesEnd) {
				unsigned int id = *indices++ ;
				indices2[offsets[inputBytes[id<<2]]++] = id ;
			}
			std::swap(indices1,indices2) ;
		}
	}

	for (int i = 0 ; i < size ; ++i) {
		centroidIDsVec[i] = centroidIDs[indices1[i]] ;
	}
	
	delete [] indices1 ;
	delete [] indices2 ;
}

/* 
 * ===  MEMBER FUNCTION : PTSort  ======================================================
 *         Name:  benchSort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *                std::vector<float> & times - Vector used to store timings.
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using a histogram radix sort written by Pierre Terdiman.
 *                This version also benchmarks.
 * =====================================================================================
 */

void PTSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos, std::vector<float> & times) {
	const int size = centroidIDsVec.size() ;

	Clock clock ;
	clock.start() ;
	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;
	clock.stop() ;
	float transformTime = clock.getDuration() ;

	clock.start() ;
	// Arrays storing index postions. //
	unsigned int * indices1 = new unsigned int[size] ;
	unsigned int * indices2 = new unsigned int[size] ;
	for (int i = 0 ; i < size ; ++i) {
		indices2[i] = i ;
		indices1[i] = i ;
	}
	// Offsets is an array of sorting offsets created by summing histograms. //
	unsigned int offsets[OFF_SIZE] ;
	// Histogram stores the number of occurrences of a bit pattern (radix) //
	unsigned int byteHistogram[OFF_SIZE*4] ;
	// Initialise to zero. //
	memset(byteHistogram,0,sizeof(unsigned int)*HIST_SIZE) ;
	memset(offsets,0,sizeof(unsigned int)*OFF_SIZE) ;

	// Type cast to unsigned int in order to perform bit ops  //
	bool isSorted = createHistogram(dists,indices1,size,byteHistogram) ;
	if (isSorted) {
		clock.stop() ;
		float sortTime = clock.getDuration() ;
		times.push_back(sortTime) ;
		times.push_back(sortTime+transformTime) ;
		for (int i = 0 ; i < size ; ++i) {
			centroidIDsVec[i] = centroidIDs[indices1[i]] ;
		}
		delete [] indices1 ;
		delete [] indices2 ;
		clock.stop() ;
		float copyTime = clock.getDuration() ;
		times.push_back(sortTime+transformTime+copyTime) ;
		return ;
	}

	// For each histogram perform the sort. //
	for (int i = 0 ; i < 4 ; ++i) {
		unsigned int * count = NULL ;
		// Only perform a pass if everything has not already been sorted. //
		bool passState = performPassCheck(i,count,size,dists,byteHistogram) ;
		if(passState) {
			// Generate write offsets. //
			offsets[0] = 0 ;
			for (int j = 1 ; j < OFF_SIZE ; ++j) {
				offsets[j] = offsets[j-1] + count[j-1] ;
			}

			// Overwrite indices. //
			ubyte * inputBytes = (ubyte*) dists ;
			unsigned int * indices = indices1 ;
			unsigned int * indicesEnd = &indices1[size] ;
			inputBytes += i ;
			while (indices != indicesEnd) {
				unsigned int id = *indices++ ;
				indices2[offsets[inputBytes[id<<2]]++] = id ;
			}
			std::swap(indices1,indices2) ;
		}
	}
	clock.stop() ;
	float sortTime = clock.getDuration() ;

	clock.start() ;
	for (int i = 0 ; i < size ; ++i) {
		centroidIDsVec[i] = centroidIDs[indices1[i]] ;
	}
	
	delete [] indices1 ;
	delete [] indices2 ;
	clock.stop() ;
	float copyTime = clock.getDuration() ;

	times.push_back(sortTime) ;
	times.push_back(sortTime+transformTime) ;
	times.push_back(sortTime+transformTime+copyTime) ;
}

/* 
 * ===  MEMBER FUNCTION : PTSort  ======================================================
 *         Name:  deAllocate
 *    Arguments:  float * centroidPos - Centroid position location.
 *                int * centroidIDs - Centroid ids location.
 *  Description:  Frees data sotred at pointers.
 * =====================================================================================
 */

void PTSort::deAllocate(float * centroidPos, int * centroidIDs) {
	delete [] centroidPos ;
	delete [] centroidIDs ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  createHistogram
 *    Arguments:  const float * buffer - Buffer of input values. 
 *                unsigned int * indices1 - Current Indices.
 *                unsigned int size - Size of input.
 *		          unsigned int * byteHistogram - Histogram array.
 *      Returns:  True or False
 *  Description:  Creates 4 - 8 bit histograms which are responsible for 8 bits of the 32
 *                bits of the float. Each histogram records the amount of times a bit
 *                pattern is encountered within a 8 bit region for each value in input.
 * =====================================================================================
 */

static bool createHistogram(const float * buffer, unsigned int * indices1, unsigned int size, 
		unsigned int * byteHistogram) {
	
	float prevVal = (float) buffer[indices1[0]] ;
	bool isSorted = true ;
	unsigned int * indices = indices1 ;

	ubyte * p = (ubyte*)buffer ;
	ubyte * pe = &p[size*4] ;
	unsigned int * hist0 = &byteHistogram[0] ;
	unsigned int * hist1 = &byteHistogram[256] ;
	unsigned int * hist2 = &byteHistogram[512] ;
	unsigned int * hist3 = &byteHistogram[768] ;

	// Check if the array is sorted. //
	while (p != pe) {
		float val = (float) buffer[*indices++] ;
		if (val < prevVal) {
			isSorted = false ;
			break ;
		}
		prevVal = val ;
		// Fill histograms for each byte. //
		hist0[*p++]++ ;
		hist1[*p++]++ ;
		hist2[*p++]++ ;
		hist3[*p++]++ ;
	}
	
	if (isSorted) {
		return true ;
	} else {
		while (p != pe) {
			// Fill histograms for each byte. //
			hist0[*p++]++ ;
			hist1[*p++]++ ;
			hist2[*p++]++ ;
			hist3[*p++]++ ;
		}
		return false ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  radix_sort8
 *    Arguments:  int histIndex - The index of the histogram.
 *                unsigned int * & count - Pointer to count.
 *                unsigned int size - Number of centroid ids.
 *                const float * input - Input float values.
 *                unsigned int * byteHistogram - Histogram array.
 *      Returns:  True or False.
 *  Description:  Returns true if input still has to be sorted, false otherwise.
 * =====================================================================================
 */

static bool performPassCheck(int histIndex, unsigned int * & count, unsigned int size, const float * input,
		unsigned int * byteHistogram) {
	// Set count to correct byteHistogram offset. //
	count = &byteHistogram[histIndex<<8] ;
	// The value of the histIndex byte of input array.
	ubyte uniqVal = *(((ubyte*)input)+histIndex) ;
	// If there are size byte occurrences of this byte pattern then no need to sort. //
	if (count[uniqVal]==size) {
		return false ;
	} else {
		return true ;
	}
}

