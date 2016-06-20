/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_pt.cpp
 *
 *    Description:  Implementaion of radix sort on cpu.
 *
 *        Version:  1.0
 *        Created:  2016-06-13 13:52
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <iostream>
#include <tuple>
#include <string.h>

// Custom Headers //
#include "../../inc/radix_sort_pt.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

namespace CPUSorts {

typedef unsigned char ubyte ;
static const unsigned int histSize = 1024 ;
static const unsigned int offSize = 256 ;

static void sortRadices(const float * input2, unsigned int * indices1, unsigned int size) ;
static bool createHistogram(const float * buffer, unsigned int * indices1, unsigned int size, 
		unsigned int * byteHistogram) ;
static bool performPassCheck(int histIndex, unsigned int * & count, unsigned int size, const float * input,
		unsigned int * byteHistogram) ;

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortPT  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void RadixSortPT::sortDistances(std::vector<std::pair<int,float>> & distances) {
	float * input = new float[distances.size()] ;
	unsigned int * indices = new unsigned int[distances.size()] ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		input[i] = distances[i].second ;
	}
	sortRadices(input,indices,distances.size()) ;
	std::vector<std::pair<int,float>> temp = distances ; 
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		temp[i] = distances[indices[i]] ;
	}
	distances = temp ;
	delete [] input ;
	delete [] indices ;
	
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortPT  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void RadixSortPT::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	float * input = new float[distances.size()] ;
	unsigned int * indices = new unsigned int[distances.size()] ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		input[i] = distances[i].second ;
	}
	Clock clock ;
	clock.start() ;
	sortRadices(input,indices,distances.size()) ;
	clock.stop() ;
	sortTime = clock.getDuration() ;
	std::vector<std::pair<int,float>> temp = distances ; 
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		temp[i] = distances[indices[i]] ;
	}
	distances = temp ;
	delete [] input ;
	delete [] indices ;
	
}		/* -----  end of member function function  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  radix_sort11
 *    Arguments:  const float * input - Array of floats.
 *                unsigned int * indicesEx - Indices to be overwitten. Contains final
 *                indices.
 *                unsigned int size - Size of input.
 *  Description:  Performs Peter Terdiman's radix sort algorithm for 4 - 8 bit histograms.
 * =====================================================================================
 */

static void sortRadices(const float * input, unsigned int * indicesEx, unsigned int size) {

	// Arrays storing index postions. //
	unsigned int * indices1 = new unsigned int[size] ;
	unsigned int * indices2 = new unsigned int[size] ;
	for (int i = 0 ; i < size ; ++i) {
		indices2[i] = i ;
		indices1[i] = i ;
	}
	// Offsets is an array of sorting offsets created by summing histograms. //
	unsigned int offsets[offSize] ;
	// Histogram stores the number of occurrences of a bit pattern (radix) //
	unsigned int byteHistogram[offSize*4] ;
	// Initialise to zero. //
	memset(byteHistogram,0,sizeof(unsigned int)*histSize) ;
	memset(offsets,0,sizeof(unsigned int)*offSize) ;

	// Type cast to unsigned int in order to perform bit ops  //
	bool isSorted = createHistogram(input,indices1,size,byteHistogram) ;
	if (isSorted) {
		memcpy(indicesEx,indices1,size*sizeof(unsigned int)) ;
		return ;
	}

	// For each histogram perform the sort. //
	for (int i = 0 ; i < 4 ; ++i) {
		unsigned int * count = NULL ;
		// Only perform a pass if everything has not already been sorted. //
		bool passState = performPassCheck(i,count,size,input,byteHistogram) ;
		if(passState) {
			// Generate write offsets. //
			offsets[0] = 0 ;
			for (int j = 1 ; j < offSize ; ++j) {
				offsets[j] = offsets[j-1] + count[j-1] ;
			}

			// Overwrite indices. //
			ubyte * inputBytes = (ubyte*) input ;
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

	memcpy(indicesEx,indices1,size*sizeof(unsigned int)) ;
	delete [] indices1 ;
	delete [] indices2 ;
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
 *                unsigned int size - Number of triangle ids.
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

}
