/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_hoff.cpp
 *
 *    Description:  Implementaion of radix sort on cpu. Hybrid of Terdiman's and
 *                  Hoff's algorithms.
 *
 *        Version:  1.0
 *        Created:  2016-06-16 11:03
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
#include <stdlib.h>

// Custom Headers //
#include "../../inc/radix_sort_hybrid.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

typedef const char *cpointer;

#define PREFETCH 1

#if PREFETCH
#include <xmmintrin.h>	// for prefetch
#define pfval	64
#define pfval2	128
#define pf(x)	_mm_prefetch(cpointer(x + i + pfval), _MM_HINT_T0)
#define pf2(x)	_mm_prefetch(cpointer(x + i + pfval2), _MM_HINT_T0)
#define pfa(x)	_mm_prefetch(cpointer(x + pfval2), _MM_HINT_T0)
#else
#define pf(x)
#define pf2(x)
#endif

// ---- utils for accessing 11-bit quantities
#define _0(x)	(x & 0x7FF)
#define _1(x)	(x >> 11 & 0x7FF)
#define _2(x)	(x >> 22 )


namespace CPUSorts {

static const unsigned int histWidth = 11 ;
static void sortRadices(const float * input, unsigned int * indicesEx, unsigned int size) ;

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortHybrid  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void RadixSortHybrid::sortDistances(std::vector<std::pair<int,float>> & distances) {
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
 * ===  MEMBER FUNCTION CLASS : RadixSortHybrid  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void RadixSortHybrid::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
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
 *         Name:  sortRadices
 *    Arguments:  const float * input - Input distance data
 *                unsigned int * indicesEx - Indices array to be overwritten. 
 *                unsigned int size - Number of distance points.
 *  Description:  Sorts 3 - 11 bit radices using Hoff's algorithm. Indices are outputted
 *                and not distances. First histograms are filled, then offsets are computed
 *                by summing histograms. These offsets are used to write back and overwrite
 *                indices.
 * =====================================================================================
 */

static void sortRadices(const float * input, unsigned int * indicesEx, unsigned int size) {

	unsigned int * array = (unsigned int*) input ;
	unsigned int * indices1 = new unsigned int[size] ;
	unsigned int * indices2 = new unsigned int[size] ;
	for (int i = 0 ; i < size ; ++i) {
		indices1[i] = i ;
		indices2[i] = i ;
	}

	// 3 histograms on the stack:
	const unsigned int kHist = 2048;
	unsigned int b0[kHist*3];
	unsigned int *b1 = b0 + kHist;
	unsigned int *b2 = b1 + kHist;

	memset(b0, 0, 3*kHist*sizeof(unsigned int));

	// Check to see if sorted. //
	bool isSorted = true ;
	unsigned int * arrayEnd = &array[size] ;
	float prevVal = (float) array[0] ;

	// Parallel histogramming pass
	while (array != arrayEnd) {
		pfa(array);
		float val = (float) *(array) ;
		if (val < prevVal) {
			isSorted = false ;
			break ;
		}
		prevVal = val ;
		b0[_0(*array)] ++;
		b1[_1(*array)] ++;
		b2[_2(*array)] ++;
		++array ;
	}
	// If not sorted proceed. //
	if (!isSorted) {
		while (array != arrayEnd) {
			pfa(array);
			b0[_0(*array)] ++;
			b1[_1(*array)] ++;
			b2[_2(*array)] ++;
			++array ;
		}
		
		array = (unsigned int *) input ;
		// 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
	// If there are size byte occurrences of this byte pattern then no need to sort. //
	    unsigned int tsum ;
		if (b0[0] != size) {
			unsigned int sum = 0 ;
			for (unsigned int i = 0; i < kHist; i++) {
				// 11 bit hist. //
				tsum = b0[i] + sum ;
				b0[i] = sum - 1 ;
				sum = tsum ;
			}
			for (unsigned int i = 0; i < size; i++) {
				unsigned int pos = _0(array[indices1[i]]);
				pf2(array) ;
				indices2[++b0[pos]] = indices1[i] ;
			}
			std::swap(indices1,indices2) ;
		} if (b1[0] != size) {
			unsigned int sum = 0 ;
			for (unsigned int i = 0; i < kHist; i++) {
				// 11 bit hist. //
				tsum = b1[i] + sum ;
				b1[i] = sum - 1 ;
				sum = tsum ;
			}
			// byte 1: read/write histogram
			for (unsigned int i = 0; i < size; i++) {
				unsigned int pos = _1(array[indices1[i]]);
				pf2(array);
				indices2[++b1[pos]] = indices1[i] ;
			}
			std::swap(indices1,indices2) ;
		} if (b2[0] != size) {
			unsigned int sum = 0 ;
			for (unsigned int i = 0; i < kHist; i++) {
				// 11 bit hist. //
				tsum = b2[i] + sum ;
				b2[i] = sum - 1 ;
				sum = tsum ;
			}
			// byte 2: read/write histogram
			for (unsigned int i = 0; i < size; i++) {
				unsigned int pos = _2(array[indices1[i]]);
				pf2(array);
				indices2[++b2[pos]] = indices1[i] ;
			}
			std::swap(indices1,indices2) ;
		}
	}

	// to write original:
	memcpy(indicesEx,indices1,sizeof(unsigned int)*size) ;

	delete [] indices1 ;
	delete [] indices2 ;
}

}
