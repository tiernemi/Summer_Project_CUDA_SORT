/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_hoff.cpp
 *
 *    Description:  Implementaion of radix sort on cpu.
 *
 *        Version:  1.0
 *        Created:  2016-06-15 12:42
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
#include "../../inc/cpp_inc/mherf_sort_policy.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/clock.hpp"

typedef const char *cpointer;

#define PREFETCH 1

#if PREFETCH
#include <xmmintrin.h>	// for prefetch
#define pfval	64
#define pfval2	128
#define pf(x)	_mm_prefetch(cpointer(x + i + pfval), _MM_HINT_T0)
#define pf2(x)	_mm_prefetch(cpointer(x + i + pfval2), _MM_HINT_T0)
#else
#define pf(x)
#define pf2(x)
#endif

// ---- utils for accessing 11-bit quantities
#define _0(x)	(x & 0x7FF)
#define _1(x)	(x >> 11 & 0x7FF)
#define _2(x)	(x >> 22 )

/* 
 * ===  MEMBER FUNCTION CLASS : pt_sort_policy  ======================================
 *         Name:  function
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

std::pair<float*,int*> MHerfSort::allocate(const std::vector<Centroid> & centroids) {
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
 * ===  FUNCTION  ======================================================================
 *         Name:  sortRadices
 *    Arguments:  const float * dists - Input distance data
 *                unsigned int * indicesEx - Indices array to be overwritten. 
 *                unsigned int size - Number of distance points.
 *  Description:  Sorts 3 - 11 bit radices using Hoff's algorithm. Indices are outputted
 *                and not distances. First histograms are filled, then offsets are computed
 *                by summing histograms. These offsets are used to write back and overwrite
 *                indices.
 * =====================================================================================
 */

void MHerfSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) {
	const int size = centroidIDsVec.size() ;
	// Create distances to be used as keys. //
	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;


	unsigned int * array = (unsigned int*) dists ;
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

	// 1.  parallel histogramming pass
	for (unsigned int i = 0; i < size; i++) {
		pf(array);
		b0[_0(array[i])] ++;
		b1[_1(array[i])] ++;
		b2[_2(array[i])] ++;
	}
	
	// 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
	unsigned int sum0 = 0 ;
	unsigned int sum1 = 0 ; 
	unsigned int sum2 = 0 ;
	unsigned int tsum;
	for (unsigned int i = 0; i < kHist; i++) {
		// 11 bit hist. //
		tsum = b0[i] + sum0;
		b0[i] = sum0 - 1;
		sum0 = tsum;
		// 11 bit hist. //
		tsum = b1[i] + sum1;
		b1[i] = sum1 - 1;
		sum1 = tsum;
		// 11 bit hist. //
		tsum = b2[i] + sum2;
		b2[i] = sum2 - 1;
		sum2 = tsum;
	}

	// byte 0: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _0(array[indices1[i]]);
		//std::cout << pos << std::endl;
		pf2(array) ;
		indices2[++b0[pos]] = indices1[i] ;
	}

	// byte 1: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _1(array[indices2[i]]);
		pf2(array);
		indices1[++b1[pos]] = indices2[i] ;
	}

	// byte 2: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _2(array[indices1[i]]);
		pf2(array);
		indices2[++b2[pos]] = indices1[i] ;
	}

	// to write original:
	//memcpy(indicesEx,indices2,sizeof(unsigned int)*size) ;
	for (int i = 0 ; i < size ; ++i) {
		centroidIDsVec[i] = centroidIDs[indices2[i]]  ;
	}

	delete [] indices1 ;
	delete [] indices2 ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sortRadices
 *    Arguments:  const float * dists - Input distance data
 *                unsigned int * indicesEx - Indices array to be overwritten. 
 *                unsigned int size - Number of distance points.
 *  Description:  Sorts 3 - 11 bit radices using Hoff's algorithm. Indices are outputted
 *                and not distances. First histograms are filled, then offsets are computed
 *                by summing histograms. These offsets are used to write back and overwrite
 *                indices.
 * =====================================================================================
 */

void MHerfSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos, std::vector<float> & times) {
	const int size = centroidIDsVec.size() ;
	// Create distances to be used as keys. //
	Clock clock ;
	clock.start() ;
	float * dists = new float[centroidIDsVec.size()] ;
	Transforms::transformToDistArray(dists,centroidPos,camera,centroidIDsVec.size()) ;
	clock.stop() ;
	float transformTime = clock.getDuration() ;


	clock.start() ;
	unsigned int * array = (unsigned int*) dists ;
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

	// 1.  parallel histogramming pass
	for (unsigned int i = 0; i < size; i++) {
		pf(array);
		b0[_0(array[i])] ++;
		b1[_1(array[i])] ++;
		b2[_2(array[i])] ++;
	}
	
	// 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
	unsigned int sum0 = 0 ;
	unsigned int sum1 = 0 ; 
	unsigned int sum2 = 0 ;
	unsigned int tsum;
	for (unsigned int i = 0; i < kHist; i++) {
		// 11 bit hist. //
		tsum = b0[i] + sum0;
		b0[i] = sum0 - 1;
		sum0 = tsum;
		// 11 bit hist. //
		tsum = b1[i] + sum1;
		b1[i] = sum1 - 1;
		sum1 = tsum;
		// 11 bit hist. //
		tsum = b2[i] + sum2;
		b2[i] = sum2 - 1;
		sum2 = tsum;
	}

	// byte 0: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _0(array[indices1[i]]);
		//std::cout << pos << std::endl;
		pf2(array) ;
		indices2[++b0[pos]] = indices1[i] ;
	}

	// byte 1: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _1(array[indices2[i]]);
		pf2(array);
		indices1[++b1[pos]] = indices2[i] ;
	}

	// byte 2: read/write histogram
	for (unsigned int i = 0; i < size; i++) {
		unsigned int pos = _2(array[indices1[i]]);
		pf2(array);
		indices2[++b2[pos]] = indices1[i] ;
	}
	clock.stop() ;
	float sortTime = clock.getDuration() ;

	clock.start() ;
	// to write original:
	//memcpy(indicesEx,indices2,sizeof(unsigned int)*size) ;
	for (int i = 0 ; i < size ; ++i) {
		centroidIDsVec[i] = centroidIDs[indices2[i]]  ;
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
 * ===  MEMBER FUNCTION CLASS : pt_sort_policy  ======================================
 *         Name:  function
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void MHerfSort::deAllocate(float * centroidPos, int * centroidIDs) {
	delete [] centroidPos ;
	delete [] centroidIDs ;
}

