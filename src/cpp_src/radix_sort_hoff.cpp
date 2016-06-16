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
#include "../../inc/radix_sort_hoff.hpp"
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
#else
#define pf(x)
#define pf2(x)
#endif

// ---- utils for accessing 11-bit quantities
#define _0(x)	(x & 0x7FF)
#define _1(x)	(x >> 11 & 0x7FF)
#define _2(x)	(x >> 22 )


namespace CPUSorts {

typedef unsigned char ubyte ;
static const unsigned int histSize = 1024 ;
static const unsigned int offSize = 256 ;
static void sortRadices(const float * input, unsigned int * indicesEx, unsigned int size) ;

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses stl sort to sort triangles.
 * =====================================================================================
 */

void RadixSortHoff::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	Transforms::transformToDistVec(distances, triangles, camera) ;
	sortDistances(distances) ;
	// Reorder triangles. //
	for (unsigned int k = 0 ; k < distances.size() ; ++k) {
		temp[k] = triangles[distances[k].first] ;
	}
	triangles = temp ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void RadixSortHoff::sortDistances(std::vector<std::pair<int,float>> & distances) {
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
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void RadixSortHoff::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	Transforms::transformToDistVec(distances, triangles, camera) ;
	sortDistances(distances, sortTime) ;
	// Reorder triangles. //
	for (unsigned int k = 0 ; k < distances.size() ; ++k) {
		temp[k] = triangles[distances[k].first] ;
	}
	triangles = temp ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void RadixSortHoff::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
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
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void RadixSortHoff::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		sortTriangles(triangles,cameras[i]) ;
	}
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixSortHoff  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void RadixSortHoff::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	std::vector<float> newTimes ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		float sortTime = 0 ;
		sortTriangles(triangles,cameras[i],sortTime) ;
		newTimes.push_back(sortTime) ;
	}
	times = newTimes ;
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
	memcpy(indicesEx,indices2,sizeof(unsigned int)*size) ;

	delete [] indices1 ;
	delete [] indices2 ;
}

}
