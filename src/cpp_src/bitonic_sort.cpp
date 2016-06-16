/*
 * =====================================================================================
 *
 *       Filename:  bitonic_sort.cpp
 *
 *    Description:  Implementaion of bitonic sort on cpu.
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

// Custom Headers //
#include "../../inc/bitonic_sort.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

namespace CPUSorts {

static void mergeUp(std::vector<std::pair<int,float>>::iterator iter, int length) ;
static void bitonicBuildNorm(std::vector<std::pair<int,float>> & distances) ;

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses stl sort to sort triangles.
 * =====================================================================================
 */

void BitonicSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	Transforms::transformToDistVec(distances, temp, camera) ;
	sortDistances(distances) ;
	// Reorder triangles. //
	for (unsigned int k = 0 ; k < distances.size() ; ++k) {
		temp[k] = triangles[distances[k].first] ;
	}
	triangles = temp ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void BitonicSort::sortDistances(std::vector<std::pair<int,float>> & distances) {
	bitonicBuildNorm(distances) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void BitonicSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
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
	Transforms::transformToDistVec(distances, triangles, camera) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void BitonicSort::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	Clock clock ;
	clock.start() ;
	bitonicBuildNorm(distances) ;
	clock.stop() ;
	sortTime = clock.getDuration() ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses bitonic sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void BitonicSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		sortTriangles(triangles,cameras[i]) ;
	}
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses bitonic sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void BitonicSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
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
 *         Name:  MergeUp
 *    Arguments:  std::vector<std::pair<int,float>>::iterator iter - Iterator of current
 *                location in the array. 
 *                int length - Length of this segment.
 *  Description:  Merges two bitonic sequences into a larger sequence.
 * =====================================================================================
 */

static void mergeUp(std::vector<std::pair<int,float>>::iterator iter, int length) {
	int mid = length/2 ;
	// outer. //
	for (int i = 1, j = mid-1 ; i < length ; i+=2, --j) {
		if ((*(iter+j)).second > (*(iter+j+i)).second) {
			std::swap((*(iter+j)),(*(iter+j+i))) ;
		}
	}
	mid /= 2 ;

	// inner. //
	while (mid > 0) {
		for (int i = 0 ; i < length ; i+=mid*2 ) {
			for (int j = i , k = 0 ; k < mid ; ++k , ++j) {
				if ((*(iter+j)).second > (*(iter+j+mid)).second) {
					std::swap((*(iter+j)),(*(iter+j+mid))) ;
				}
			}
		}
		mid /= 2 ;
	}

}		/* -----  end of function MergeUp  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cpu_bitonic_sort
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Distances vector.
 *  Description:  Uses bitonic sorting network to progressively merge smaller bitonic
 *                sequences into a larger sorted sequence. Uses normalised version of this
 *                network.
 * =====================================================================================
 */

static void bitonicBuildNorm(std::vector<std::pair<int,float>> & distances) {
	for (unsigned int i = 2 ; i <= distances.size() ; i*=2) {
		for (unsigned int j = 0 ; j < distances.size() ; j+=i) {
			mergeUp(distances.begin()+j, i) ;
		}
	}
}		/* -----  end of function bitonic_build  ----- */

}
