/*
 * =====================================================================================
 *
 *       Filename:  bitonic_gpu_sort.cpp
 *
 *    Description:  Implementaion of bitonic sort on cpu.
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
#include "../../inc/bitonic_gpu_sort.hpp"
#include "../../inc/bitonic_gpu_sort_funcs.hpp" // CUDA code
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

namespace GPUSorts {

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses bitonic gpu sort to sort triangles.
 * =====================================================================================
 */

void BitonicGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	std::vector<Camera> cameras(1) ;
	cameras[0] = camera ;
	cudaBitonicSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses bitonic gpu sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void BitonicGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
	std::vector<Camera> cameras(1) ;
	std::vector<float> times(1) ;
	cameras[0] = camera ;
	sortTriangles(triangles,camera) ;
	cudaBitonicSortTriangles(triangles,cameras,times) ;
	sortTime = times[0] ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses bitonic gpu sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void BitonicGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	cudaBitonicSortTriangles(triangles,cameras) ;
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : BitonicGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses bitonic gpu sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void BitonicGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	cudaBitonicSortTriangles(triangles,cameras,times) ;
}		/* -----  end of member function function  ----- */





}
