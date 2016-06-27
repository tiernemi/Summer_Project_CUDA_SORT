/*
 * =====================================================================================
 *
 *       Filename:  radix_gpu_sort.cpp
 *
 *    Description:  Implementaion of radix sort on gpu.
 *
 *        Version:  1.0
 *        Created:  2016-06-27 11:15
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <iostream>
#include <stdlib.h>

// Custom Headers //
#include "../../inc/cpp_inc/radix_gpu_sort.hpp"
#include "../../inc/cpp_inc/radix_gpu_sort_funcs.hpp"

namespace GPUSorts {

/* 
 * ===  MEMBER FUNCTION CLASS : RadixGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses stl sort to sort triangles.
 * =====================================================================================
 */

void RadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	std::vector<Camera> cameras(1) ;
	cameras[0] = camera ;
	cudaRadixSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void RadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
	std::vector<Camera> cameras(1) ;
	std::vector<float> times(1) ;
	cameras[0] = camera ;
	cudaRadixSortTriangles(triangles,cameras,times) ;
	sortTime = times[0] ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void RadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	cudaRadixSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : RadixGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void RadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	cudaRadixSortTriangles(triangles,cameras,times) ;
}		/* -----  end of member function function  ----- */

}
