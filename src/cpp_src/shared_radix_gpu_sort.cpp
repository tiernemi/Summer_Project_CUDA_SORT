/*
 * =====================================================================================
 *
 *       Filename:  shared_radix_gpu_sort.cpp
 *
 *    Description:  Implementaion of shared radix sort on the gpu.
 *
 *        Version:  1.0
 *        Created:  2016-07-01 14:50
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
#include "../../inc/cpp_inc/shared_radix_gpu_sort.hpp"
#include "../../inc/cpp_inc/shared_radix_gpu_sort_funcs.hpp"

namespace GPUSorts {

/* 
 * ===  MEMBER FUNCTION CLASS : SharedRadixGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses shared radix sort to sort triangles.
 * =====================================================================================
 */

void SharedRadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	std::vector<Camera> cameras(1) ;
	cameras[0] = camera ;
	cudaSharedRadixSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : SharedRadixGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses shared radix sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void SharedRadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) {
	std::vector<Camera> cameras(1) ;
	std::vector<float> gpuTimes ;
	cameras[0] = camera ;
	cudaSharedRadixSortTriangles(triangles,cameras,gpuTimes) ;
	times = gpuTimes ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : SharedRadixGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses shared radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void SharedRadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	cudaSharedRadixSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : SharedRadixGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses shared radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void SharedRadixGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	cudaSharedRadixSortTriangles(triangles,cameras,times) ;
}		/* -----  end of member function function  ----- */

}
