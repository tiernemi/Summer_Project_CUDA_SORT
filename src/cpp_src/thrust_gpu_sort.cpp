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
#include "../../inc/cpp_inc/thrust_gpu_sort.hpp"
#include "../../inc/cpp_inc/thrust_gpu_sort_funcs.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/clock.hpp"
#include "../../inc/cpp_inc/test_funcs.hpp"

namespace GPUSorts {

/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *  Description:  Uses stl sort to sort triangles.
 * =====================================================================================
 */

void ThrustGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
	std::vector<Camera> cameras(1) ;
	cameras[0] = camera ;
	cudaThrustSortTriangles(triangles,cameras) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void ThrustGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) {
	std::vector<Camera> cameras(1) ;
	std::vector<float> gpuTimes ;
	cameras[0] = camera ;
	sortTriangles(triangles,camera) ;
	cudaThrustSortTriangles(triangles,cameras,gpuTimes) ;
	times = gpuTimes ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void ThrustGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	cudaThrustSortTriangles(triangles,cameras) ;
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void ThrustGPUSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	cudaThrustSortTriangles(triangles,cameras,times) ;
}		/* -----  end of member function function  ----- */

}
