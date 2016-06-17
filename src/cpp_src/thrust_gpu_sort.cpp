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
#include "../../inc/thrust_gpu_sort.hpp"
#include "../../inc/thrust_gpu_sort_funcs.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"

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
	cudaSortTriangles(triangles,cameras) ;
	/* 
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
	*/
}		/* -----  end of member function function  ----- */
/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void ThrustGPUSort::sortDistances(std::vector<std::pair<int,float>> & distances) {
	std::vector<int> ids(distances.size()) ;
	std::vector<float> dists(distances.size()) ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		ids[i] = distances[i].first ;
		dists[i] = distances[i].second ;
	}
	cudaSortDistances(ids,dists) ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		distances[i].first = ids[i] ;
		distances[i].second = dists[i] ;
	}
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

void ThrustGPUSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
	std::vector<Camera> cameras(1) ;
	cameras[0] = camera ;
	sortTriangles(triangles,camera) ;
	cudaSortTriangles(triangles,cameras) ;
	/* 
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	Transforms::transformToDistVec(distances, triangles, camera) ;
	sortDistances(distances, sortTime) ;
	// Reorder triangles. //
	for (unsigned int k = 0 ; k < distances.size() ; ++k) {
		temp[k] = triangles[distances[k].first] ;
	}
	triangles = temp ; */
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : ThrustGPUSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void ThrustGPUSort::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	std::vector<int> ids(distances.size()) ;
	std::vector<float> dists(distances.size()) ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		ids[i] = distances[i].first ;
		dists[i] = distances[i].second ;
	}
	cudaSortDistances(ids,dists,sortTime) ;
	for (unsigned int i = 0 ; i < distances.size() ; ++i) {
		distances[i].first = ids[i] ;
		distances[i].second = dists[i] ;
	}
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
	cudaSortTriangles(triangles,cameras) ;
	/* 
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		sortTriangles(triangles,cameras[i]) ;
	} */
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
	cudaSortTriangles(triangles,cameras) ;
	/*  
	std::vector<float> newTimes ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		float sortTime = 0 ;
		sortTriangles(triangles,cameras[i],sortTime) ;
		newTimes.push_back(sortTime) ;
	}
	times = newTimes ;
	*/
}		/* -----  end of member function function  ----- */

}
