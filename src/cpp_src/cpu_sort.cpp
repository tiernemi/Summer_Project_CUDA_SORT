/*
 * =====================================================================================
 *
 *       Filename:  cpu_sort.cpp
 *
 *    Description:  Source file for cpu sort class.
 *
 *        Version:  1.0
 *        Created:  20/06/16 17:07:33
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "../../inc/cpp_inc/cpu_sort.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/clock.hpp"

namespace CPUSorts {

void CpuSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera) {
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
}

/* 
 * ===  MEMBER FUNCTION CLASS : CpuSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                Camera & camera - Camera to sort relative to.
 *                float & sortTime - Times taken to sort for each camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void CpuSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) {
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	std::vector<float> cpuTimes ;

	Clock clock ;
	clock.start() ;
	Transforms::transformToDistVec(distances, triangles, camera) ;
	clock.stop() ;
	float transformTime = clock.getDuration() ;
	float sortTime = 0 ;

	sortDistances(distances, sortTime) ;
	// Reorder triangles. //
	for (unsigned int k = 0 ; k < distances.size() ; ++k) {
		temp[k] = triangles[distances[k].first] ;
	}

	cpuTimes.push_back(transformTime+sortTime) ;
	cpuTimes.push_back(transformTime+sortTime) ;
	cpuTimes.push_back(sortTime) ;
	times = cpuTimes ;

	triangles = temp ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : CpuSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void CpuSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		sortTriangles(triangles,cameras[i]) ;
	}
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : CpuSort  ===========================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> times - Vector of sort times for each camera.
 *  Description:  Uses radix sort to sort triangles based on vector of cameras.
 * =====================================================================================
 */

void CpuSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras,
		std::vector<float> & times) {
	std::vector<float> newTimes ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		std::vector<float> tempTimes ;
		sortTriangles(triangles,cameras[i],tempTimes) ;
		// Push back tot, trans and sort times. //
		newTimes.push_back(tempTimes[0]) ;
		newTimes.push_back(tempTimes[1]) ;
		newTimes.push_back(tempTimes[2]) ;
	}
	times = newTimes ;
}		/* -----  end of member function function  ----- */

}

