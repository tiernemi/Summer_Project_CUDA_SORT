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

#include "../../inc/cpu_sort.hpp"
#include "../../inc/transforms.hpp"

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

void CpuSort::sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) {
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
		float sortTime = 0 ;
		sortTriangles(triangles,cameras[i],sortTime) ;
		newTimes.push_back(sortTime) ;
	}
	times = newTimes ;
}		/* -----  end of member function function  ----- */

}

