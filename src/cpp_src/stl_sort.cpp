/*
 * =====================================================================================
 *
 *       Filename:  stl_sort.cpp
 *
 *    Description:  Implementaion of stl sort on cpu.
 *
 *        Version:  1.0
 *        Created:  13/06/16 09:53:32
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <algorithm>
#include <iostream>

// Custom Headers //
#include "../../inc/stl_sort.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/clock.hpp"
#include "../../inc/test_funcs.hpp"


namespace CPUSorts {

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2) ;

/* 
 * ===  MEMBER FUNCTION CLASS : StlSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *  Description:  Uses stl sort to sort triangles.
 * =====================================================================================
 */

void STLSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	std::vector<Triangle> orig_triangles = triangles ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		Transforms::transformToDistVec(distances, triangles, cameras[i]) ;
		sortDistances(distances) ;
		// Reorder triangles. //
		for (unsigned int k = 0 ; k < distances.size() ; ++k) {
			temp[k] = orig_triangles[distances[k].first] ;
		}
		triangles = temp ;
	}
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : STLSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *  Description:  Sorts vector of distances based on camera location. Use stl sort.
 * =====================================================================================
 */

void STLSort::sortDistances(std::vector<std::pair<int,float>> & distances) {
	std::sort(distances.begin(), distances.end(), comparVec) ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : StlSort  ==============================================
 *         Name:  sortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<float> & times - Times taken to sort for eacg camera.
 *  Description:  Uses stl sort to sort triangles and times these sorts.
 * =====================================================================================
 */

void STLSort::sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
		std::vector<float> & times) {

	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	std::vector<Triangle> temp = triangles ;
	std::vector<Triangle> orig_triangles = triangles ;
	std::vector<float> newTimes ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		Transforms::transformToDistVec(distances, triangles, cameras[i]) ;
		float sortTime = 0.f ;
		sortDistances(distances, sortTime) ;
		// Reorder triangles. //
		for (unsigned int k = 0 ; k < distances.size() ; ++k) {
			temp[k] = orig_triangles[distances[k].first] ;
		}
		triangles = temp ;
		newTimes.push_back(sortTime) ;
	}
	times = newTimes ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : STLSort  ==============================================
 *         Name:  function
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Vector of distances and
 *                ids.
 *                float & sortTime - Time taken to sort.
 *  Description:  Sorts vector of distances based on camera location. Use std sort.
 * =====================================================================================
 */

void STLSort::sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) {
	Clock clock ;
	clock.start() ;
	std::sort(distances.begin(), distances.end(), comparVec) ;
	clock.stop() ;
	sortTime = clock.getDuration() ;
}		/* -----  end of member function function  ----- */

static bool comparVec(const std::pair<int,float> & el1, const std::pair<int,float> & el2)  {
		return (el1.second < el2.second) ;
}

}
