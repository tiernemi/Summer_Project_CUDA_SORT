/*
 * =====================================================================================
 *
 *       Filename:  transforms.cpp
 *
 *    Description:  Source for useful transform functions used to change the form of the
 *                  sorting data.
 *
 *        Version:  1.0
 *        Created:  09/06/16 09:52:42
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include <tuple>
#include <iostream>
#include <cmath>
#include <unordered_map>

// Custom Headers //
#include "../../inc/triangle.hpp"
#include "../../inc/camera.hpp"
#include "../../inc/transforms.hpp"

namespace Transforms {

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcDistance
 *    Arguments:  const Triangle & triangle - Triangle (Model space)
 *                const Camera & camera - Camera (Model space)
 *      Returns:  Distance between camera and triangle.
 *  Description:  Helper function for calculating distance between camera and triangle.
 * =====================================================================================
 */

static float calcDistance(const Triangle & triangle, const Camera & camera) {
	const float * trico = triangle.getCoords() ;
	const float * camco = camera.getCoords() ;	
	return std::sqrt(std::pow(trico[0]-camco[0],2) + std::pow(trico[1]-camco[1],2) +
			std::pow(trico[2]-camco[2],2)) ;
}		/* -----  end of function   ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transformToDistVec
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Array of distances to
 *                be overwritten.
 *                const std::vector<Triangle> & triangles - Array of truangles in model
 *                space.
 *                const Camera & camera - Camera for this frame.
 *  Description:  Transforms the triangle - camera system to a sortable array of distances.
 * =====================================================================================
 */

void transformToDistVec(std::vector<std::pair<int,float>> & distances, const std::vector<Triangle> & triangles,
		const Camera & camera) {
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		distances[i].first = i ; 
		distances[i].second = calcDistance(triangles[i],camera) ;
	}
}		/* -----  end of function transfor  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transformToDistMap
 *    Arguments:  std::unordered_map<int,float> & distances- Map of distances to
 *                be overwritten.
 *                const std::vector<Triangle> & triangles - Array of truangles in model
 *                space.
 *                const Camera & camera - Camera for this frame.
 *  Description:  Transforms the triangle - camera system to a sortable map of distances.
 * =====================================================================================
 */

void transformToDistMap(std::unordered_map<int,float> & distances, const std::vector<Triangle> & triangles,
		const Camera & camera) {
	distances.clear() ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		distances[triangles[i].getID()] = calcDistance(triangles[i],camera) ;
	}
}		/* -----  end of function transfor  ----- */

}
