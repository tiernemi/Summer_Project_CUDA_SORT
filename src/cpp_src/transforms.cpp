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
#include "../../inc/cpp_inc/centroid.hpp"
#include "../../inc/cpp_inc/camera.hpp"
#include "../../inc/cpp_inc/transforms.hpp"

namespace Transforms {

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcDistance
 *    Arguments:  const Centroid & centroid - Centroid (Model space)
 *                const Camera & camera - Camera (Model space)
 *      Returns:  Distance between camera and centroid.
 *  Description:  Helper function for calculating distance between camera and centroid.
 * =====================================================================================
 */

static float calcDistance(const Centroid & centroid, const Camera & camera) {
	const float * trico = centroid.getCoords() ;
	const float * camco = camera.getCoords() ;	
	return std::sqrt(std::pow(trico[0]-camco[0],2) + std::pow(trico[1]-camco[1],2) +
			std::pow(trico[2]-camco[2],2)) ;
}		/* -----  end of function   ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcDistance
 *    Arguments:  const float * coords - Co-ordinates of centroid.
 *                const Camera & camera - Camera.
 *      Returns:  Distance between camera and raw centroid co-ordinates.
 *  Description:  Helper function for calculating distance between camera and centroid.
 * =====================================================================================
 */

static float calcDistance(const float * cenco, const Camera & camera) {
	const float * camco = camera.getCoords() ;	
	return std::sqrt(std::pow(cenco[0]-camco[0],2) + std::pow(cenco[1]-camco[1],2) +
			std::pow(cenco[2]-camco[2],2)) ;
}		/* -----  end of function   ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transformToDistVec
 *    Arguments:  std::vector<std::pair<int,float>> & distances - Array of distances to
 *                be overwritten.
 *                const std::vector<Centroid> & centroids - Array of truangles in model
 *                space.
 *                const Camera & camera - Camera for this frame.
 *  Description:  Transforms the centroid - camera system to a sortable array of distances.
 * =====================================================================================
 */

void transformToDistVec(std::vector<std::pair<int,float>> & distances, const std::vector<Centroid> & centroids,
		const Camera & camera) {
	for (unsigned int i = 0 ; i < centroids.size() ; ++i) {
		distances[i].first = i ; 
		distances[i].second = calcDistance(centroids[i],camera) ;
	}
}		/* -----  end of function transfor  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transformToDistMap
 *    Arguments:  std::unordered_map<int,float> & distances- Map of distances to
 *                be overwritten.
 *                const std::vector<Centroid> & centroids - Array of truangles in model
 *                space.
 *                const Camera & camera - Camera for this frame.
 *  Description:  Transforms the centroid - camera system to a sortable map of distances.
 * =====================================================================================
 */

void transformToDistMap(std::unordered_map<int,float> & distances, const std::vector<Centroid> & centroids,
		const Camera & camera) {
	distances.clear() ;
	for (unsigned int i = 0 ; i < centroids.size() ; ++i) {
		distances[centroids[i].getID()] = calcDistance(centroids[i],camera) ;
	}
}		/* -----  end of function transfor  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transforms
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void transformToDistArray(float * dists, float * coords, const Camera & camera, const int numElements) {
	for (int i = 0 ; i < numElements ; ++i) {
		dists[i] = calcDistance(&coords[3*i],camera) ;
	}
}

}

