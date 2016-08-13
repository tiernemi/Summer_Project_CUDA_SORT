#ifndef TRANSFORMS_HPP_M98NPKHR
#define TRANSFORMS_HPP_M98NPKHR

/*
 * =====================================================================================
 *
 *       Filename:  transforms.hpp
 *
 *    Description:  Header file for useful transform functions.
 *
 *        Version:  1.0
 *        Created:  09/06/16 10:27:12
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "centroid.hpp"

namespace Transforms {
	void transformToDistVec(std::vector<std::pair<int,float> > & distances, 
			const std::vector<Centroid> & centroids, const Camera & camera) ;
	void transformToDistArray(float * dists, float * coords, const Camera & camera, const int numElements) ;
}

#endif /* end of include guard: TRANSFORMS_HPP_M98NPKHR */
