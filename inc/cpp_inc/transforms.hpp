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
 *   Organization:  
 *
 * =====================================================================================
 */

#include "triangle.hpp"

namespace Transforms {
	void transformToDistVec(std::vector<std::pair<int,float> > & distances, 
			const std::vector<Triangle> & triangles, const Camera & camera) ;
//	void transformToDistMap(std::unordered_map<int,float> & distances, 
	//		const std::vector<Triangle> & triangles, const Camera & camera) ;
}

#endif /* end of include guard: TRANSFORMS_HPP_M98NPKHR */
