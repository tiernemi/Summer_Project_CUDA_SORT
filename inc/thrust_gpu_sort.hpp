
#ifndef thrust_SORT_HPP11_J60FNTL1
#define thrust_SORT_HPP11_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  thrust_sort.hpp
 *
 *    Description:  Sort object for thrust sort
 *
 *        Version:  1.0
 *        Created:  2016-06-16 12:49
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Custom Headers //
#include "sort.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  ThrustGPUSort
 *  Description:  Implementaion of thrust sort.
 * =====================================================================================
 */

namespace GPUSorts {

class ThrustGPUSort : public Sort {
 public:
	ThrustGPUSort() : Sort("Thrust_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class thrust_sort  ----- */

}

#endif /* end of include guard: thrust_SORT_HPP_J60FNTL1 */
