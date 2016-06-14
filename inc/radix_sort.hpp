
#ifndef radix_SORT_HPP_J60FNTL1
#define radix_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.hpp
 *
 *    Description:  Sort object for radix sort
 *
 *        Version:  1.0
 *        Created:  2016-06-13 13:50
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
 *         Name:  RadixSort
 *  Description:  Implementaion of radix sort. Uses generic sort interface.
 * =====================================================================================
 */

namespace CPUSorts {

class RadixSort : public Sort {
 public:
	RadixSort() : Sort("Radix_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class radix_sort  ----- */

}

#endif /* end of include guard: radix_SORT_HPP_J60FNTL1 */
