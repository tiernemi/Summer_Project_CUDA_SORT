
#ifndef bitonic_SORT_HPP_J60FNTL1
#define bitonic_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  bitonic_sort.hpp
 *
 *    Description:  Sort object for bitonic sort
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
 *         Name:  BitonicSort
 *  Description:  Implementaion of bitonic sort. Uses generic sort interface.
 * =====================================================================================
 */

namespace CPUSorts {

class BitonicSort : public Sort {
 public:
	BitonicSort() : Sort("Bitonic_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class bitonic_sort  ----- */

}

#endif /* end of include guard: bitonic_SORT_HPP_J60FNTL1 */
