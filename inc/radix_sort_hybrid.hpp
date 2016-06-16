
#ifndef radix_SORT_HYB_HPP11_J60FNTL1
#define radix_SORT_HYB_HPP11_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_hybrid.hpp
 *
 *    Description:  Sort object for radix sort. Hybrid of Terdiman's and Hoff's
 *                  algorithm.
 *
 *        Version:  1.0
 *        Created:  2016-06-16 11:04
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
 *         Name:  RadixSortHybrid
 *  Description:  Implementaion of radix sort using optimisations used by Terdiman and
 *                Herf.
 * =====================================================================================
 */

namespace CPUSorts {

class RadixSortHybrid : public Sort {
 public:
	RadixSortHybrid() : Sort("Radix_SortHybrid") { ; } ;
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
