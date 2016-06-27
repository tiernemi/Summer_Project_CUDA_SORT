#ifndef bitonic_SORT_HPP11_J60FNTL1
#define bitonic_SORT_HPP11_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  bitonic_gpu_sort.hpp
 *
 *    Description:  Sort object for bitonic gpu sort.
 *
 *        Version:  1.0
 *        Created:  2016-06-16 12:49
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Custom Headers //
#include "sort.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  BitonicGPUSort
 *  Description:  Implementaion of bitonic sort ont he GPU. Unoptimised Global memory.
 * =====================================================================================
 */

namespace GPUSorts {

class BitonicGPUSort : public Sort {
 public:
	BitonicGPUSort() : Sort("Bitonic_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class bitonic_sort  ----- */

}

#endif /* end of include guard: bitonic_SORT_HPP_J60FNTL1 */
