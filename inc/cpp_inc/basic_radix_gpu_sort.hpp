#ifndef BASIC_RADIX_GPU_SORT_HPP_CPZHDQ4I
#define BASIC_RADIX_GPU_SORT_HPP_CPZHDQ4I

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.hpp
 *
 *    Description:  Sort object for radix sort on the gpu.
 *
 *        Version:  1.0
 *        Created:  2016-06-27 11:10
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
 *         Name:  RadixGPUSort
 *  Description:  Implementaion of radix on the gpu sort.
 * =====================================================================================
 */

namespace GPUSorts {

class BasicRadixGPUSort : public Sort {
 public:
	BasicRadixGPUSort() : Sort("Basic_Radix_GPU_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class radix_sort  ----- */

}

#endif /* end of include guard: radix_GPU_SORT_HPP_CPZHDQ4I */

