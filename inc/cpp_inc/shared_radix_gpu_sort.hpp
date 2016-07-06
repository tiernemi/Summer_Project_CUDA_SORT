#ifndef SHARED_RADIX_GPU_SORT_HPP_CPZHDQ4I
#define SHARED_RADIX_GPU_SORT_HPP_CPZHDQ4I

/*
 * =====================================================================================
 *
 *       Filename:  shared_radix_gpu_sort.hpp
 *
 *    Description:  Sort object for radix sort on the gpu. Uses shared memory for the
 *                  prefix sum step.
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
 *         Name:  SharedRadixGPUSort
 *  Description:  Implementaion of radix on the gpu sort.
 * =====================================================================================
 */

namespace GPUSorts {

class SharedRadixGPUSort : public Sort {
 public:
	SharedRadixGPUSort() : Sort("Shared_Radix_GPU_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class radix_sort  ----- */

}

#endif /* end of include guard: radix_GPU_SORT_HPP_CPZHDQ4I */

