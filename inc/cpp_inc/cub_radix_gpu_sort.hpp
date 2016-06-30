#ifndef CUB_RADIX_GPU_SORT_HPP_YMQIXTVO
#define CUB_RADIX_GPU_SORT_HPP_YMQIXTVO

/*
 * =====================================================================================
 *
 *       Filename:  cub_radix_gpu_sort.hpp
 *
 *    Description:  An implementation of the CUB library radix sort.
 *
 *        Version:  1.0
 *        Created:  2016-06-30 10:51
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
 *         Name:  CubRadixGPUSort
 *  Description:  Implementaion of cub radix sort.
 * =====================================================================================
 */

namespace GPUSorts {

class CubRadixGPUSort : public Sort {
 public:
	CubRadixGPUSort() : Sort("Cub_Radix_GPU_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera)  ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & times) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras ,
			std::vector<float> & times) ;
} ;		/* -----  end of class radix_sort  ----- */

}

#endif /* end of include guard: CUB_RADIX_GPU_SORT_HPP_YMQIXTVO */
