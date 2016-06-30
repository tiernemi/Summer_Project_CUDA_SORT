
#ifndef CUB_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL
#define CUB_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.hpp
 *
 *    Description:  Header file for linking cub cuda radix sort with cpu code.
 *
 *        Version:  1.0
 *        Created:  2016-06-30 10:54
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "triangle.hpp"
#include "camera.hpp"

void cudaCubRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaCubRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & sortTimes) ;

#endif /* end of include guard: CUB_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL */
