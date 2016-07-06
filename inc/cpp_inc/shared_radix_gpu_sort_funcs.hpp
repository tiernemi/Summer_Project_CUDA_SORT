#ifndef SHARED_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL
#define SHARED_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  shared_radix_sort.hpp
 *
 *    Description:  Header file for linking cuda shared radix sort with cpu code.
 *
 *        Version:  1.0
 *        Created:  2016-07-05 14:34
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "triangle.hpp"
#include "camera.hpp"

void cudaSharedRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaSharedRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & sortTimes) ;

#endif /* end of include guard: SHARED_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL */
