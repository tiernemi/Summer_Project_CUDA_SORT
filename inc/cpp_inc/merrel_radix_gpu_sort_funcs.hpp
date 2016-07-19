#ifndef MERREL_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL
#define MERREL_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  merrel_radix_sort.hpp
 *
 *    Description:  Header file for linking cuda merrel radix sort with cpu code.
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

void cudaMerrelRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaMerrelRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & sortTimes) ;

#endif /* end of include guard: MERREL_RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL */
