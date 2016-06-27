#ifndef RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL
#define RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.hpp
 *
 *    Description:  Header file for linking cuda radix sort with cpu code.
 *
 *        Version:  1.0
 *        Created:  07/06/16 17:22:08
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "triangle.hpp"
#include "camera.hpp"

void cudaRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, std::vector<float> & sortTimes) ;

#endif /* end of include guard: RADIX_GPU_SORT_FUNCS_HPP_T02OEQEL */
