#ifndef BITONIC_GPU_SORT_FUNCS_HPP_TKJYSMNX
#define BITONIC_GPU_SORT_FUNCS_HPP_TKJYSMNX

/*
 * =====================================================================================
 *
 *       Filename:  thrust_gpu_sort_funcs.hpp
 *
 *    Description:  Header containing gpu code.
 *
 *        Version:  1.0
 *        Created:  16/06/16 14:52:48
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "triangle.hpp"
#include "camera.hpp"

void cudaBitonicSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaBitonicSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, std::vector<float> & times) ;

#endif /* end of include guard:  */
