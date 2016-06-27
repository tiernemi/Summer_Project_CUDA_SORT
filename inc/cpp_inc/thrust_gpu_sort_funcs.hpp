#ifndef THRUST_GPU_SORT_FUNCS_HPP_TKJYSMNX
#define THRUST_GPU_SORT_FUNCS_HPP_TKJYSMNX

/*
 * =====================================================================================
 *
 *       Filename:  thrust_gpu_sort_funcs.hpp
 *
 *    Description:  Header containing gpu code for thrust sort.
 *
 *        Version:  1.0
 *        Created:  16/06/16 14:52:48
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "triangle.hpp"
#include "camera.hpp"

void cudaThrustSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
void cudaThrustSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, std::vector<float> & times) ;

#endif /* end of include guard:  */
