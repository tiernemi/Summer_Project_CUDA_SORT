#ifndef THRUST_GPU_SORT_FUNCS_HPP_TKJYSMNX
#define THRUST_GPU_SORT_FUNCS_HPP_TKJYSMNX

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

void cudaSortDistances(std::vector<int> & ids, std::vector<float> & dists) ;
void cudaSortDistances(std::vector<int> & ids, std::vector<float> & dists, float & sortTime) ;

#endif /* end of include guard:  */
