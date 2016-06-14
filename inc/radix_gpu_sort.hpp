#ifndef RADIX_SORT_HPP_WTWS6ZYK
#define RADIX_SORT_HPP_WTWS6ZYK

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

#include <unordered_map>

void cudaRadixSort(std::unordered_map<int,float> & data) ;

#endif /* end of include guard: RADIX_SORT_HPP_WTWS6ZYK */
