
#ifndef bitonic_SORT_HPP_J60FNTL1
#define bitonic_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  bitonic_sort.hpp
 *
 *    Description:  Sort object for bitonic sort
 *
 *        Version:  1.0
 *        Created:  2016-06-13 13:50
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Custom Headers //
#include "cpu_sort.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  BitonicSort
 *  Description:  Implementaion of bitonic sort. Uses generic pu_rt interface.
 * =====================================================================================
 */

namespace CPUSorts {

class BitonicSort : public CpuSort {
 public:
	BitonicSort() : CpuSort("Bitonic_Sort") { ; } ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
} ;		/* -----  end of class bitonic_sort  ----- */

}

#endif /* end of include guard: bitonic_SORT_HPP_J60FNTL1 */
