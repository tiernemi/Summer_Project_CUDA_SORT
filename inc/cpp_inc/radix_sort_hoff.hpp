
#ifndef radix_SORT_HPP11_J60FNTL1
#define radix_SORT_HPP11_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_hoff.hpp
 *
 *    Description:  Sort object for radix sort
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
 *         Name:  RadixSortHoff
 *  Description:  Implementaion of radix sort using 11 bit histograms as suggested by
 *                Hoff (stereopsis). Uses generic sort interface.
 * =====================================================================================
 */

namespace CPUSorts {

class RadixSortHoff : public CpuSort {
 public:
	RadixSortHoff() : CpuSort("Radix_SortHoff") { ; } ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
} ;		/* -----  end of class radix_sort  ----- */

}

#endif /* end of include guard: radix_SORT_HPP_J60FNTL1 */
