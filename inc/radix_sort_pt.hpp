
#ifndef radix_SORT_HPP_J60FNTL1
#define radix_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  radix_sort_pt.hpp
 *
 *    Description:  Sort object for radix cpu_sort
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
 *         Name:  RadixSortPT
 *  Description:  Implementaion of radix cpu_sort. Uses generic cpu_sort interface. Uses 8 bit
 *                histogram suggested by Pierre Terdiman (codercorner).
 * =====================================================================================
 */

namespace CPUSorts {

class RadixSortPT : public CpuSort {
 public:
	RadixSortPT() : CpuSort("Radix_SortPT") { ; } ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & cpu_sortTime) ;
} ;		/* -----  end of class radix_cpu_sort  ----- */

}

#endif /* end of include guard: radix_SORT_HPP_J60FNTL1 */
