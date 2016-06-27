
#ifndef bubble_SORT_HPP_J60FNTL1
#define bubble_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  bubble_sort.hpp
 *
 *    Description:  Sort object for bubble sort
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
 *         Name:  BubbleSort
 *  Description:  Implementaion of bubble sort. Uses generic cpu_sort interface.
 * =====================================================================================
 */

namespace CPUSorts {

class BubbleSort : public CpuSort {
 public:
	BubbleSort() : CpuSort("Bubble_Sort") { ; } ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
} ;		/* -----  end of class bubble_sort  ----- */

}

#endif /* end of include guard: bubble_SORT_HPP_J60FNTL1 */
