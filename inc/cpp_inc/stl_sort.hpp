#ifndef STL_SORT_HPP_J60FNTL1
#define STL_SORT_HPP_J60FNTL1

/*
 * =====================================================================================
 *
 *       Filename:  stl_sort.hpp
 *
 *    Description:  Sort object for stl sort
 *
 *        Version:  1.0
 *        Created:  13/06/16 09:47:56
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
 *         Name:  stl_sort
 *  Description:  Implementaion of stl sort. Uses generic sort interface.
 * =====================================================================================
 */

namespace CPUSorts {

class STLSort : public CpuSort {
 public:
	STLSort() : CpuSort("Stl_Sort") { ; } ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
} ;		/* -----  end of class stl_sort  ----- */

}

#endif /* end of include guard: STL_SORT_HPP_J60FNTL1 */
