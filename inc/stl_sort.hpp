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


#include "sort.hpp"


/* 
 * ===  CLASS  =========================================================================
 *         Name:  stl_sort
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

namespace CPUSorts {

class STLSort : public Sort {
 public:
	STLSort() : Sort("Stl_Sort") { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras)  ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, std::vector<float> & times) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) ;
} ;		/* -----  end of class stl_sort  ----- */

}

#endif /* end of include guard: STL_SORT_HPP_J60FNTL1 */
