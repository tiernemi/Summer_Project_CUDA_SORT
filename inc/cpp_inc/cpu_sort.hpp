#ifndef CPU_SORT_HPP_K1QUMADP
#define CPU_SORT_HPP_K1QUMADP

/*
 * =====================================================================================
 *
 *       Filename:  cpu_sort.hpp
 *
 *    Description:  Header file for cpu sorts.
 *
 *        Version:  1.0
 *        Created:  20/06/16 17:01:47
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
 *         Name:  CpuSort
 *  Description:  Interface for cpu sorts.
 * =====================================================================================
 */

namespace CPUSorts {

class CpuSort : public Sort {
 public:
	CpuSort(std::string algName) : Sort(algName) { ; } ; 
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, std::vector<float> & sortTime) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
 		std::vector<float> & times) ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) = 0 ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) = 0 ;
} ;		/* -----  end of class CpuSort  ----- */

}

#endif /* end of include guard: CPU_SORT_HPP_K1QUMADP */
