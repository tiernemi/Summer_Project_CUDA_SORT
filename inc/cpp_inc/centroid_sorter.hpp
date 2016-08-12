#ifndef CENTROID_SORTER_HPP_C7KQHQXR
#define CENTROID_SORTER_HPP_C7KQHQXR

/*
 * =====================================================================================
 *
 *       Filename:  centroid_sorter.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/08/16 10:36:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *   Organization:  
 *
 * =====================================================================================
 */

#include "centroid.hpp"
#include "camera.hpp"
#include "pt_sort_policy.hpp"
#include "mherf_sort_policy.hpp"
#include "thrust_sort_policy.hpp"
#include "stl_sort_policy.hpp"
#include "cub_sort_policy.hpp"
#include "impl_radix_sort_policy.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  CentroidSorter
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

template <class Sorter>
class CentroidSorter : private Sorter {
	using Sorter::allocate ;
	using Sorter::sort ;
	using Sorter::benchSort ;
	using Sorter::deAllocate ;
 public:
	CentroidSorter(const std::vector<Centroid> & centrocentroidIDs) ;
	std::vector<int> sort(const Camera & camera) ;
	std::vector<int> benchSort(const Camera & camera, std::vector<float> & times) ;
	virtual ~CentroidSorter() ;
 private:
	float * centroidPos ;
	int * centroidIDs ;
	int numElements ;
	/* data */
} ;		/* -----  end of class CentroidSorter  ----- */


template <class Sorter>
CentroidSorter<Sorter>::CentroidSorter(const std::vector<Centroid> & centrocentroidIDs) {
	numElements = centrocentroidIDs.size() ;
	std::pair<float*,int*> dataPtrs = allocate(centrocentroidIDs) ;
	centroidPos = dataPtrs.first ;
	centroidIDs = dataPtrs.second ;
}

template <class Sorter>
std::vector<int> CentroidSorter<Sorter>::sort(const Camera & camera) {
	std::vector<int> centroidIDsVec(numElements) ;
	sort(camera,centroidIDsVec,centroidIDs,centroidPos) ;
	return centroidIDsVec ;
}

template <class Sorter>
std::vector<int> CentroidSorter<Sorter>::benchSort(const Camera & camera, std::vector<float> & times) {
	std::vector<int> centroidIDsVec(numElements) ;
	benchSort(camera,centroidIDsVec,centroidIDs,centroidPos, times) ;
	return centroidIDsVec ;
}


template <class Sorter>
CentroidSorter<Sorter>::~CentroidSorter() {
	deAllocate(centroidPos, centroidIDs) ;
}




#endif /* end of include guard: CENTROID_SORTER_HPP_C7KQHQXR */
