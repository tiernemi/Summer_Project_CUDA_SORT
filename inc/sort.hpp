#ifndef SORT_HPP_P3SBVJOK
#define SORT_HPP_P3SBVJOK

/*
 * =====================================================================================
 *
 *       Filename:  sort.hpp
 *
 *    Description:  Sort abstract base class.
 *
 *        Version:  1.0
 *        Created:  13/06/16 09:41:13
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <string>
#include <vector>

// Custom Headers //
#include "triangle.hpp"
#include "camera.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Sort
 *       Fields:  const std::string algName - The algorithm name.
 *  Description:  Abstract based class for all sorts. This interface makes benchmarking
 *                and testing easier.
 * =====================================================================================
 */

class Sort {
 public:
	Sort(std::string algName) : algName{algName} { ; } ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera) = 0 ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances) = 0 ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, Camera & camera, float & sortTime) = 0 ;
	virtual void sortDistances(std::vector<std::pair<int,float>> & distances, float & sortTime) = 0 ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) = 0 ;
	virtual void sortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
			std::vector<float> & times) = 0 ;
	std::string getAlgName() const { return algName ; } ;
 private:
	const std::string algName ;
} ;		/* -----  end of class Sort  ----- */

#endif /* end of include guard: SORT_HPP_P3SBVJOK */
