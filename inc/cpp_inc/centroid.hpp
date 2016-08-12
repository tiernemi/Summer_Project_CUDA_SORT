#ifndef CENTROID_HPP_WKAZPM6H
#define CENTROID_HPP_WKAZPM6H

/*
 * =====================================================================================
 *
 *       Filename:  centroid.hpp
 *
 *    Description:  Header file for centroid class.
 *
 *        Version:  1.0
 *        Created:  07/06/16 18:20:28
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Centroid
 *       Fields:  id - Id of centroid
 *                coords - Co-ordinates of centroid.
 *  Description:  Storage class for centroid data. Can be translated to camera space.
 * =====================================================================================
 */

class Centroid {
 public:
	Centroid() { ; } ;
	Centroid(const float & x, const float & y, const float & z, int id) ;
	Centroid(const Centroid & obj) ;
	virtual ~Centroid() ;
	const float * getCoords() const {return coords ;} ;
	unsigned int getID() const {return id ;} ;
	void setCoords(const float & x, const float & y, const float & z) ;
	void setID(const unsigned int id) ;
 private:
	unsigned int id ;
	float coords[3] ;
} ;		/* -----  end of class Centroid  ----- */

#endif /* end of include guard: CENTROID_HPP_WKAZPM6H */
