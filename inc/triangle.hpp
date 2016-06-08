#ifndef TRIANGLE_HPP_WKAZPM6H
#define TRIANGLE_HPP_WKAZPM6H

/*
 * =====================================================================================
 *
 *       Filename:  triangle.hpp
 *
 *    Description:  Header file for triangle class.
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
#include "camera.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Triangle
 *       Fields:  id - Id of triangle
 *                coords - Co-ordinates of triangle.
 *  Description:  Storage class for triangle data. Can be translated to camera space.
 * =====================================================================================
 */

class Triangle {
 public:
	Triangle() { ; } ;
	Triangle(const float & x, const float & y, const float & z, int id) ;
	Triangle(const Triangle & obj) ;
	virtual ~Triangle() ;
	void translate(const Camera & camera) ;
	const float * getCoords() const {return coords ;} ;
	unsigned int getID() const {return id ;} ;
	void setCoords(const float & x, const float & y, const float & z) ;
	void setID(const unsigned int id) ;
 private:
	unsigned int id ;
	float coords[3] ;
} ;		/* -----  end of class Triangle  ----- */

#endif /* end of include guard: TRIANGLE_HPP_WKAZPM6H */
