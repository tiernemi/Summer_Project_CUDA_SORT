/*
 * =====================================================================================
 *
 *       Filename:  triangle.cpp
 *
 *    Description:  Source file for triangle storage class.
 *
 *        Version:  1.0
 *        Created:  08/06/16 11:17:27
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "../../inc/triangle.hpp"
#include "../../inc/camera.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : Triangle  ======================================
 *         Name:  Triangle
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *                const int id - ID of triangle ;
 *  Description:  Constructor for triangle object.
 * =====================================================================================
 */

Triangle::Triangle(const float & x, const float & y, const float & z, const int id) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
	this->id = id ;
}		/* -----  end of member function Triangle  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Triangle  ======================================
 *         Name:  Triangle
 *    Arguments:  const Triangle & tri - Triangle to copy.
 *  Description:  Copy constructor for truangle.
 * =====================================================================================
 */

Triangle::Triangle(const Triangle & obj) {
	const float * objco = obj.getCoords() ;
	coords[0] = objco[0] ; 
	coords[1] = objco[1] ; 
	coords[2] = objco[2] ; 
}		/* -----  end of member function Triangle  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : triangle  ======================================
 *         Name:  translate
 *    Arguments:  const std::vector<float> & camera - Camera co-ordinates.
 *  Description:  Translates the triangle to the camera space.
 * =====================================================================================
 */

void Triangle::translate(const Camera & camera) {
	const float * cameraco = camera.getCoords() ;
	coords[0] = coords[0] - cameraco[0] ;
	coords[1] = coords[1] - cameraco[1] ;
	coords[2] = coords[2] - cameraco[2] ;
}

/* 
 * ===  MEMBER FUNCTION CLASS : Triangle  ======================================
 *         Name:  setCoords
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *  Description:  Sets co-ords of triangle.
 * =====================================================================================
 */

void Triangle::setCoords(const float & x, const float & y, const float & z) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
}		/* -----  end of member function setCoords  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS :   ======================================
 *         Name:  setID
 *    Arguments:  const unsigned int id - ID of triangle/
 *  Description:  Sets ID of triangle.
 * =====================================================================================
 */

void Triangle::setID(const unsigned int id) {
	this->id = id ;
}		/* -----  end of member function   ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Triangle  ======================================
 *         Name: ~Triangle 
 *  Description: Virtual destructor for triangle object.
 * =====================================================================================
 */

Triangle::~Triangle() {
	
}		/* -----  end of member function   ----- */
