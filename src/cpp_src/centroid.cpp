/*
 * =====================================================================================
 *
 *       Filename:  centroid.cpp
 *
 *    Description:  Source file for centroid storage class.
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

#include "../../inc/cpp_inc/centroid.hpp"
#include "../../inc/cpp_inc/camera.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : Centroid  ======================================
 *         Name:  Centroid
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *                const int id - ID of centroid ;
 *  Description:  Constructor for centroid object.
 * =====================================================================================
 */

Centroid::Centroid(const float & x, const float & y, const float & z, const int id) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
	this->id = id ;
}		/* -----  end of member function Centroid  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Centroid  ======================================
 *         Name:  Centroid
 *    Arguments:  const Centroid & tri - Centroid to copy.
 *  Description:  Copy constructor for truangle.
 * =====================================================================================
 */

Centroid::Centroid(const Centroid & obj) {
	const float * objco = obj.getCoords() ;
	coords[0] = objco[0] ; 
	coords[1] = objco[1] ; 
	coords[2] = objco[2] ; 
	id = obj.id ;
}		/* -----  end of member function Centroid  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Centroid  ======================================
 *         Name:  setCoords
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *  Description:  Sets co-ords of centroid.
 * =====================================================================================
 */

void Centroid::setCoords(const float & x, const float & y, const float & z) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
}		/* -----  end of member function setCoords  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS :   ======================================
 *         Name:  setID
 *    Arguments:  const unsigned int id - ID of centroid/
 *  Description:  Sets ID of centroid.
 * =====================================================================================
 */

void Centroid::setID(const unsigned int id) {
	this->id = id ;
}		/* -----  end of member function   ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Centroid  ======================================
 *         Name: ~Centroid 
 *  Description: Virtual destructor for centroid object.
 * =====================================================================================
 */

Centroid::~Centroid() {
	
}		/* -----  end of member function   ----- */
