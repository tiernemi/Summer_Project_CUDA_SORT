/*
 * =====================================================================================
 *
 *       Filename:  camera.cpp
 *
 *    Description:  Source file for camera storage class.
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

#include "../../inc/camera.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : Camera  ======================================
 *         Name:  Camera
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *  Description:  Constructor for camera object.
 * =====================================================================================
 */

Camera::Camera(const float & x, const float & y, const float & z) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
}		/* -----  end of member function Camera  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Camera  ======================================
 *         Name:  Camera
 *    Arguments:  const Camera & tri - Camera to copy.
 *  Description:  Copy constructor for camera.
 * =====================================================================================
 */

Camera::Camera(const Camera & obj) {
	const float * objco = obj.getCoords() ;
	coords[0] = objco[0] ; 
	coords[1] = objco[1] ; 
	coords[2] = objco[2] ; 
}		/* -----  end of member function Camera  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Camera  ======================================
 *         Name:  setCoords
 *    Arguments:  const float & x - x-co-ord ;
 *                const float & y - y-co-ord ;
 *                const float & z - z-co-ord ;
 *  Description:  Sets co-ords of camera.
 * =====================================================================================
 */

void Camera::setCoords(const float & x, const float & y, const float & z) {
	coords[0] = x ;
	coords[1] = y ;
	coords[2] = z ;
}		/* -----  end of member function setCoords  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Camera  ======================================
 *         Name: ~Camera 
 *  Description: Virtual destructor for camera object.
 * =====================================================================================
 */

Camera::~Camera() {
	
}		/* -----  end of member function   ----- */
