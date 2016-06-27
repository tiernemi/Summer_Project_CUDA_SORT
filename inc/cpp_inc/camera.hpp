#ifndef CAMERA_HPP_EX9TMDYR
#define CAMERA_HPP_EX9TMDYR

/*
 * =====================================================================================
 *
 *       Filename:  camera.hpp
 *
 *    Description:  Class for camera positions.
 *
 *        Version:  1.0
 *        Created:  08/06/16 14:21:20
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Camera
 *       Fields:  coords - Co-ordinates of camera.
 *  Description:  Storage class for Camera data.
 * =====================================================================================
 */

class Camera {
 public:
	Camera() { ; } ;
	Camera(const float & x, const float & y, const float & z) ;
	Camera(const Camera & obj) ;
	virtual ~Camera() ;
	const float * getCoords() const {return coords ;} ;
	void setCoords(const float & x, const float & y, const float & z) ;
 private:
	float coords[3] ;
} ;		/* -----  end of class Camera  ----- */

#endif /* end of include guard: CAMERA_HPP_EX9TMDYR */
