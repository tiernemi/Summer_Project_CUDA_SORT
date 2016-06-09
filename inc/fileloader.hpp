#ifndef FILELOADER_HPP_NTWPG8VR
#define FILELOADER_HPP_NTWPG8VR

/*
 * =====================================================================================
 *
 *       Filename:  fileloader.hpp
 *
 *    Description:  Header file for file loader class.
 *
 *        Version:  1.0
 *        Created:  08/06/16 11:46:29
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <string>
#include <fstream>

// Custom Headers //
#include "triangle.hpp"
#include "camera.hpp"

/* 
 * ===  CLASS  =========================================================================
 *         Name:  FileLoader
 *  Description:  FileLoader helper class.
 * =====================================================================================
 */

class FileLoader {
 public:
	static void loadFile(std::vector<Triangle> & data, std::vector<Camera> & cameraPos, const std::string & filename) ;
 private:
	static void loadTriangles(std::vector<Triangle> & data, std::ifstream & input) ;
	static void loadCameraPos(std::vector<Camera> & cameraPos, std::ifstream & input) ;
} ;		/* -----  end of class FileLoader  ----- */

#endif /* end of include guard: FILELOADER_HPP_NTWPG8VR */
