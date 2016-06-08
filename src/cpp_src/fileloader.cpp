/*
 * =====================================================================================
 *
 *       Filename:  fileloader.cpp
 *
 *    Description:  FileLoader helper class used to load in triangle and camera data.
 *
 *        Version:  1.0
 *        Created:  08/06/16 12:00:11
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <exception>
#include <iostream>
#include <sstream>

// Custom Headers //
#include "../../inc/fileloader.hpp"
#include "../../inc/triangle.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : FileLoader  ===========================================
 *         Name:  loadFile
 *    Arguments:  std::vector<Triangle> & cameraPos - Array containing triangle data.
 *                std::vector<std::vector<float>> & cameraPos - Array containing camera
 *                positions.
 *                const std::string & filename - Name of the file.
 *  Description:  Loads the triangle and camera data from the file specified.
 * =====================================================================================
 */

void FileLoader::loadFile(std::vector<Triangle> & data, std::vector<Camera> & cameraPos, const std::string & filename){
	std::ifstream input(filename) ;
	if (!input.is_open()) {
		std::cerr << "Error! file does not exist" << std::endl;
		exit(-1) ;
	}
	// Extract header file. //
	unsigned int numPoints ;
	for (int i = 0 ; i < 4 ; ++i) {
		input.ignore(1E3, ' ') ;
	}
	input >> numPoints ;
	data.resize(numPoints) ;
	loadTriangles(data, input) ;
	loadCameraPos(cameraPos, input) ;
	input.close() ;
}

/* 
 * ===  MEMBER FUNCTION CLASS : fileloader  ==========================================
 *         Name:  loadTriangles
 *    Arguments:  std::vector<Triangle> & cameraPos - Array containing triangle data.
 *                std::ifstream & input - Input stream of file.
 *  Description:  Loads in triangle positions.
 * =====================================================================================
 */

void FileLoader::loadTriangles(std::vector<Triangle> & data, std::ifstream & input) {
	unsigned int id ;
	float coBuffer[3] ;
	for (unsigned int i = 0 ; i < data.size() ; ++i) {
		input.ignore(1E3, ' ') ;
		input >> id ;
		for (int i = 0 ; i < 3 ; ++i) {
			input >> coBuffer[i] ;
		}
		data[i].setCoords(coBuffer[0], coBuffer[1], coBuffer[2]) ;
		data[i].setID(id) ;
	}
}

/* 
 * ===  MEMBER FUNCTION CLASS : fileloader  ===========================================
 *         Name:  loadCameraPos
 *    Arguments:  std::vector<std::vector<float>> & cameraPos - Array containing camera
 *                positions.
 *                std::ifstream & input - Input stream of file.
 *  Description:  Loads in camera positions.
 * =====================================================================================
 */

void FileLoader::loadCameraPos(std::vector<Camera> & cameraPos, std::ifstream & input) {
	cameraPos.resize(0) ;
	std::vector<float> camBuffer(3) ;
	std::string test ;
	do {
		input.ignore(1E3, '\t') ;
		for (int i = 0 ; i < 3 ; ++i) {
			input >> camBuffer[i] ;
		}
		cameraPos.push_back(Camera(camBuffer[0], camBuffer[1], camBuffer[2])) ;
	} while (input);
}
