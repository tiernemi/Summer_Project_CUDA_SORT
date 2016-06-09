/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Main function for gpu-sort / cpu-sort comparison
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:33:20
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <iostream>
#include <unordered_map>

// Custom headers. //
#include "../../inc/test_funcs.hpp"
#include "../../inc/fileloader.hpp"
#include "../../inc/cmd_parser.hpp"
#include "../../inc/c_variable.hpp"
#include "../../inc/c_flag.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/camera.hpp"
#include "../../inc/cpu_sorts.hpp"
#include "../../inc/test_funcs.hpp"

// Options //
CFlag verbose("verbose",false,"v") ;
CFlag help("help",false,"h") ;
CVariable<std::string> filename("filename","","f:") ;

int main(int argc, char *argv[]) {

	// Command Line Options //
	std::vector<CommandLineOption*> options ;
	options.push_back(&verbose) ;
	options.push_back(&help) ;
	options.push_back(&filename) ;

	// Process Command Line Options //
	CommandParser::processArgs(argc, argv, options) ;

	// Read in the data //
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	FileLoader::loadFile(triangles,cameras,filename.getValue()) ;

	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances ;
	Transforms::transformToDistVec(distances, triangles, cameras[0]) ;

	return EXIT_SUCCESS ;
}


