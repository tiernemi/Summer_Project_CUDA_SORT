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
#include "../../inc/cpp_inc/test_funcs.hpp"
#include "../../inc/cpp_inc/fileloader.hpp"
#include "../../inc/cpp_inc/cmd_parser.hpp"
#include "../../inc/cpp_inc/c_variable.hpp"
#include "../../inc/cpp_inc/c_flag.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/camera.hpp"
#include "../../inc/cpp_inc/test_funcs.hpp"
#include "../../inc/cpp_inc/sort_algs.hpp"

// Options //
CFlag verbose("verbose",false,"v") ;
CFlag help("help",false,"h") ;
CVariable<std::string> filename("filename","","f:") ;

int main(int argc, char *argv[]) {

	std::mt19937 gen ;
	gen.seed(872712872412) ;

	// Command Line Options //
	std::vector<CommandLineOption*> options ;
	options.push_back(&verbose) ;
	options.push_back(&help) ;
	options.push_back(&filename) ;

	// Process Command Line Options //
	CommandParser::processArgs(argc, argv, options) ;
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	FileLoader::loadFile(triangles,cameras,filename.getValue()) ;

	//GPUSorts::ThrustGPUSort thrustSorter ;
//	thrustSorter.sortTriangles(triangles,cameras) ;
//	std::vector<std::pair<int,float>> distances(triangles.size()) ;
//	Transforms::transformToDistVec(distances,triangles,cameras[0]) ;

	std::vector<Triangle> temp = triangles ;
	Tests::makePercentSorted(triangles,cameras[0],1,gen) ;

	std::cout << Tests::calcPercentSorted(triangles,cameras[0]) << std::endl ;

	return EXIT_SUCCESS ;
}


