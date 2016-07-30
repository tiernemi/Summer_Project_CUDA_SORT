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
#include "../../inc/cpp_inc/clock.hpp"

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
	std::vector<Triangle> temp = triangles ;

	GPUSorts::MerrelRadixGPUSort merSorter ;
	//GPUSorts::ThrustGPUSort tSorter ;

	Clock myClock ;
	myClock.start() ;
	merSorter.sortTriangles(temp,cameras[0]) ; 
	myClock.stop() ;
	std::cout << myClock.getDuration() << std::endl;

	std::vector<std::pair<int,float>> dists(temp.size()) ;
	Transforms::transformToDistVec(dists,temp,cameras[0]) ;
	/*  
	for (int i = 0 ; i < temp.size() ; ++i) {
		std::cout << dists[i].second << std::endl;
	} */

	temp = triangles ;

	

	return EXIT_SUCCESS ;
}


