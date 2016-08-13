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
#include "../../inc/cpp_inc/clock.hpp"
#include "../../inc/cpp_inc/centroid_sorter.hpp"

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
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	FileLoader::loadFile(centroids,cameras,filename.getValue()) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;
	Clock myClock ;

	CentroidSorter<ImplRadixSort> implSorter(centroids) ;
	ids = implSorter.sort(cameras[0]) ;
	return EXIT_SUCCESS ;
}


