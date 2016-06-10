/*
 * =====================================================================================
 *
 *       Filename:  benchmark.cpp
 *
 *    Description:  Code used to benchmarking sorts.
 *
 *        Version:  1.0
 *        Created:  09/06/16 16:15:20
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <libgen.h>
#include <string.h>
#include <fstream>
#include "../inc/test_funcs.hpp"
#include "../inc/fileloader.hpp"
#include "../inc/cmd_parser.hpp"
#include "../inc/c_variable.hpp"
#include "../inc/c_flag.hpp"
#include "../inc/transforms.hpp"
#include "../inc/camera.hpp"
#include "../inc/cpu_sorts.hpp"
#include "../inc/clock.hpp"
#include "../inc/test_funcs.hpp"

void runCPUBenchs(std::vector<void (*)(std::vector<std::pair<int,float>> & dist)> & algs,
		std::vector<Triangle> & triangles, std::vector<Camera> & camera, 
		std::vector<std::vector<float>> & times) ;
void outputTimes(std::vector<std::vector<std::vector<float>>> & times, std::vector<std::string> & names,
		std::vector<unsigned int> & numElements, std::vector<std::string> filenames) ;

int main(int argc, char *argv[]) {

	// Variables storing benchmark data. //
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	std::vector<std::vector<std::vector<float>>> times ;
	std::vector<void (*)(std::vector<std::pair<int,float>> & dist)> algs ;
	std::vector<std::string> names ;
	std::vector<std::string> filenames ;
	std::vector<unsigned int> numElements ;

	// Add sorts. //
	names.push_back("STLSort") ;
	algs.push_back(CPUSorts::cpuSTLSort) ;

	// Read in file names. //
	for (int i = 1 ; i < argc ; ++i) {
		filenames.push_back(argv[i]) ;
	}
	numElements.resize(filenames.size()) ;
	times.resize(filenames.size()) ;

	// For each filename, load and run benchmarks. //
	for (unsigned int i = 0 ; i < filenames.size() ; ++i) {
		FileLoader::loadFile(triangles,cameras,filenames[i]) ;
		numElements[i] = triangles.size() ;
		// Run CPU sorts for this data set. //
		runCPUBenchs(algs,triangles,cameras,times[i]) ;
	}

	// Output time data. //
	outputTimes(times, names, numElements, filenames) ;

	return EXIT_SUCCESS ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  runBenchs
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void runCPUBenchs(std::vector<void (*)(std::vector<std::pair<int,float>> & )> & algs, std::vector<Triangle> & triangles
		, std::vector<Camera> & cameras, std::vector<std::vector<float>> & times) {

	Clock clock ;
	std::vector<Triangle> sortingTriVec = triangles ;
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < algs.size() ; ++i) {
		std::vector<float> newTimes ;
		for (unsigned int j = 0 ; j < cameras.size() ; ++j) {
			// Convert to sortable form //
			std::vector<std::pair<int,float>> distances ;
			Transforms::transformToDistVec(distances, sortingTriVec, cameras[j]) ;
			// Reorder triangles. //
			for (unsigned int k = 0 ; k < distances.size() ; ++k) {
				temp[k] = sortingTriVec[distances[k].first] ;
			}
			sortingTriVec = temp ;
			clock.start() ;
			algs[i](distances) ;
			clock.stop() ;
			newTimes.push_back(clock.getDuration()) ;
		}
		times.push_back(newTimes) ;
	}
}		/* -----  end of function runBenchs  ----- */

/* 
 ======================================================================================
 *         Name:  
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void outputTimes(std::vector<std::vector<std::vector<float>>> & times, std::vector<std::string> & names, 
		std::vector<unsigned int> & numElements, std::vector<std::string> filenames) {
	for (unsigned int i = 0 ; i < numElements.size() ; ++i) {
		for (unsigned int j = 0 ; j < times[i].size() ; ++j) {	
			char base[1000] ;
			strcpy(base,filenames[i].c_str()) ;
			std::string datFileName = "./bench_data/times" + names[j] + 
				std::to_string(numElements[i]) + basename(base) ;
			std::ofstream output(datFileName) ;
			for (unsigned int k = 0 ; k < times[i][j].size() ; ++k) {
				output << numElements[i] << " " << times[i][j][k] << " " <<  numElements[i]/(times[i][j][k]*1E6) 
					<< std::endl ;
			}
			output.close() ;
		}
	}
}		/* -----  end of function   ----- */
