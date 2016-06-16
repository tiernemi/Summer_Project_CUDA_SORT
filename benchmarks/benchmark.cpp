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
#include <random>
#include <string>
#include <unordered_map>
#include <libgen.h>
#include <string.h>
#include <fstream>

// Custom Headers //
#include "../inc/test_funcs.hpp"
#include "../inc/fileloader.hpp"
#include "../inc/cmd_parser.hpp"
#include "../inc/c_variable.hpp"
#include "../inc/c_flag.hpp"
#include "../inc/transforms.hpp"
#include "../inc/camera.hpp"
#include "../inc/clock.hpp"
#include "../inc/test_funcs.hpp"
#include "../inc/sort_algs.hpp"

void runCPUBenchs(std::vector<Sort*> & sorts,
		std::vector<Triangle> & triangles, std::vector<Camera> & camera, 
		std::vector<std::vector<float>> & times, std::vector<std::vector<float>> & percentSorts) ;
void outputResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & percentSorts) ;
float calcPercentSorted(const std::vector<Triangle> & triangles, const Camera & camera) ;

std::mt19937 gen ;

int main(int argc, char *argv[]) {

	gen.seed(199293939) ;

	// Variables storing benchmark data. //
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	std::vector<std::vector<std::vector<float>>> times ;
	std::vector<std::vector<std::vector<float>>> percentSorts ;
	std::vector<Sort*> sorts ;
	std::vector<std::string> names ;
	std::vector<std::string> filenames ;
	std::vector<unsigned int> numElements ;

	// Run time benchmarks. //
	CPUSorts::STLSort stlSorter ;
	CPUSorts::BitonicSort bitonicSorter ;
	CPUSorts::RadixSortPT radixSorterPT ;
	CPUSorts::RadixSortHoff radixSorterHoff ;
	// Add sorts. //
	sorts.push_back(&stlSorter) ;
	sorts.push_back(&bitonicSorter) ;
	sorts.push_back(&radixSorterPT) ;
	sorts.push_back(&radixSorterHoff) ;

	// Read in file names. //
	for (int i = 1 ; i < argc ; ++i) {
		filenames.push_back(argv[i]) ;
	}
	numElements.resize(filenames.size()) ;
	times.resize(filenames.size()) ;
	percentSorts.resize(filenames.size()) ;

	// For each filename, load and run benchmarks. //
	for (unsigned int i = 0 ; i < filenames.size() ; ++i) {
		FileLoader::loadFile(triangles,cameras,filenames[i]) ;
		numElements[i] = triangles.size() ;
		// Run CPU sorts for this data set. //
		runCPUBenchs(sorts,triangles,cameras,times[i],percentSorts[i]) ;
	}

	// Output time data. //
	outputResults(times, sorts, numElements, filenames,percentSorts) ;

	return EXIT_SUCCESS ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  runBenchs
 *    Arguments:  std::vector<Sort*> & sorts - Array of sort pointers.
 *                std::vector<Triangle> & triangles - Vector of triangle objects.
 *                std::vector<Camera> & cameras - Vector of cameras.
 *                std::vector<std::vector<float>> & times - Vector for timing data.
 *                std::vector<std::vector<float>> & percentSorts - Vector for percentage
 *                sorted.
 *  Description:  Runs timing and coherency benchmarks for each sorting algorithm.
 * =====================================================================================
 */

void runCPUBenchs(std::vector<Sort*> & sorts, std::vector<Triangle> & triangles
		, std::vector<Camera> & cameras, std::vector<std::vector<float>> & times
		, std::vector<std::vector<float>> & percentSorts) {

	std::vector<Triangle> tempTriangles = triangles ;
	for (unsigned int i = 0 ; i < sorts.size() ; ++i) {
		tempTriangles = triangles ;
		std::vector<float> newTimes ;
		std::vector<float> newSorts ;
		for (unsigned int j = 0 ; j < cameras.size() ; ++j) {
			float sortTime = 0 ;
			newSorts.push_back(Tests::calcPercentSorted(tempTriangles,cameras[j])) ;
			sorts[i]->sortTriangles(tempTriangles,cameras[j],sortTime) ;
			newTimes.push_back(sortTime) ;
		}
		std::vector<float> newSortPercents ;
		times.push_back(newTimes) ;
		percentSorts.push_back(newSorts) ;
	}
}		/* -----  end of function runBenchs  ----- */

/* 
 ======================================================================================
 *         Name:  outputResults
 *    Arguments:  std::vector<std::vector<std::vector<float>>> & times - All bench times.
 *                std::vector<Sort*> & sorts - Array of sorting algorithms.
 *		          std::vector<unsigned int> & numElements - Number of elements per file.
 *		          std::vector<std::string> & filenames - Names of the files.
 *                std::vector<std::vector<std::vector<float>>> & percentSorts - All
 *                bench sorting percentage data.
 *  Description:  Outputs benchmark data.
 * =====================================================================================
 */

void outputResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & percentSorts) {
	for (unsigned int i = 0 ; i < numElements.size() ; ++i) {
		for (unsigned int j = 0 ; j < times[i].size() ; ++j) {	
			char base[1000] ;
			strcpy(base,filenames[i].c_str()) ;
			std::string datFileName = "./bench_data/times" + sorts[j]->getAlgName() + 
				std::to_string(numElements[i]) + basename(base) ;
			std::ofstream output(datFileName) ;
			for (unsigned int k = 0 ; k < times[i][j].size() ; ++k) {
				float sortRate = numElements[i]/(times[i][j][k]*1E6) ;
				output << numElements[i] << " " << times[i][j][k] << " " <<  sortRate 
				<< " " << percentSorts[i][j][k] << std::endl ;
			}
			output.close() ;
		}
	}
}		/* -----  end of function   ----- */



