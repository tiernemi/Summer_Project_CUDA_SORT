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
void runSortednessBenchs(std::vector<Sort*> & sorts, std::vector<Triangle> & triangles
		, Camera & camera, std::vector<std::vector<float>> & times
		, std::vector<std::vector<float>> & sortedness, std::mt19937 & gen) ;
void outputTimeResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & percentSorts) ;
void outputSortednessResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & sortedness) ;
void outputSpeedUpResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		int id1, int id2, std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & percentSorts) ;

std::mt19937 gen ;

int main(int argc, char *argv[]) {

	gen.seed(199293939) ;

	// Variables storing benchmark data. //
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	std::vector<std::vector<std::vector<float>>> sortTimes ;
	std::vector<std::vector<std::vector<float>>> sortednessTimes ;
	std::vector<std::vector<std::vector<float>>> percentSorts ;
	std::vector<std::vector<std::vector<float>>> sortedness ;
	std::vector<Sort*> sorts ;
	std::vector<std::string> names ;
	std::vector<std::string> filenames ;
	std::vector<unsigned int> numElements ;

	// Run time benchmarks. //
	CPUSorts::STLSort stlSorter ;
	CPUSorts::BitonicSort bitonicSorter ;
	CPUSorts::RadixSortPT radixSorterPT ;
	CPUSorts::RadixSortHoff radixSorterHoff ;
	CPUSorts::RadixSortHybrid radixSorterHybrid ;
	//CPUSorts::BubbleSort bubbleSorter ;
	GPUSorts::ThrustGPUSort thrustSorter ;
	// Add sorts. //
	sorts.push_back(&stlSorter) ;
	sorts.push_back(&stlSorter) ;
	sorts.push_back(&bitonicSorter) ;
	sorts.push_back(&radixSorterPT) ;
	sorts.push_back(&radixSorterHoff) ;
	sorts.push_back(&radixSorterHybrid) ;
	sorts.push_back(&thrustSorter) ;
	//sorts.push_back(&bubbleSorter) ;

	// Read in file names. //
	for (int i = 1 ; i < argc ; ++i) {
		filenames.push_back(argv[i]) ;
	}
	numElements.resize(filenames.size()) ;
	sortTimes.resize(filenames.size()) ;
	sortednessTimes.resize(filenames.size()) ;
	percentSorts.resize(filenames.size()) ;
	sortedness.resize(filenames.size()) ;

	// For each filename, load and run benchmarks. //
	for (unsigned int i = 0 ; i < filenames.size() ; ++i) {
		FileLoader::loadFile(triangles,cameras,filenames[i]) ;
		numElements[i] = triangles.size() ;
		// Run CPU sorts for this data set. //
		runCPUBenchs(sorts,triangles,cameras,sortTimes[i],percentSorts[i]) ;
		// Measure performance of sorting algorithms Vs sortedness of array. //
		runSortednessBenchs(sorts,triangles,cameras[0],sortednessTimes[i],sortedness[i],gen) ;
	}

	// Output time data. //
	outputTimeResults(sortTimes, sorts, numElements, filenames,percentSorts) ;
	// Output sortedness data. //
	outputSortednessResults(sortednessTimes, sorts, numElements, filenames, sortedness) ;
	// SpeedUp data. //
	outputSpeedUpResults(sortTimes, sorts, 6, 3, numElements, filenames, percentSorts) ;

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
			std::cout << "Benchmarking : " << sorts[i]->getAlgName() <<
				" Camera : " << j  << std::endl ;
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
 * ===  FUNCTION  ======================================================================
 *         Name:  runSortednessBenchs
 *    Arguments:  std::vector<Sort*> & sorts - Array of sort pointers.
 *                std::vector<Triangle> & triangles - Vector of triangle objects.
 *                std::vector<Camera> & cameras - Camera to sort relative to.
 *                std::vector<std::vector<float>> & times - Vector for timing data.
 *                std::vector<std::vector<float>> & percentSorts - Vector for storing
 *                sort scores.
 *                std::mt19937 gen - Random number generator.
 *  Description:  Runs benchmarks tracking how various sorting algorithms perform as
 *                a function of the "sortedness" of the array.
 * =====================================================================================
 */

void runSortednessBenchs(std::vector<Sort*> & sorts, std::vector<Triangle> & triangles
		, Camera & camera, std::vector<std::vector<float>> & times
		, std::vector<std::vector<float>> & sortedness, std::mt19937 & gen) {


	std::vector<Triangle> tempTriangles = triangles ;
	std::vector<float> targetScores = {0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,
		0.009, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1} ;

	for (unsigned int i = 0 ; i < sorts.size() ; ++i) {
		std::vector<float> newTimes ;
		std::vector<float> scores ;
		for (int j = 0 ; j < targetScores.size() ; ++j) {
			tempTriangles = triangles ;
			std::cout << "Benchmarking : " << sorts[i]->getAlgName() <<
				" Sort Score : " << targetScores[j]  << std::endl ;
			float sortTime = 0 ;
			Tests::makePercentSorted(tempTriangles,camera,targetScores[j],gen) ;
			scores.push_back(Tests::calcPercentSorted(tempTriangles,camera)) ;
			sorts[i]->sortTriangles(tempTriangles,camera,sortTime) ;
			newTimes.push_back(sortTime) ;
		}
		times.push_back(newTimes) ;
		sortedness.push_back(scores) ;
	}
}		/* -----  end of function runSortednessBenchs  ----- */

/* 
 ======================================================================================
 *         Name:  outputTimeResults
 *    Arguments:  std::vector<std::vector<std::vector<float>>> & times - All bench times.
 *                std::vector<Sort*> & sorts - Array of sorting algorithms.
 *		          std::vector<unsigned int> & numElements - Number of elements per file.
 *		          std::vector<std::string> & filenames - Names of the files.
 *                std::vector<std::vector<std::vector<float>>> & percentSorts - All
 *                bench sorting percentage data.
 *  Description:  Outputs time benchmark data.
 * =====================================================================================
 */

void outputTimeResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
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

/* 
 ======================================================================================
 *         Name:  outputSortednessResults
 *    Arguments:  std::vector<std::vector<std::vector<float>>> & times - All bench times.
 *                std::vector<Sort*> & sorts - Array of sorting algorithms.
 *		          std::vector<unsigned int> & numElements - Number of elements per file.
 *		          std::vector<std::string> & filenames - Names of the files.
 *                std::vector<std::vector<std::vector<float>>> & sorts - All
 *                bench sorting score data.
 *  Description:  Outputs sortedness benchmark data.
 * =====================================================================================
 */

void outputSortednessResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & sortedness) {
	for (unsigned int i = 0 ; i < numElements.size() ; ++i) {
		for (unsigned int j = 0 ; j < times[i].size() ; ++j) {	
			char base[1000] ;
			strcpy(base,filenames[i].c_str()) ;
			std::string datFileName = "./bench_data/sortedness" + sorts[j]->getAlgName() + 
				std::to_string(numElements[i]) + basename(base) ;
			std::ofstream output(datFileName) ;
			for (unsigned int k = 0 ; k < times[i][j].size() ; ++k) {
				float sortRate = numElements[i]/(times[i][j][k]*1E6) ;
				output << numElements[i] << " " << times[i][j][k] << " " <<  sortRate 
				<< " " << sortedness[i][j][k] << std::endl ;
			}
			output.close() ;
		}
	}
}		/* -----  end of function   ----- */

/* 
 ======================================================================================
 *         Name:  outputSpeedUpResults
 *    Arguments:  std::vector<std::vector<std::vector<float>>> & times - All bench times.
 *                std::vector<Sort*> & sorts - Array of sorting algorithms.
 *		          std::vector<unsigned int> & numElements - Number of elements per file.
 *		          std::vector<std::string> & filenames - Names of the files.
 *                std::vector<std::vector<std::vector<float>>> & sorts - All
 *                bench sorting score data.
 *  Description:  Outputs sortedness benchmark data.
 * =====================================================================================
 */

void outputSpeedUpResults(std::vector<std::vector<std::vector<float>>> & times, std::vector<Sort*> & sorts, 
		int id1, int id2, std::vector<unsigned int> & numElements, std::vector<std::string> & filenames, 
		std::vector<std::vector<std::vector<float>>> & percentSorts) {
	for (unsigned int i = 0 ; i < numElements.size() ; ++i) {
		char base[1000] ;
		strcpy(base,filenames[i].c_str()) ;
		std::string datFileName = "./bench_data/speedup" + sorts[id1]->getAlgName() + "_"
			+ sorts[id2]->getAlgName() + std::to_string(numElements[i]) + basename(base) ;
		std::ofstream output(datFileName) ;
		for (unsigned int k = 0 ; k < times[i][id1].size() ; ++k) {
			float speedUp = times[i][id1][k]/times[i][id2][k] ;
			output << numElements[i] << " " << times[i][id1][k] << " " << times[i][id2][k]
			<< " " << percentSorts[i][id1][k] << std::endl ;
		}
		output.close() ;
	}
}		/* -----  end of function   ----- */





