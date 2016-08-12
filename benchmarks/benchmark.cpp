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
#include "../inc/cpp_inc/test_funcs.hpp"
#include "../inc/cpp_inc/fileloader.hpp"
#include "../inc/cpp_inc/cmd_parser.hpp"
#include "../inc/cpp_inc/c_variable.hpp"
#include "../inc/cpp_inc/c_flag.hpp"
#include "../inc/cpp_inc/transforms.hpp"
#include "../inc/cpp_inc/camera.hpp"
#include "../inc/cpp_inc/clock.hpp"
#include "../inc/cpp_inc/test_funcs.hpp"
#include "../inc/cpp_inc/centroid_sorter.hpp"

typedef std::vector<std::vector<std::vector<float>>>  threeVec ;
typedef std::vector<std::vector<float>>  twoVec ;

static void outputTimeData(std::string algName, twoVec & cameraTimes, int numElements, std::string filename) ;
static void outputSpeedUpCompar(std::vector<std::string> algNames, threeVec & algTimes, std::vector<int> indices, int numElements,std::string filename) ;

int main(int argc, char *argv[]) {


	// Variables storing benchmark data. //
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename = argv[1] ;

	FileLoader::loadFile(centroids,cameras,filename) ;
	int numCentroids = centroids.size() ;
	int numSorts = 6 ;
	std::vector<int> ids ;
	threeVec algTimes(numSorts) ;
	int numElements = centroids.size() ;

	CentroidSorter<STLSort> * stlSorter = new CentroidSorter<STLSort>(centroids) ;
	twoVec cameraTimes(cameras.size()) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = stlSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking STL SORT " << i <<  std::endl;
	}
	algTimes[0] = cameraTimes ;
	outputTimeData("STL_Sort", cameraTimes, numElements, filename) ;
	delete stlSorter ;
	
/*  
	CentroidSorter<PTSort> ptSorter(centroids) ;
	CentroidSorter<MHerfSort> mherfSorter(centroids) ;
	CentroidSorter<ThrustSort> thrustSorter(centroids) ;
	CentroidSorter<CUBSort> cubSorter(centroids) ;
	CentroidSorter<ImplRadixSort> implSorter(centroids) ;
*/

	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<PTSort> * ptSorter = new CentroidSorter<PTSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = ptSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking PT SORT " << i <<  std::endl;
	}
	algTimes[1] = cameraTimes ;
	outputTimeData("PT_Sort", cameraTimes, numElements, filename) ;
	delete ptSorter ;
	

	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<MHerfSort> * mhSorter = new CentroidSorter<MHerfSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = mhSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking MHerf SORT " << i <<  std::endl;
	}
	algTimes[2] = cameraTimes ;
	outputTimeData("MHerf_Sort", cameraTimes, numElements, filename) ;
	delete mhSorter ;

	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<ThrustSort> * thrustSorter = new CentroidSorter<ThrustSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = thrustSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking Thrust SORT " << i <<  std::endl;
	}
	algTimes[3] = cameraTimes ;
	outputTimeData("Thrust_Sort", cameraTimes, numElements, filename) ;
	delete thrustSorter ;
	
	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<CUBSort> * cubSorter = new CentroidSorter<CUBSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = cubSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking CUB SORT " << i <<  std::endl;
	}
	algTimes[4] = cameraTimes ;
	outputTimeData("CUB_Sort", cameraTimes, numElements, filename) ;
	delete cubSorter ;
	
	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<ImplRadixSort> * implSorter = new CentroidSorter<ImplRadixSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		Clock clock ;
		clock.start() ;
		ids = implSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking Project Radix SORT " << i <<  std::endl;
	}
	algTimes[5] = cameraTimes ;
	outputTimeData("IMPL_Sort", cameraTimes, numElements, filename) ;
	delete implSorter ;
	

	return EXIT_SUCCESS ;
}


static void outputTimeData(std::string algName, twoVec & cameraTimes, int numElements, std::string filename) {
	char base[1000] ;
	strcpy(base,filename.c_str()) ;
	std::string datFileName = "./bench_data/times" + algName + std::to_string(numElements) +  basename(base) ;
	std::ofstream output(datFileName) ;
	for (unsigned int i = 0 ; i < cameraTimes.size() ; ++i) {
		float sortRateSortOnly = numElements/(cameraTimes[i][0]*1E6) ;
		float sortRateTransformsInc = numElements/(cameraTimes[i][1]*1E6) ;
		float sortRateSortSum = numElements/(cameraTimes[i][2]*1E6) ;
		float sortRateCPUTot = numElements/(cameraTimes[i][3]*1E6) ;
		output << numElements << " " << cameraTimes[i][0] << " " << cameraTimes[i][1] 
		<< " " << cameraTimes[i][2] <<  " " << cameraTimes[i][3]  << " " << sortRateSortOnly << " " << 
		sortRateTransformsInc << " " << sortRateSortSum << " " << sortRateCPUTot << " " << std::endl ;
	}
	output.close() ;
}

static void outputSpeedUpCompar(std::vector<std::string> algNames, threeVec & algTimes, std::vector<int> indices, int numElements,std::string filename) {
	
}
