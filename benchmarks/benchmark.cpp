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

static void outputTimeCamData(std::string algName, twoVec & cameraTimes, int numElements, std::string filename) ;
static void outputTimeSizeData(std::string algName, twoVec & sizeTimes, std::vector<int> & numElements) ;
static void outputSpeedUpComparSize(std::vector<std::string> algNames, threeVec & algTimes, std::vector<int> indices, std::vector<int> & numElements) ;
static std::vector<Centroid> generateCentroids(long long numCentroids) ;

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
	Clock clock ;

	CentroidSorter<STLSort> * stlSorter = new CentroidSorter<STLSort>(centroids) ;
	twoVec cameraTimes(cameras.size()) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = stlSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking STL SORT " << i <<  std::endl;
	}
	algTimes[0] = cameraTimes ;
	outputTimeCamData("STL_Sort", cameraTimes, numElements, filename) ;
	delete stlSorter ;
	
	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<PTSort> * ptSorter = new CentroidSorter<PTSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = ptSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking PT SORT " << i <<  std::endl;
	}
	algTimes[1] = cameraTimes ;
	outputTimeCamData("PT_Sort", cameraTimes, numElements, filename) ;
	delete ptSorter ;
	

	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<MHerfSort> * mhSorter = new CentroidSorter<MHerfSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = mhSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking MHerf SORT " << i <<  std::endl;
	}
	algTimes[2] = cameraTimes ;
	outputTimeCamData("MHerf_Sort", cameraTimes, numElements, filename) ;
	delete mhSorter ;

	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<ThrustSort> * thrustSorter = new CentroidSorter<ThrustSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = thrustSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking Thrust SORT " << i <<  std::endl;
	}
	algTimes[3] = cameraTimes ;
	outputTimeCamData("Thrust_Sort", cameraTimes, numElements, filename) ;
	delete thrustSorter ;
	
	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<CUBSort> * cubSorter = new CentroidSorter<CUBSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = cubSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking CUB SORT " << i <<  std::endl;
	}
	algTimes[4] = cameraTimes ;
	outputTimeCamData("CUB_Sort", cameraTimes, numElements, filename) ;
	delete cubSorter ;
	
	cameraTimes.clear() ;
	cameraTimes.resize(cameras.size()) ;
	CentroidSorter<ImplRadixSort> * implSorter = new CentroidSorter<ImplRadixSort>(centroids) ;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		clock.start() ;
		ids = implSorter->benchSort(cameras[i], cameraTimes[i]) ;
		clock.stop() ;
		cameraTimes[i].push_back(clock.getDuration()) ;
		std::cout << "Benchmarking Project Radix SORT " << i <<  std::endl;
	}
	algTimes[5] = cameraTimes ;
	outputTimeCamData("IMPL_Sort", cameraTimes, numElements, filename) ;
	delete implSorter ;

	long long pow2 = 1 ;
	long long maxPow2 = 28 ;
	threeVec algGenDataTimes(6) ;
	std::vector<float> sizeTimes ;
	std::vector<int> dataSizes ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << "Benchmarking Data Size " << pow2 << std::endl;
		dataSizes.push_back(pow2) ;
		std::vector<Centroid> ranData = generateCentroids(pow2) ;
		
		CentroidSorter<STLSort> * stlSorter = new CentroidSorter<STLSort>(ranData) ;
		clock.start() ;
		ids = stlSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete stlSorter ;
		algGenDataTimes[0].push_back(sizeTimes) ;
		sizeTimes.clear() ;
		
		CentroidSorter<PTSort> * ptSorter = new CentroidSorter<PTSort>(ranData) ;
		clock.start() ;
		ids = ptSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete ptSorter ;
		algGenDataTimes[1].push_back(sizeTimes) ;
		sizeTimes.clear() ;
		
		CentroidSorter<MHerfSort> * mherfSorter = new CentroidSorter<MHerfSort>(ranData) ;
		clock.start() ;
		ids = mherfSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete mherfSorter ;
		algGenDataTimes[2].push_back(sizeTimes) ;
		sizeTimes.clear() ;
		
		CentroidSorter<ThrustSort> * thrustSorter = new CentroidSorter<ThrustSort>(ranData) ;
		clock.start() ;
		ids = thrustSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete thrustSorter ;
		algGenDataTimes[3].push_back(sizeTimes) ;
		sizeTimes.clear() ;
		
		CentroidSorter<CUBSort> * cubSorter = new CentroidSorter<CUBSort>(ranData) ;
		clock.start() ;
		ids = cubSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete cubSorter ;
		algGenDataTimes[4].push_back(sizeTimes) ;
		sizeTimes.clear() ;

		CentroidSorter<ImplRadixSort> * implSorter = new CentroidSorter<ImplRadixSort>(ranData) ;
		Clock clock ;
		clock.start() ;
		ids = implSorter->benchSort(cameras[0], sizeTimes) ;
		clock.stop() ;
		sizeTimes.push_back(clock.getDuration()) ;
		delete implSorter ;
		algGenDataTimes[5].push_back(sizeTimes) ;
		sizeTimes.clear() ;

		pow2 *= 2 ;
	}

	outputTimeSizeData("STL_Sort", algGenDataTimes[0], dataSizes) ;
	outputTimeSizeData("PT_Sort", algGenDataTimes[1], dataSizes) ;
	outputTimeSizeData("MHerf_Sort", algGenDataTimes[2], dataSizes) ;
	outputTimeSizeData("Thrust_Sort", algGenDataTimes[3], dataSizes) ;
	outputTimeSizeData("CUB_Sort", algGenDataTimes[4], dataSizes) ;
	outputTimeSizeData("IMPL_Sort", algGenDataTimes[5], dataSizes) ;

	return EXIT_SUCCESS ;
}


static void outputTimeCamData(std::string algName, twoVec & cameraTimes, int numElements, std::string filename) {
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


static void outputTimeSizeData(std::string algName, twoVec & sizeTimes, std::vector<int> & numElements) {
	char base[1000] ;
	std::string datFileName = "./bench_data/sizetimes" + algName + ".txt" ;
	std::ofstream output(datFileName) ;
	for (unsigned int i = 0 ; i < sizeTimes.size() ; ++i) {
		float sortRateSortOnly = numElements[i]/(sizeTimes[i][0]*1E6) ;
		float sortRateTransformsInc = numElements[i]/(sizeTimes[i][1]*1E6) ;
		float sortRateSortSum = numElements[i]/(sizeTimes[i][2]*1E6) ;
		float sortRateCPUTot = numElements[i]/(sizeTimes[i][3]*1E6) ;
		output << numElements[i] << " " << sizeTimes[i][0] << " " << sizeTimes[i][1] 
		<< " " << sizeTimes[i][2] <<  " " << sizeTimes[i][3]  << " " << sortRateSortOnly << " " << 
		sortRateTransformsInc << " " << sortRateSortSum << " " << sortRateCPUTot << " " << std::endl ;
	}
	output.close() ;
}

static std::vector<Centroid> generateCentroids(long long numCentroids) {
	std::mt19937 gen ;
	gen.seed(15071992) ;
	std::uniform_real_distribution<float> distrib(30,80) ;
	std::vector<Centroid> data ;
	for (long long i = 0 ; i < numCentroids ; ++i) {
		float x = distrib(gen) ;
		float y = distrib(gen) ;
		float z = distrib(gen) ;
		data.push_back(Centroid(x,y,z,i)) ;
	}
	return data ;
}
