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
#include <set>

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

static std::vector<Centroid> generateCentroidsNormal(long long numCentroids) ;
static std::vector<Centroid> generateCentroidsUni(long long numCentroids) ;

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


	CentroidSorter<ImplRadixSort> implSorter(centroids) ;

	ids = implSorter.sort(cameras[0]) ;
	/*  
	int pow2 = 16 ;
	int maxPow2 = 2 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<ImplRadixSort> * implSorter = new CentroidSorter<ImplRadixSort>(ranData) ;
		ids = implSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		if(idSet.size() == ids.size()) {
			std::cout << "Ids ok I: " << pow2 << std::endl;
		} else {
			std::cout << "Ids not ok I: " << pow2 << std::endl;
		}
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		if (Tests::checkSorted(distsRan)) {
			std::cout << "Dists ok I: " << pow2 << std::endl;
		} else {
			std::cout << "Dists not ok I: " << pow2 << std::endl;
		}
		delete implSorter ;
		pow2 *= 2 ;
	}
	*/



	return EXIT_SUCCESS ;
}

static std::vector<Centroid> generateCentroidsNormal(long long numCentroids) {
	std::mt19937 gen ;
	gen.seed(150992) ;
	std::normal_distribution<float> distrib(50,15) ;
	std::vector<Centroid> data ;
	for (long long i = 0 ; i < numCentroids ; ++i) {
		float x = distrib(gen) ;
		float y = distrib(gen) ;
		float z = distrib(gen) ;
		data.push_back(Centroid(x,y,z,i)) ;
	}
	return data ;
}


static std::vector<Centroid> generateCentroidsUni(long long numCentroids) {
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


