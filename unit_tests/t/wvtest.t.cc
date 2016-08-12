#include <vector>
#include <algorithm>
#include <iostream>
#include <set>

#include "../wvtest.h"
#include "../../inc/cpp_inc/test_funcs.hpp"
#include "../../inc/cpp_inc/fileloader.hpp"
#include "../../inc/cpp_inc/cmd_parser.hpp"
#include "../../inc/cpp_inc/c_variable.hpp"
#include "../../inc/cpp_inc/c_flag.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/camera.hpp"
#include "../../inc/cpp_inc/centroid_sorter.hpp"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  WVTEST_MAIN("Sorting tests") 
 *  Description:  Unit tests for successful sorts)
 * =====================================================================================
 */

WVTEST_MAIN("STL TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testData.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<STLSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}


WVTEST_MAIN("PT RADIX TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<PTSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}

WVTEST_MAIN("MHERF RADIX TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<MHerfSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}

WVTEST_MAIN("THRUST TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<ThrustSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}

WVTEST_MAIN("IMPLEMENTED RADIX SORT TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<ImplRadixSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}


WVTEST_MAIN("CUB RADIX TEST")
{
	std::vector<Centroid> centroids ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(centroids,cameras,filename) ;

	std::unordered_map<int,Centroid> centroidsMap ;
	for (int i = 0 ; i < centroids.size() ; ++i) {
		centroidsMap[centroids[i].getID()] = centroids[i] ;
	}

	std::vector<Centroid> checkVector(centroids.size()) ;
	std::vector<std::pair<int,float>> dists(centroids.size()) ;
	std::vector<int> ids ;

	CentroidSorter<CUBSort> sorter(centroids) ;
	std::cout << cameras.size() << std::endl;
	for (int i = 0 ; i < cameras.size() ; ++i) {
		ids = sorter.sort(cameras[i]) ;
		// Check if all IDs are unique //
		std::set<int> idSet ;
		for (int i = 0 ; i < ids.size() ; ++i) {
			idSet.insert(ids[i]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < centroids.size() ; ++j) {
			checkVector[j] = centroidsMap[ids[j]] ;
		}
		Transforms::transformToDistVec(dists,checkVector,cameras[i]) ;
		WVPASS(Tests::checkSorted(dists)) ;
	}
}






