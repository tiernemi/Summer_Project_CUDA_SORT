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

static std::vector<Centroid> generateCentroidsUni(long long numCentroids) ;
static std::vector<Centroid> generateCentroidsNorm(long long numCentroids) ;

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

	// Test on input dataset. //
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<STLSort> * stlSorter = new CentroidSorter<STLSort>(ranData) ;
		ids = stlSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete stlSorter ;
		pow2 *= 2 ;
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<PTSort> * ptSorter = new CentroidSorter<PTSort>(ranData) ;
		ids = ptSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete ptSorter ;
		pow2 *= 2 ;
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<MHerfSort> * mherfSorter = new CentroidSorter<MHerfSort>(ranData) ;
		ids = mherfSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete mherfSorter ;
		pow2 *= 2 ;
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<ThrustSort> * thrustSorter = new CentroidSorter<ThrustSort>(ranData) ;
		ids = thrustSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete thrustSorter ;
		pow2 *= 2 ;
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
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
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete implSorter ;
		pow2 *= 2 ;
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

	int pow2 = 1 ;
	int maxPow2 = 22 ;
	for (int i = 1 ; i < maxPow2 ; ++i) {
		std::cout << i << std::endl;
		std::vector<Centroid> ranData = generateCentroidsUni(pow2) ;
		std::vector<Centroid> checkRanVector(pow2) ;
		std::unordered_map<int,Centroid> ranDataMap ;
		std::vector<std::pair<int,float>> distsRan(pow2) ;
		for (int j = 0 ; j < ranData.size() ; ++j) {
			ranDataMap[ranData[j].getID()] = ranData[j] ;
		}

		CentroidSorter<CUBSort> * cubSorter = new CentroidSorter<CUBSort>(ranData) ;
		ids = cubSorter->sort(cameras[0]) ;
		std::set<int> idSet ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			idSet.insert(ids[j]) ;
		}
		// Check if distances are sorted //
		WVPASS(idSet.size() == ids.size()) ;
		for (int j = 0 ; j < ids.size() ; ++j) {
			checkRanVector[j] = ranDataMap[ids[j]] ;
		}
		Transforms::transformToDistVec(distsRan,checkRanVector,cameras[0]) ;
		WVPASS(Tests::checkSorted(distsRan)) ;
		delete cubSorter ;
		pow2 *= 2 ;
	}


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






