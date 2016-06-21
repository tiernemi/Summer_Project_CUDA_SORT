#include <vector>
#include <iostream>

#include "../wvtest.h"
#include "../../inc/test_funcs.hpp"
#include "../../inc/fileloader.hpp"
#include "../../inc/cmd_parser.hpp"
#include "../../inc/c_variable.hpp"
#include "../../inc/c_flag.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/camera.hpp"
#include "../../inc/sort_algs.hpp"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  WVTEST_MAIN("Sorting tests") 
 *  Description:  Unit tests for successful sorts)
 * =====================================================================================
 */

WVTEST_MAIN("Sorting tests")
{
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt") ;
	FileLoader::loadFile(triangles,cameras,filename) ;

	std::vector<std::pair<int,float>> distances(triangles.size()) ;
	// Convert to sortable form //
	std::vector<Triangle> temp = triangles ;
	CPUSorts::STLSort stlSorter ;
	Transforms::transformToDistVec(distances,triangles,cameras[0]) ;
	stlSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	CPUSorts::BitonicSort bitonicSorter ;
	bitonicSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	CPUSorts::BubbleSort bubbleSorter ;
	bubbleSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	CPUSorts::RadixSortPT radixSorterPT ;
	radixSorterPT.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	CPUSorts::RadixSortHoff radixSorterHoff ;
	radixSorterHoff.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	CPUSorts::RadixSortHybrid radixSorterHybrid ;
	radixSorterHybrid.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	GPUSorts::ThrustGPUSort thrustSorter ;
	Transforms::transformToDistVec(distances,triangles,cameras[0]) ;
	thrustSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;
}



