#include <vector>
#include <iostream>

#include "../wvtest.h"
#include "../../inc/cpp_inc/test_funcs.hpp"
#include "../../inc/cpp_inc/fileloader.hpp"
#include "../../inc/cpp_inc/cmd_parser.hpp"
#include "../../inc/cpp_inc/c_variable.hpp"
#include "../../inc/cpp_inc/c_flag.hpp"
#include "../../inc/cpp_inc/transforms.hpp"
#include "../../inc/cpp_inc/camera.hpp"
#include "../../inc/cpp_inc/sort_algs.hpp"

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
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/TVS2_TransparentGeometryData_2016_06_21_00001.txt") ;
	FileLoader::loadFile(triangles,cameras,filename) ;


	std::vector<Triangle> temp = triangles ;
	std::vector<std::pair<int,float>> distances(triangles.size()) ;

	/* 
	// Convert to sortable form //
	CPUSorts::STLSort stlSorter ;
	stlSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;
	*/
	/*  
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
	*/
	/*  

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
	thrustSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	GPUSorts::RadixGPUSort radixGPUSorter ;
	radixGPUSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

	*/

	/*
	GPUSorts::CubRadixGPUSort cubSorter ;
	cubSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;
*/
	/* 
	GPUSorts::BasicRadixGPUSort baseSorter ;
	baseSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;
	*/

	/*  
	GPUSorts::SharedRadixGPUSort shareSorter ;
	shareSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;
	*/

	GPUSorts::MerrelRadixGPUSort merSorter ;
	merSorter.sortTriangles(triangles,cameras) ;
	Transforms::transformToDistVec(distances,triangles,cameras[cameras.size()-1]) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
	triangles = temp ;

}



