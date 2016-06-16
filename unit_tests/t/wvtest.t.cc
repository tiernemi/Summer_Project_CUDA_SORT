#include <vector>

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

	// Convert to sortable form //
	std::vector<std::pair<int,float>> unsortedDistances(triangles.size()) ;
	Transforms::transformToDistVec(unsortedDistances, triangles, cameras[0]) ;

	std::vector<std::pair<int,float>> distances = unsortedDistances ;
	CPUSorts::STLSort stlSorter ;
	stlSorter.sortDistances(distances) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;

	distances = unsortedDistances ;
	CPUSorts::BitonicSort bitonicSorter ;
	bitonicSorter.sortDistances(distances) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;

	distances = unsortedDistances ;
	CPUSorts::RadixSortPT radixSorterPT ;
	radixSorterPT.sortDistances(distances) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;

	distances = unsortedDistances ;
	CPUSorts::RadixSortHoff radixSorterHoff ;
	radixSorterHoff.sortDistances(distances) ;
	WVPASSEQ(Tests::checkSorted(distances),1) ;
}



