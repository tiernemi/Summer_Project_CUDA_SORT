#include <vector>

#include "../wvtest.h"
#include "../../inc/test_funcs.hpp"
#include "../../inc/fileloader.hpp"
#include "../../inc/cmd_parser.hpp"
#include "../../inc/c_variable.hpp"
#include "../../inc/c_flag.hpp"
#include "../../inc/transforms.hpp"
#include "../../inc/camera.hpp"
#include "../../inc/cpu_sorts.hpp"

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
	std::string filename("/home/users/mschpc/2015/tiernemi/project/data/testData.txt") ;
	FileLoader::loadFile(triangles,cameras,filename) ;

	// Convert to sortable form //
	std::vector<std::pair<int,float>> distances ;
	Transforms::transformToDistVec(distances, triangles, cameras[0]) ;

	std::vector<std::pair<int,float>> unsortedDistances = distances ;
	CPUSorts::cpuSTLSort(unsortedDistances) ;
	WVPASSEQ(Tests::checkSorted(unsortedDistances),1) ;

	unsortedDistances = distances ;
	CPUSorts::cpuBitonicSort(unsortedDistances) ;
//	WVPASSEQ(Tests::checkSorted(unsortedDistances),1) ;
}



