
#ifndef PREFIX_SUMS_CU_T02OEQEL
#define PREFIX_SUMS_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  prefix_sums.cuh
 *
 *    Description:  Header file for prefix sum functions used in radix sort.
 *
 *        Version:  1.0
 *        Created:  07/06/16 17:22:08
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

__global__ void calcExclusivePrefixSum(int * localSumArray) ;
__global__ void calcInclusivePrefixSum(int * localSumArray) ;

#endif /* end of header gaurd PREFIX_SUM_CU_T02OEQEL */
