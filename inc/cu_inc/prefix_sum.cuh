
#ifndef PREFIX_SUM_CU_T02OEQEL
#define PREFIX_SUM_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  prefix_sum.cuh
 *
 *    Description:  Header file for prefix sum function used in radix sort.
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

__global__ void prefixSum(int * indices, int * globalValues, int size, int mask) ;

#endif /* end of header gaurd PREFIX_SUM_CU_T02OEQEL */
