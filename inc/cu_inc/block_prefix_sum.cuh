#ifndef BLOCK_PREFIX_SUM_CU_T02OEQEL
#define BLOCK_PREFIX_SUM_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  blockPrefix_sum.cuh
 *
 *    Description:  Header file for blockPrefix sum function used in radix sort.
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

__global__ void blockPrefixSum(int * indices, int * globalValues, int * blockSumArray) ;

#endif /* end of header gaurd BLOCK_PREFIX_SUM_CU_T02OEQEL */
