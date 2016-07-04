#ifndef TEST_SUM_CU_T02OEQEL
#define TEST_SUM_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  test_sum.cuh
 *
 *    Description:  Header file for test utility functions used for debugging.
 *
 *        Version:  1.0
 *        Created:  2016-07-02 14:08
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

__global__ void printRadixValues(int * keys, const int numElements, int numDigits) ;
__global__ void checkSortedGlobal(int * keys, const int numElements, int numDigits) ;
__global__ void printPrefixValues(int * prefixSum, const int numElements) ;
__global__ void checkPrefixSumGlobal(int * prefixSum, const int numElements) ;

#endif /* end of header gaurd TEST_SUM_CU_T02OEQEL */
