/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.cu
 *
 *    Description:  CUDA code for basic radix sort on gpu.
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:33:20
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdio.h>

#include "../../inc/cu_inc/test_utils.cuh"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  printRadixValues
 *    Arguments:  int * keys - Array of keys.
 *                const int numElements - Number of keys. 
 *                int numDigits - The number of digits to print (right to left).
 *  Description:  Prints the radix values of numDigits bits for the keys (right to left).
 * =====================================================================================
 */

__global__ void printRadixValues(int * keys, const int numElements, int numDigits) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int mask = (1 << (numDigits)) - 1 ;
	if (globalID == 0) {
		for (int i = 0 ; i < numElements ; ++i) {
			printf("%d %d\n", (keys[i]) & (mask), mask ) ;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkSortedGlobal
 *    Arguments:  int * keys - Array of keys.
 *                const int numElements - Number of keys. 
 *                int numDigits - The number of digits to check (right to left).
 *  Description:  Prints if keys aren't sorted for a given number of digit bits.
 * =====================================================================================
 */

__global__ void checkSortedGlobal(int * keys, const int numElements, int numDigits) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int mask = (1 << (numDigits)) - 1 ;
	if (globalID == 0) {
		int preval = keys[0] & mask ;
		for (int i = 1 ; i < numElements ; ++i) {
			int curval = (keys[i]) & (mask) ;
			if (curval < preval) {
				printf("Not Sorted : %d %d %d %d\n", numDigits, i, curval, preval) ;
			}
			preval = curval ;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  printPrefixValues
 *    Arguments:  int * prefixSum - Array used for prefixSum.
 *                const int numElements - Number of keys. 
 *                int numDigits - The number of digits to print (right to left).
 *  Description:  Prints the prefix sum array.
 * =====================================================================================
 */

__global__ void printPrefixValues(int * prefixSum, const int numElements) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID == 0) {
		for (int i = 0 ; i < numElements ; ++i) {
			printf("%d %d\n", i, prefixSum[i]) ;
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  checkPrefixSum
 *    Arguments:  int * prefixSum - Prefix sum array (should be ordered).
 *                const int numElements - Number of elements.
 *  Description:  Prints if prefixSum isn't valid,
 * =====================================================================================
 */

__global__ void checkPrefixSumGlobal(int * prefixSum, const int numElements) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID == 0) {
		int preval = prefixSum[0] ;
		for (int i = 1 ; i < numElements ; ++i) {
			int curval = preval ;
			if (curval < preval) {
				printf("Not ordered prefix-sum : %d %d %d %d\n", i, curval, preval) ;
			}
			preval = curval ;
		}
	}
}
