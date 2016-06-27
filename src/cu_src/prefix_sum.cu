/*
 * =====================================================================================
 *
 *       Filename:  prefix_sum.cu
 *
 *    Description:  CUDA code for prefix sum on gpu. Prefix sum is an essential algorithmic
 *                  primitive in radix sorts. It's used to calculate offsets.
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

#include "stdio.h"

#define RADIXSIZE 4

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  prefixSum
 *    Arguments:  int * indices - Indices array.
 *                float * values - Values array.
 *  Description:  Generates 2^d radix bitmasks and calculates their prefix sum for a
 *                given block of indices and values. 
 * =====================================================================================
 */

__global__ void prefixSum(int * indices, int * globalValues, int size, int mask) {

	extern __shared__ int count[][RADIXSIZE] ;
	extern __shared__ int blockValues[] ;

	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int threadID = threadIdx.x ;

	blockValues[threadID] = globalValues[globalID] ;

	int digit = (blockValues[threadID] & mask) ;
	for (int i = 0 ; i < RADIXSIZE ; ++i) {
		count[threadID][digit] = 1 ;
	}

	int offset = 1 ;
	if (threadID % 2 == 0) {
		for (int i = size>>1 ; i > 0 ; i>>=1 ) {
			__syncthreads() ;
			if (threadID < i) {
				int a = offset*(2*threadID+1)-1;  
				int b = offset*(2*threadID+2)-1;  
				for (int j = 0 ; j < RADIXSIZE ; ++j) {
					count[b][j] += count[a][j] ;
				}
			}
			offset*= 2 ;
		}
	} else {
		for (int i = size>>1 ; i > 0 ; i>>=1 ) {
			__syncthreads() ;
		}
	}

	if (threadID < RADIXSIZE) { 
		count[size-1][threadID] = 0 ;
	} // clear the last element  

	printf("%d\n", count[threadID][1]);
}
