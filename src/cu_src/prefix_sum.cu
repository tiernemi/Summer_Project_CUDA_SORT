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

__global__ void prefixSum(int * indices, int * globalValues, int numTriangles, int mask) {

	extern __shared__ int compositeArray[] ;
	int * count = &compositeArray[0] ;
	int * blockValues = &compositeArray[RADIXSIZE*blockDim.x] ;

	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int threadID = threadIdx.x ;


	// Initialise shared mem to zero. //
	for (int i = 0 ; i < RADIXSIZE ; ++i) {
		count[RADIXSIZE*threadID+i] = 0 ;
	}

	if (globalID < numTriangles) {
		blockValues[threadID] = globalValues[globalID] ;
		int digit = (blockValues[threadID] & mask) ;
		count[RADIXSIZE*threadID+digit] = 1 ;
	} else {
		blockValues[threadID] = 0 ;
	}

	__syncthreads() ;
	int offset = 1 ;
	if (threadID % 2 == 0) {
		for (int i = blockDim.x>>1 ; i > 0 ; i>>=1 ) {
			__syncthreads() ;
			if (threadID < 2*i) {
				int a = offset*(threadID+1)-1 ;
				int b = offset*(threadID+2)-1 ;
				for (int j = 0 ; j < RADIXSIZE ; ++j) {
					count[RADIXSIZE*b+j] += count[RADIXSIZE*a+j] ;
				}
			}
			offset*= 2 ;
		}
	} else {
		for (int i = blockDim.x>>1 ; i > 0 ; i>>=1 ) {
			__syncthreads() ;
		}
	}

	
	if (threadID < RADIXSIZE) { 
		count[4*(blockDim.x-1)+threadID] = 0 ;
	} // clear the last element  

	__syncthreads() ;
	if (threadID==0) {
		for (int i = 0 ; i < blockDim.x ; ++i) {
				printf("%d : %d\n", i, count[RADIXSIZE*i]);
		}
	} 


	if (threadID % 2 == 0) {
		for (int i = 1 ; i < blockDim.x ; i*=2) {
			offset >>= 1 ;
			__syncthreads() ;
			if (threadID < 2*i) {
				int a = offset*(threadID+1)-1;  
				int b = offset*(threadID+2)-1;  
				printf("%d : %d %d\n", i, a,b );
				for (int j = 0 ; j < RADIXSIZE ; ++j) {
					int temp = count[RADIXSIZE*a+j] ;
					count[RADIXSIZE*a+j] = count[RADIXSIZE*b+j] ;
					count[RADIXSIZE*b+j] += temp ;
				}
			}
		}
	}
	else {
		for (int i = 1 ; i < blockDim.x ; i*=2) {
			__syncthreads() ;
		}
	}
	__syncthreads() ;

	if (threadID==0) {
		for (int i = 0 ; i < blockDim.x ; ++i) {
			printf("%d : %d\n", i, count[RADIXSIZE*i]);
		}
	}

}
