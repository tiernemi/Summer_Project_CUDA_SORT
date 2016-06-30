/*
 * =====================================================================================
 *
 *       Filename:  scattered_write.cu
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

#define WARPSIZE 32
#define RADIXSIZE 4

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  scatteredWrite
 *    Arguments:  int * indices - Indices array.
 *                float * values - Values array.
 *  Description:  Generates 2^d radix bitmasks and calculates their prefix sum for a
 *                given block of indices and values. 
 * =====================================================================================
 */

__global__ void scatteredWrite(int * indices, int * globalValues, int * localPrefixSumArray, int * blockSumArray, int numTriangles, int bits) {

	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int threadID = threadIdx.x ;
	extern __shared__ int compositeArray[] ;
	
	__shared__ int blockPrefixSum[RADIXSIZE] ;

	if (globalID < numTriangles) {
			if (threadID < RADIXSIZE) {
				blockPrefixSum[threadID] = blockSumArray[threadID*gridDim.x+blockIdx.x] ;
			}

			int * blockSortedData = &compositeArray[0] ;
			int * blockIndices = &compositeArray[blockDim.x] ;

			blockSortedData[threadID] = globalValues[globalID] ;
			blockIndices[threadID] = indices[globalID] ;
			__syncthreads() ;

				//printf("%d %d\n", threadID, localPrefixSumArray[threadID]) ;
				int finalPos = blockPrefixSum[(blockSortedData[threadID]>>bits)&3] + localPrefixSumArray[threadID] ;
				globalValues[finalPos] = blockSortedData[threadID] ;
				indices[finalPos] = blockIndices[threadID] ;
		}
}
