/*
 * =====================================================================================
 *
 *       Filename:  calc_shared_prefix_sum.cu
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

#include "../../inc/cu_inc/prefix_sums.cuh"

#define RADIXSIZE 4

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcExclusivePrefixSum
 *    Arguments:  int * localSumArray - Array storing the local prefix sums.
 *                int numThreadsReq - Number of threads required for reduction.
 *  Description:  Uses a work efficient two phase prefix scan algorithm for gpuGems 3.
 *                Every thread gets two data points. Scan is exclusive.
 * =====================================================================================
 */

__global__ void calcExclusivePrefixSum(int * localSumArray,  int numThreadsReq) {

	extern __shared__ int sharedSum[] ;

	int threadID = threadIdx.x ;
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;

		// Load global data into shared memory. //
		sharedSum[2*threadID] = localSumArray[2*globalID] ;
		sharedSum[2*threadID+1] = localSumArray[2*globalID+1] ;

		int offset = 1 ;

		// Upsweep. //
		for (int i = 2*blockDim.x>>1 ; i > 0 ; i >>= 1) {
			__syncthreads() ;
			if (threadID < i) {
				int pos1 = offset*(2*threadID+1)-1 ;
				int pos2 = offset*(2*threadID+2)-1 ;
				sharedSum[pos2] += sharedSum[pos1] ;
			}
			offset *= 2 ;
		}

		// Seed exclusive scan. //
		if (threadID == 0) {
			sharedSum[2*blockDim.x-1] = 0 ;
		}

		// Downsweep. //
		for (int i = 1 ; i < 2*blockDim.x ; i *= 2) {
			offset >>= 1 ;
			__syncthreads() ;
			if (threadID < i) {
				int pos1 = offset*(2*threadID+1)-1 ;
				int pos2 = offset*(2*threadID+2)-1 ;
				int tempVal = sharedSum[pos1] ;
				sharedSum[pos1] = sharedSum[pos2] ;
				sharedSum[pos2] += tempVal ;
			}
		}
		__syncthreads() ;

		// Read back data to global memory. //
		localSumArray[2*globalID] = sharedSum[2*threadID] ;
		localSumArray[2*globalID+1] = sharedSum[2*threadID+1] ;

}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcInclusivePrefixSum
 *    Arguments:  int * localSumArray - Array storing the local prefix sums.
 *                int numThreadsReq - Number of threads required for reduction.
 *  Description:  Uses a work efficient two phase prefix scan algorithm for gpuGems 3.
 *                Every thread gets two data points. Scan is inclusive.
 * =====================================================================================
 */

__global__ void calcInclusivePrefixSum(int * localSumArray,  int numThreadsReq) {

	extern __shared__ int sharedSum[] ;

	int threadID = threadIdx.x ;
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	
	if (globalID < numThreadsReq) {

		// Load global data into shared memory. //
		sharedSum[2*threadID] = localSumArray[2*globalID] ;
		sharedSum[2*threadID+1] = localSumArray[2*globalID+1] ;

		int offset = 1 ;

		// Upsweep. //
		for (int i = 2*blockDim.x>>1 ; i > 0 ; i >>= 1) {
			__syncthreads() ;
			if (threadID < i) {
				int pos1 = offset*(2*threadID+1)-1 ;
				int pos2 = offset*(2*threadID+2)-1 ;
				sharedSum[pos2] += sharedSum[pos1] ;
			}
			offset *= 2 ;
		}

		// Downsweep. //
		for (int i = 1 ; i < 2*blockDim.x ; i *= 2) {
			offset >>= 1 ;
			__syncthreads() ;
			if (threadID < i) {
				int pos1 = offset*(2*threadID+1)-1 ;
				int pos2 = offset*(2*threadID+2)-1 ;
				int tempVal = sharedSum[pos1] ;
				sharedSum[pos1] = sharedSum[pos2] ;
				sharedSum[pos2] += tempVal ;
			}
		}
		__syncthreads() ;

		// Read back data to global memory. //
		localSumArray[2*globalID] = sharedSum[2*threadID] ;
		localSumArray[2*globalID+1] = sharedSum[2*threadID+1] ;
	}
}
