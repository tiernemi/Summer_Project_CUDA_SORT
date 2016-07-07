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

#define WARPSIZE 32
#define RADIXSIZE 4
#define NUM_BANKS WARPSIZE/2
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcExclusivePrefixSum
 *    Arguments:  int * localSumArray - Array storing the local prefix sums.
 *                int numThreadsReq - Number of threads required for reduction.
 *  Description:  Uses a work efficient two phase prefix scan algorithm for gpuGems 3.
 *                Every thread gets two data points. Scan is exclusive.
 * =====================================================================================
 */

__global__ void calcExclusivePrefixSum(int * localSumArray) {

	extern __shared__ int sharedSum[] ;

	int n = 2*blockDim.x ;
	int numThreads = blockDim.x ;
	int threadID = threadIdx.x ;
	int globalID = threadIdx.x + n*blockIdx.x ;

	// Position of data in shared memory. //
	int pos1B = threadID ;   
	int pos2B = threadID  + (numThreads) ;	
	int bankOffset1 = CONFLICT_FREE_OFFSET(pos1B) ;
	int bankOffset2 = CONFLICT_FREE_OFFSET(pos2B) ;

	// Load global data into shared memory. //
	sharedSum[pos1B+bankOffset1] = localSumArray[globalID] ;
	sharedSum[pos2B+bankOffset2] = localSumArray[globalID+(numThreads)] ;

	int offset = 1 ;
	const int loc1 = 2*threadID+1 ;
	const int loc2 = 2*threadID+2 ;

	// Upsweep. //
	for (int i = n>>1 ; i > 0 ; i >>= 1) {
		__syncthreads() ;
		if (threadID < i) {
			int pos1 = offset*(loc1)-1 ;
			int pos2 = offset*(loc2)-1 ;
			pos1 += CONFLICT_FREE_OFFSET(pos1) ;
			pos2 += CONFLICT_FREE_OFFSET(pos2) ;
			sharedSum[pos2] += sharedSum[pos1] ;
		}
		offset *= 2 ;
	}

	__syncthreads() ;
	// Seed exclusive scan. //
	if (threadID == 0) {
		sharedSum[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0 ;
	}

	// Downsweep. //
	for (int i = 1 ; i < n ; i *= 2) {
		offset >>= 1 ;
		__syncthreads() ;
		if (threadID < i) {
			int pos1 = offset*(loc1)-1 ;
			int pos2 = offset*(loc2)-1 ;
			pos1 += CONFLICT_FREE_OFFSET(pos1) ;
			pos2 += CONFLICT_FREE_OFFSET(pos2) ;
			int tempVal = sharedSum[pos1] ;
			sharedSum[pos1] = sharedSum[pos2] ;
			sharedSum[pos2] += tempVal ;
		}
	}

	__syncthreads() ;
	// Read back data to global memory. //
	localSumArray[globalID] = sharedSum[pos1B+bankOffset1] ;
	localSumArray[globalID+numThreads] = sharedSum[pos2B+bankOffset2] ;
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

__global__ void calcInclusivePrefixSum(int * localSumArray) {

		extern __shared__ int sharedSum[] ;

	int n = 2*blockDim.x ;
	int numThreads = blockDim.x ;
	int threadID = threadIdx.x ;
	int globalID = threadIdx.x + n*blockIdx.x ;

	// Position of data in shared memory. //
	int pos1B = threadID ;   
	int pos2B = threadID  + (numThreads) ;	
	int bankOffset1 = CONFLICT_FREE_OFFSET(pos1B) ;
	int bankOffset2 = CONFLICT_FREE_OFFSET(pos2B) ;

	// Load global data into shared memory. //
	sharedSum[pos1B+bankOffset1] = localSumArray[globalID] ;
	sharedSum[pos2B+bankOffset2] = localSumArray[globalID+(numThreads)] ;

	int offset = 1 ;
	const int loc1 = 2*threadID+1 ;
	const int loc2 = 2*threadID+2 ;

	// Upsweep. //
	for (int i = n>>1 ; i > 0 ; i >>= 1) {
		__syncthreads() ;
		if (threadID < i) {
			int pos1 = offset*(loc1)-1 ;
			int pos2 = offset*(loc2)-1 ;
			pos1 += CONFLICT_FREE_OFFSET(pos1) ;
			pos2 += CONFLICT_FREE_OFFSET(pos2) ;
			sharedSum[pos2] += sharedSum[pos1] ;
		}
		offset *= 2 ;
	}

	__syncthreads() ;

	// Downsweep. //
	for (int i = 1 ; i < n ; i *= 2) {
		offset >>= 1 ;
		__syncthreads() ;
		if (threadID < i) {
			int pos1 = offset*(loc1)-1 ;
			int pos2 = offset*(loc2)-1 ;
			pos1 += CONFLICT_FREE_OFFSET(pos1) ;
			pos2 += CONFLICT_FREE_OFFSET(pos2) ;
			int tempVal = sharedSum[pos1] ;
			sharedSum[pos1] = sharedSum[pos2] ;
			sharedSum[pos2] += tempVal ;
		}
	}

	__syncthreads() ;
	// Read back data to global memory. //
	localSumArray[globalID] = sharedSum[pos1B+bankOffset1] ;
	localSumArray[globalID+numThreads] = sharedSum[pos2B+bankOffset2] ;
}
