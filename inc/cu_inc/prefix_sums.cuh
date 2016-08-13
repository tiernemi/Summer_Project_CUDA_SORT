
#ifndef PREFIX_SUMS_CU_T02OEQEL
#define PREFIX_SUMS_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  prefix_sums.cuh
 *
 *    Description:  Header file for warp level prefix sum functions.
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

__device__ __forceinline__ int warpIncPrefixSum(int localVal, int laneID, int numThreads) {
	// Per warp reduction. //
	#pragma unroll
	for (int j = 1 ; j <= numThreads ; j <<= 1) {
		int temp = __shfl_up(localVal,j) ;
		if (laneID >= j) {
			localVal += temp ;
		}
	}
	return localVal ;
}

__device__ __forceinline__ int warpExPrefixSum(int localVal, int laneID, int numThreads) {
	// Per warp reduction. //
	#pragma unroll
	for (int j = 1 ; j <= numThreads ; j <<= 1) {
		int temp = __shfl_up(localVal,j) ;
		if (laneID >= j) {
			localVal += temp ;
		}
	}
	localVal = __shfl_up(localVal,1) ;
	if (laneID == 0) {
		localVal = 0 ;
	}
	return localVal ;
}

#endif /* end of header gaurd PREFIX_SUM_CU_T02OEQEL */
