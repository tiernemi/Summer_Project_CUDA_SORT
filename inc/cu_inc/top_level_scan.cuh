#ifndef TOP_LEVEL_SCAN
#define TOP_LEVEL_SCAN

#include "gpu_params.cuh"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  topLevelScan
 *    Arguments:  int * input - Array of digit offsets to be scanned.
 *                int * output - Location to write scanned offsets to.
 *  Description:  Performs a prefix scan using warp shuffles. This has the benefit of
 *                not requiring thread synchronisation. Unrolled for RADIXSIZE of 4.
 *                Taken from paper on light prefix sum.
 * =====================================================================================
*/

template <int digitPos>
__global__ void topLevelScan(const int * __restrict__ input, int * output) {
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int threadID = threadIdx.x ;
	int warpID = threadID/WARPSIZE ;
	int offset = blockIdx.x*blockDim.x*RADIXSIZE + warpID*WARPSIZE*RADIXSIZE ;

	int localVal1 = input[offset+laneID] ;
	int localVal2 = input[offset+laneID+32] ;
	int localVal3 = input[offset+laneID+64] ;
	int localVal4 = input[offset+laneID+96] ;

	#pragma unroll
	for (int i = 1 ; i <= WARPSIZE ; i <<= 1) {
		int temp = __shfl_up(localVal1,i) ;
		if (laneID >= i) {
			localVal1 += temp ;
		}
	}

	#pragma unroll
	for (int i = 1 ; i <= WARPSIZE ; i <<= 1) {
		int temp = __shfl_up(localVal2,i) ;
		if (laneID >= i) {
			localVal2 += temp ;
		}
	}

	#pragma unroll
	for (int i = 1 ; i <= WARPSIZE ; i <<= 1) {
		int temp = __shfl_up(localVal3,i) ;
		if (laneID >= i) {
			localVal3 += temp ;
		}
	}

	#pragma unroll
	for (int i = 1 ; i <= WARPSIZE ; i <<= 1) {
		int temp = __shfl_up(localVal4,i) ;
		if (laneID >= i) {
			localVal4 += temp ;
		}
	}

	localVal2 += __shfl(localVal1,WARPSIZE_MIN_1) ;
	localVal3 += __shfl(localVal2,WARPSIZE_MIN_1) ;
	localVal4 += __shfl(localVal3,WARPSIZE_MIN_1) ;

	 __shared__ int warpReduceVals[NUM_BLOCKS/WARPSIZE] ;

	if (laneID == WARPSIZE_MIN_1) {
		warpReduceVals[warpID] = localVal4 ;
	}
	__syncthreads() ;

	int temp1 = 0 ;
	if (warpID == 0) {
		if (laneID < NUM_BLOCKS/WARPSIZE) {
			temp1 = warpReduceVals[laneID] ;
			#pragma unroll
			for (int i = 1 ; i <= blockDim.x/WARPSIZE ; i<<=1) {
				int temp2 = __shfl_up(temp1,i) ;
				if (laneID >= i) {
					temp1 += temp2 ;
				}
			}
			warpReduceVals[laneID] = temp1 ;
		}
	}
	__syncthreads() ;

	temp1 = warpID == 0 ? 0 : warpReduceVals[warpID-1] ;

	localVal1 += temp1 ;
	localVal2 += temp1 ;
	localVal3 += temp1 ;
	localVal4 += temp1 ;

	output[offset+laneID] = localVal1 ;
	output[offset+32+laneID] = localVal2 ;
	output[offset+64+laneID] = localVal3 ;
	output[offset+96+laneID] = localVal4 ;

}		/* -----  end of function topLevelScan  ----- */



#endif
