#ifndef UPSWEEP
#define UPSWEEP

#include "gpu_params.cuh"

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  upsweepScan
 *    Arguments:  int * keys - Array of keys to be read in.
 *                int * blockOffsetsArray - Array to which reductions are written.
 *                int numElements - Number of keys.
 *                int digitPos - The digit position to decode.
 *                const int numTiles - Number of tiles to process in serial.
 *  Description:  Performs shuffle reductions in order to compute the digit counts for
 *                a given block. These counts are written to global memory.
 * =====================================================================================
 */

template <int digitPos>
__global__ void upsweepReduce(const int * __restrict__ keys, int * blockOffsetsArray, int numElements, int numTiles) {

	int blockOffset = (NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD * blockIdx.x) * numTiles ;

	if (blockOffset >= numElements) {
		if (threadIdx.x < RADIXSIZE) {
			blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] = 0 ;
		}
		return ;
	}

	// Declarations. //
	__shared__ int sharedNumFlags[NUM_WARPS] ;
	__shared__ int tileTotal[RADIXSIZE] ;
	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int globalOffset = blockOffset + laneID + warpID * WARPSIZE * NUM_KEYS_PER_THREAD ;
	// Allocate array containing flag reductions. //
	int numFlags ;

	if (threadIdx.x < RADIXSIZE) {
		tileTotal[threadIdx.x] = 0 ;
	}
	
	for (int k = 0 ; k < numTiles ; ++k) {
		numFlags = 0  ;
		// Decode keys. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if ((globalOffset + i*WARPSIZE) < numElements) {
				int digit = (keys[globalOffset+i*WARPSIZE] >> digitPos) & RADIXMASK ;
				numFlags += (1 << PARA_BIT_SIZE*digit) ;
			}
		}

		// Warp level reduction. //
		#pragma unroll
		for (int j = WARPSIZE_HALF ; j > 0 ; j>>=1) {
			numFlags += __shfl_down(numFlags,j) ;
		}
		if (laneID == 0) {
			sharedNumFlags[warpID] = numFlags ;
		}
		__syncthreads() ;

		// Get data into first warp. //
		numFlags = (threadIdx.x < NUM_WARPS) ? 
		sharedNumFlags[laneID] : 0 ;
		// Final warp reduction. //
		if (warpID == 0) {
			#pragma unroll
			for (int i = 0 ; i < RADIXSIZE ; ++i) {
				int temp = (numFlags >> (i * PARA_BIT_SIZE)) & PARA_BIT_MASK_0 ;
				#pragma unroll
				for (int j = WARPSIZE_HALF ; j > 0 ; j>>=1) {
					temp += __shfl_down(temp,j) ;
				}
				if (threadIdx.x == 0) {
					tileTotal[i] += temp ;
				}
			}
		}

		globalOffset += NUM_THREADS_PER_BLOCK*NUM_KEYS_PER_THREAD ;
	}

	__syncthreads() ;

	if (threadIdx.x < RADIXSIZE) {
		blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] = tileTotal[threadIdx.x] ;
	}
}

#endif
