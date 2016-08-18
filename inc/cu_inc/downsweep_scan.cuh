#ifndef DOWNSWEEP
#define DOWNSWEEP

#include "gpu_params.cuh"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  downsweepScan
 *    Arguments:  int * keysIn - Array of keys to be read in.
 *                int * keysOut - Array to scatter keys into.
 *                int * valuesIn - Array of values to be read in.
 *                int * valuesOut - Array to scatter values into.
 *                int * blockOffsetsArray - Array containing digit block offsets.
 *                int numElements - Number of keys.
 *                int digitPos - The digit position to decode.
 *                const int numTiles - Number of tiles to process in serial.
 *  Description:  Performs a prefix scan of the digit flag vectors and adds the block
 *                offsets calculated in the top level scan. Does an interal shared memory
 *                shuffle to boost coalesced writing to global memory.
 * =====================================================================================
*/ 

template <int digitPos>
__global__ void downsweepScan(const int * __restrict__ keysIn, int * keysOut, const int * __restrict__ valuesIn, int * valuesOut, const int * __restrict__ blockOffsetsArray, int numElements, int numTiles) {

	int blockOffset = (NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD * blockIdx.x) * numTiles ;
	if (blockOffset >= numElements) {
		return ;
	}


	// Declarations. //
	__shared__ int warpReduceVals[NUM_WARPS] ;
	__shared__ int warpReduceValsSplit[NUM_WARPS*RADIXSIZE] ;
	__shared__ int digitTotals[RADIXSIZE] ;
	__shared__ int digitOffsets[RADIXSIZE] ;
	__shared__ int seedValues[RADIXSIZE] ;
	__shared__ int keysInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;
	//__shared__ int valuesInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;


	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int globalOffset = blockOffset  + laneID + warpID * WARPSIZE * NUM_KEYS_PER_THREAD ;
	int digit[NUM_KEYS_PER_THREAD] ;
	int numFlags[NUM_KEYS_PER_THREAD] ;
	int currentKey[NUM_KEYS_PER_THREAD] ;

	if (threadIdx.x < RADIXSIZE) {
		seedValues[threadIdx.x] = blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] ;
	}

	__syncthreads() ;
	

	// Early exit. //
	#pragma unroll
	for (int i = 0 ; i < RADIXSIZE-2 ; ++i) {
		if (seedValues[i+1] == numElements) {
			if (seedValues[i+1] - seedValues[i] + blockOffset == numElements) {
				for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
					keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
					valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
				}
				return ;
			}
		}
	}
	int temp = seedValues[RADIXSIZE-1] ;
	if (temp == blockOffset){
		for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
			keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
			valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
		}
		return ;
	} else if (temp == numElements) {
		if (temp - seedValues[RADIXSIZE-2] + blockOffset == numElements) {
			for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
				keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
				valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
			}
			return ;
		}
	}

	
	// Process each tile sequentially. //
	for (int k = 0 ; k < numTiles ; ++k) {

		// Load and decode keys plus store in shared memory. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			numFlags[i] = 0 ;
			if (globalOffset+i*WARPSIZE < numElements) {
				currentKey[i] = keysIn[globalOffset+i*WARPSIZE] ;
				digit[i] = (currentKey[i] >> digitPos) & RADIXMASK ;
				numFlags[i] = (1 << PARA_BIT_SIZE*digit[i]) ;
			}
		}

		// Warp level prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			numFlags[i] = warpIncPrefixSum(numFlags[i], laneID, WARPSIZE) ;
		}

		// Combine prefix sums. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD-1 ; ++i) {
			numFlags[i+1] += __shfl(numFlags[i],WARPSIZE_MIN_1) ;
		}

		// Save warp digit total. //
		if (laneID == WARPSIZE_MIN_1) {
			warpReduceVals[warpID] = numFlags[NUM_KEYS_PER_THREAD-1] ;
		}
		__syncthreads() ;

		// Do a final interwarp prefix sum. //
		temp = 0 ;
		if (warpID == 0) {
			if (laneID < NUM_WARPS) {
				int temp2 = warpReduceVals[laneID] ;
				// Load warp digit totals. //
				#pragma unroll
				for (int i = 0 ; i < RADIXSIZE ; ++i) {
					temp = (temp2 >> i*PARA_BIT_SIZE) & PARA_BIT_MASK_0 ;
					temp = warpIncPrefixSum(temp, laneID, NUM_WARPS) ;
					// Save digit totals for the block. //
					if (laneID == (NUM_WARPS - 1)) {
						digitTotals[i] = temp  ;
					}
					warpReduceValsSplit[i*NUM_WARPS+laneID] = temp ;
				}
				// Save the new prefix summed warp totals. //
			}
		}

		__syncthreads() ;

		// Scan block digit totals. //
		temp = 0 ;
		if (warpID == 0) {
			if (laneID < RADIXSIZE) {
				// Load warp digit totals. //
				temp = (digitTotals[laneID]) ;
				temp = warpIncPrefixSum(temp, laneID, RADIXSIZE) ;
				// Save the new prefix summed warp digit totals. //
				digitOffsets[laneID] = temp ;
			}
		}

		__syncthreads() ;

		// Shuffle keys and values in shared memory per tile. //
		int newOffset[4] ;
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int digOffset = (digit[i] == 0 ? 0 : digitOffsets[digit[i]-1]) ;
				int warpOffset = (warpID == 0 ? 0:warpReduceValsSplit[(digit[i]*NUM_WARPS+warpID-1)]) ;
				int localFlag = (numFlags[i] >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK_0 ;
				newOffset[i]  = localFlag + digOffset + warpOffset - 1 ;
				keysInShr[newOffset[i]] = currentKey[i] ;
		//		valuesInShr[newOffset] = valuesIn[globalOffset+i*WARPSIZE] ;
			}
		}

		__syncthreads() ;

		int globalWriteLoc[4] ;
		// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				int currentKey = keysInShr[readLoc] ;
			//	int currentVal = valuesInShr[readLoc] ;
				int newDigit = (currentKey >> digitPos) & RADIXMASK ;
				int writeOffset = (newDigit == 0 ? 0 : digitOffsets[newDigit-1]) ;
				globalWriteLoc[i] = seedValues[newDigit] + readLoc - writeOffset ;
				keysOut[globalWriteLoc[i]] = currentKey ;
			//	valuesOut[globalWriteLoc] = valuesInShr[readLoc] ;
			} 
		}

		
		__syncthreads() ;

		// Shuffle keys and values in shared memory per tile. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				keysInShr[newOffset[i]] = valuesIn[globalOffset+i*WARPSIZE] ;
			}
		}

		__syncthreads() ;

		// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				valuesOut[globalWriteLoc[i]] = keysInShr[readLoc] ;
			} 
		}
	
		if (threadIdx.x < RADIXSIZE) {
			seedValues[threadIdx.x] += (digitTotals[threadIdx.x]) ;
		}

		globalOffset += NUM_THREADS_PER_BLOCK*NUM_KEYS_PER_THREAD ;
	}
}

template <int digitPos>
__global__ void downsweepScanFinal(const int * __restrict__ keysIn, const int * __restrict__ valuesIn, int * valuesOut, const int * __restrict__ blockOffsetsArray, int numElements, int numTiles) {

	int blockOffset = (NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD * blockIdx.x) * numTiles ;
	if (blockOffset >= numElements) {
		return ;
	}


	// Declarations. //
	__shared__ int warpReduceVals[NUM_WARPS] ;
	__shared__ int warpReduceValsSplit[NUM_WARPS*RADIXSIZE] ;
	__shared__ int digitTotals[RADIXSIZE] ;
	__shared__ int digitOffsets[RADIXSIZE] ;
	__shared__ int seedValues[RADIXSIZE] ;
	__shared__ int keysInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;
	//__shared__ int valuesInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;


	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int globalOffset = blockOffset  + laneID + warpID * WARPSIZE * NUM_KEYS_PER_THREAD ;
	int digit[NUM_KEYS_PER_THREAD] ;
	int numFlags[NUM_KEYS_PER_THREAD] ;
	int currentKey[NUM_KEYS_PER_THREAD] ;

	if (threadIdx.x < RADIXSIZE) {
		seedValues[threadIdx.x] = blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] ;
	}

	__syncthreads() ;
	

	// Early exit. //
	#pragma unroll
	for (int i = 0 ; i < RADIXSIZE-2 ; ++i) {
		if (seedValues[i+1] == numElements) {
			if (seedValues[i+1] - seedValues[i] + blockOffset == numElements) {
				for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
					valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
				}
				return ;
			}
		}
	}
	int temp = seedValues[RADIXSIZE-1] ;
	if (temp == blockOffset){
		for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
			valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
		}
		return ;
	} else if (temp == numElements) {
		if (temp - seedValues[RADIXSIZE-2] + blockOffset == numElements) {
			for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
				valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
			}
			return ;
		}
	}

	
	// Process each tile sequentially. //
	for (int k = 0 ; k < numTiles ; ++k) {

		// Load and decode keys plus store in shared memory. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			numFlags[i] = 0 ;
			if (globalOffset+i*WARPSIZE < numElements) {
				currentKey[i] = keysIn[globalOffset+i*WARPSIZE] ;
				digit[i] = (currentKey[i] >> digitPos) & RADIXMASK ;
				numFlags[i] = (1 << PARA_BIT_SIZE*digit[i]) ;
			}
		}

		// Warp level prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			numFlags[i] = warpIncPrefixSum(numFlags[i], laneID, WARPSIZE) ;
		}

		// Combine prefix sums. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD-1 ; ++i) {
			numFlags[i+1] += __shfl(numFlags[i],WARPSIZE_MIN_1) ;
		}

		// Save warp digit total. //
		if (laneID == WARPSIZE_MIN_1) {
			warpReduceVals[warpID] = numFlags[NUM_KEYS_PER_THREAD-1] ;
		}
		__syncthreads() ;

		// Do a final interwarp prefix sum. //
		temp = 0 ;
		if (warpID == 0) {
			if (laneID < NUM_WARPS) {
				int temp2 = warpReduceVals[laneID] ;
				// Load warp digit totals. //
				#pragma unroll
				for (int i = 0 ; i < RADIXSIZE ; ++i) {
					temp = (temp2 >> i*PARA_BIT_SIZE) & PARA_BIT_MASK_0 ;
					temp = warpIncPrefixSum(temp, laneID, NUM_WARPS) ;
					// Save digit totals for the block. //
					if (laneID == (NUM_WARPS - 1)) {
						digitTotals[i] = temp  ;
					}
					warpReduceValsSplit[i*NUM_WARPS+laneID] = temp ;
				}
				// Save the new prefix summed warp totals. //
			}
		}

		__syncthreads() ;

		// Scan block digit totals. //
		temp = 0 ;
		if (warpID == 0) {
			if (laneID < RADIXSIZE) {
				// Load warp digit totals. //
				temp = (digitTotals[laneID]) ;
				temp = warpIncPrefixSum(temp, laneID, RADIXSIZE) ;
				// Save the new prefix summed warp digit totals. //
				digitOffsets[laneID] = temp ;
			}
		}

		__syncthreads() ;

		// Shuffle keys and values in shared memory per tile. //
		int newOffset[4] ;
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int digOffset = (digit[i] == 0 ? 0 : digitOffsets[digit[i]-1]) ;
				int warpOffset = (warpID == 0 ? 0:warpReduceValsSplit[(digit[i]*NUM_WARPS+warpID-1)]) ;
				int localFlag = (numFlags[i] >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK_0 ;
				newOffset[i]  = localFlag + digOffset + warpOffset - 1 ;
				keysInShr[newOffset[i]] = currentKey[i] ;
		//		valuesInShr[newOffset] = valuesIn[globalOffset+i*WARPSIZE] ;
			}
		}

		__syncthreads() ;

		int globalWriteLoc[4] ;
		// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				int currentKey = keysInShr[readLoc] ;
			//	int currentVal = valuesInShr[readLoc] ;
				int newDigit = (currentKey >> digitPos) & RADIXMASK ;
				int writeOffset = (newDigit == 0 ? 0 : digitOffsets[newDigit-1]) ;
				globalWriteLoc[i] = seedValues[newDigit] + readLoc - writeOffset ;
			//	keysOut[globalWriteLoc[i]] = currentKey ;
			//	valuesOut[globalWriteLoc] = valuesInShr[readLoc] ;
			} 
		}

		
		__syncthreads() ;

		// Shuffle keys and values in shared memory per tile. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				keysInShr[newOffset[i]] = valuesIn[globalOffset+i*WARPSIZE] ;
			}
		}

		__syncthreads() ;

		// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
		#pragma unroll
		for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
			if (globalOffset+i*WARPSIZE < numElements) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				valuesOut[globalWriteLoc[i]] = keysInShr[readLoc] ;
			} 
		}
	
		if (threadIdx.x < RADIXSIZE) {
			seedValues[threadIdx.x] += (digitTotals[threadIdx.x]) ;
		}

		globalOffset += NUM_THREADS_PER_BLOCK*NUM_KEYS_PER_THREAD ;
	}
}



#endif
