/*
 * =====================================================================================
 *
 *       Filename:  impl_radix_sort_policy.cu
 *
 *    Description:  CUDA code for my implemenation of an allocation based 3-kernel radix 
 *                  sort.
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:33:20
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#define WARPSIZE 32
#define WARPSIZE_HALF 16
#define P2_WARPSIZE_HALF 5
#define WARPSIZE_MIN_1 31
#define RADIXSIZE 4
#define RADIXMASK 3
#define RADIXWIDTH 2
#define NUM_KEYS_PER_THREAD 4

#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 128
#define NUM_WARPS NUM_THREADS_PER_BLOCK/WARPSIZE
#define PARA_BIT_SIZE 8
#define PARA_BIT_MASK 255

#include <stdio.h>
#include <cmath>
#include <math.h>

#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../inc/cu_inc/prefix_sums.cuh"
#include "../../inc/cpp_inc/impl_radix_sort_policy.hpp"


// Function Declarations. //
static __global__ void upsweepReduce(const int * __restrict__ keys, int * blockOffsetsArray, 
		int numElements, const int digitPos, const int numTiles) ;

static __global__ void topLevelScan(const int * __restrict__ input, int * output) ;

static __global__ void downsweepScan(const int * __restrict__ keysIn, int * keysOut, 
		const int * __restrict__ valuesIn, int * valuesOut, const int * __restrict__ blockOffsetsArray, 
		const int numElements, const int digitPos, const int numTiles) ;

static __global__ void downsweepScanFinal(const int * __restrict__ keysIn, const int * __restrict__ valuesIn, 
		int * valuesOut, const int * __restrict__ blockOffsetsArray, 
		const int numElements, const int digitPos, const int numTiles) ;

/* 
 * ===  MEMBER FUNCTION : ImplRadixSort  ============================================= ==
 *         Name:  allocate
 *    Arguments:  const std::vector<Centroid> & centroids - Centroid data to allocate.
 *      Returns:  Pair of pointers to centroid position data and ids.
 *  Description:  Allocates position and id data on the GPU.
 * =====================================================================================
 */

std::pair<float*,int*> ImplRadixSort::allocate(const std::vector<Centroid> & centroids) {
	std::pair<float*,int*> ptrs ;
	// Pre process triangle co-ordinates. //
	std::vector<float> cenCo(3*centroids.size()) ;
	std::vector<int> cenIds(centroids.size()) ;
	for (unsigned int i = 0 ; i < centroids.size() ; ++i) {
		cenIds[i] = centroids[i].getID() ;
		const float * coords = centroids[i].getCoords() ;
		cenCo[i] = coords[0] ;
		cenCo[i+centroids.size()] = coords[1] ;
		cenCo[i+2*centroids.size()] = coords[2] ;
	}

	cudaMalloc((void**) &ptrs.first, sizeof(float)*cenCo.size()) ;
	cudaMalloc((void**) &ptrs.second, sizeof(int)*cenIds.size()) ;
	cudaMemcpy(ptrs.first, cenCo.data(), sizeof(float)*cenCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(ptrs.second, cenIds.data(), sizeof(int)*cenIds.size(), cudaMemcpyHostToDevice) ;
	return ptrs ;
}

/* 
 * ===  MEMBER FUNCTION : ImplRadixSort ===============================================
 *         Name:  sort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using an allocation based 3-kernel sort from
 *                Duane Merril's paper.
 * =====================================================================================
 */

void ImplRadixSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, 
		int * centroidIDs, float * centroidPos) {

	// Tranfer camera data. //
	float * gpuCamCo = NULL ;
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	cudaMalloc((void**) &gpuCamCo, sizeof(float)*3) ;
	cudaMemcpy(gpuCamCo,camCo.data(),sizeof(float)*3,cudaMemcpyHostToDevice) ;

	// Create distances to be used as keys. //
	const int numCentroids = centroidIDsVec.size() ;
	const int numKeysPerTile = NUM_KEYS_PER_THREAD * NUM_THREADS_PER_BLOCK ;

	// Calculate number of tiles and add padding in order to remove bounds checking. //
	int numTiles = 1 ;	
	float tempVar = numCentroids / float(NUM_THREADS_PER_BLOCK * numTiles * NUM_BLOCKS * NUM_KEYS_PER_THREAD) ; 
	while (tempVar > 1) {
		++numTiles ;
		tempVar = numCentroids / float(NUM_THREADS_PER_BLOCK * numTiles * NUM_BLOCKS * NUM_KEYS_PER_THREAD)  ;
	}
	int padding = (numCentroids % numKeysPerTile) ? (numKeysPerTile - (numCentroids % numKeysPerTile)) : 0 ;

	// Calculate distance keys. //
	float * gpuDistancesSq = NULL ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*(numCentroids+padding)) ;
	dim3 distanceBlock(1024) ;	
	dim3 distanceGrid(numCentroids/distanceBlock.x + (!(numCentroids%distanceBlock.x)?0:1)) ;
	cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(centroidPos, gpuCamCo, gpuDistancesSq+padding, numCentroids) ;
	cudaMemset(gpuDistancesSq, 0, sizeof(float)*(padding)) ;

	// Create temporary memory. //
	int * keyPtr1 = (int*) gpuDistancesSq ;
	int * keyPtr2 = NULL ;
	int * valPtr1 = NULL ;
	int * valPtr2 = NULL ;
	cudaMalloc((void**) &valPtr1, sizeof(int)*(numCentroids+padding)) ;
	cudaMalloc((void**) &keyPtr2, sizeof(int)*(numCentroids+padding)) ;
	cudaMalloc((void**) &valPtr2, sizeof(int)*(numCentroids+padding)) ;
	cudaMemcpy(valPtr1+padding,centroidIDs,sizeof(int)*(numCentroids),cudaMemcpyDeviceToDevice) ;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) ;

	dim3 reductionBlock(NUM_THREADS_PER_BLOCK) ;
	dim3 reductionGrid(NUM_BLOCKS) ;

	// Assign more temporary memory for block reductions. //
	int * blockReduceArray = NULL ;
	cudaMalloc((void**) &blockReduceArray, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;
	cudaMemset(blockReduceArray, 0, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;

	// Launch sorting kernels for each bit group. //
	for (int i = 0 ; i < 30-RADIXWIDTH ; i+=RADIXWIDTH) {
		upsweepReduce<<<reductionGrid,reductionBlock>>>(keyPtr1,blockReduceArray+1,numCentroids+padding,i,numTiles) ;
		topLevelScan<<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
		downsweepScan<<<reductionGrid,reductionBlock>>>(keyPtr1, keyPtr2, valPtr1, valPtr2, blockReduceArray,
				numCentroids+padding, i, numTiles) ;
		std::swap(keyPtr1,keyPtr2) ;
		std::swap(valPtr1,valPtr2) ;
	}
	// Final kernel does not need to copy keys. //
	upsweepReduce<<<reductionGrid,reductionBlock>>>(keyPtr1,blockReduceArray+1,numCentroids+padding,30-RADIXWIDTH,numTiles) ;
	topLevelScan<<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
	downsweepScanFinal<<<reductionGrid,reductionBlock>>>(keyPtr1, valPtr1, valPtr2, blockReduceArray,
			numCentroids+padding, 30-RADIXWIDTH, numTiles) ;
	std::swap(keyPtr1,keyPtr2) ;
	std::swap(valPtr1,valPtr2) ;

	// Copy back memory to host. //
	cudaMemcpy(centroidIDsVec.data(), valPtr1+padding, sizeof(int)*numCentroids, cudaMemcpyDeviceToHost) ;

	cudaFree(blockReduceArray) ;
	cudaFree(valPtr1) ;
	cudaFree(keyPtr1) ;
	cudaFree(valPtr2) ;
	cudaFree(keyPtr2) ;
}

/* 
 * ===  MEMBER FUNCTION : ImplRadixSort ===============================================
 *         Name:  benchSort
 *    Arguments:  const Camera & camera, - Camera to sort relative to.
 *                std::vector<int> & centroidIDsVec, - Array to write ids to.
 *                int * centroidIDs - Array of centroid ids (GPU).
 *                float * centroidPos - Array of centroid positions (GPU).
 *                std::vector<float> & times - Vector used to store timings.
 *  Description:  Transforms centorid positions to distances and sorts these keys
 *                and ids (values) using an allocation based 3-kernel sort from
 *                Duane Merril's paper. This version benchmarks aswell.
 * =====================================================================================
 */

void ImplRadixSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, 
		int * centroidIDs, float * centroidPos, std::vector<float> & times) {

	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	
	// Tranfer camera data. //
	float * gpuCamCo = NULL ;
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	cudaMalloc((void**) &gpuCamCo, sizeof(float)*3) ;
	cudaMemcpy(gpuCamCo,camCo.data(),sizeof(float)*3,cudaMemcpyHostToDevice) ;


	// Create distances to be used as keys. //
	const int numCentroids = centroidIDsVec.size() ;
	const int numKeysPerTile = NUM_KEYS_PER_THREAD * NUM_THREADS_PER_BLOCK ;
	int numTiles = 1 ;	
	float tempVar = numCentroids / float(NUM_THREADS_PER_BLOCK * numTiles * NUM_BLOCKS * NUM_KEYS_PER_THREAD) ; 
	while (tempVar > 1) {
		++numTiles ;
		tempVar = numCentroids / float(NUM_THREADS_PER_BLOCK * numTiles * NUM_BLOCKS * NUM_KEYS_PER_THREAD)  ;
	}
	int padding = (numCentroids % numKeysPerTile) ? (numKeysPerTile - (numCentroids % numKeysPerTile)) : 0 ;

	// Calculate distance keys. //
	float * gpuDistancesSq = NULL ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*(numCentroids+padding)) ;
	dim3 distanceBlock(1024) ;	
	dim3 distanceGrid(numCentroids/distanceBlock.x + (!(numCentroids%distanceBlock.x)?0:1)) ;
	cudaEventRecord(start, 0) ;
	cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(centroidPos, gpuCamCo, gpuDistancesSq+padding, numCentroids) ;
	cudaMemset(gpuDistancesSq, 0, sizeof(float)*(padding)) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float transformTime ;
	cudaEventElapsedTime(&transformTime , start, stop) ;
	
	int * keyPtr1 = (int*) gpuDistancesSq ;
	int * keyPtr2 = NULL ;
	int * valPtr1 = NULL ;
	int * valPtr2 = NULL ;
	cudaMalloc((void**) &valPtr1, sizeof(int)*(numCentroids+padding)) ;
	cudaMalloc((void**) &keyPtr2, sizeof(int)*(numCentroids+padding)) ;
	cudaMalloc((void**) &valPtr2, sizeof(int)*(numCentroids+padding)) ;
	cudaMemcpy(valPtr1+padding,centroidIDs,sizeof(int)*numCentroids,cudaMemcpyDeviceToDevice) ;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) ;

	dim3 reductionBlock(NUM_THREADS_PER_BLOCK) ;
	dim3 reductionGrid(NUM_BLOCKS) ;

	int * blockReduceArray = NULL ;
	cudaMalloc((void**) &blockReduceArray, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;
	cudaMemset(blockReduceArray, 0, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;

	cudaEventRecord(start, 0) ;

	// Launch sorting kernels for each bit group. //
	for (int i = 0 ; i < 30-RADIXWIDTH ; i+=RADIXWIDTH) {
		upsweepReduce<<<reductionGrid,reductionBlock>>>(keyPtr1,blockReduceArray+1,numCentroids+padding,i,numTiles) ;
		topLevelScan<<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
		downsweepScan<<<reductionGrid,reductionBlock>>>(keyPtr1, keyPtr2, valPtr1, valPtr2, blockReduceArray,
				numCentroids+padding, i, numTiles) ;
		std::swap(keyPtr1,keyPtr2) ;
		std::swap(valPtr1,valPtr2) ;
	}
	// Final kernel does not need to copy keys. //
	upsweepReduce<<<reductionGrid,reductionBlock>>>(keyPtr1,blockReduceArray+1,numCentroids+padding,30-RADIXWIDTH,numTiles) ;
	topLevelScan<<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
	downsweepScanFinal<<<reductionGrid,reductionBlock>>>(keyPtr1, valPtr1, valPtr2, blockReduceArray,
			numCentroids+padding, 30-RADIXWIDTH, numTiles) ;
	std::swap(keyPtr1,keyPtr2) ;
	std::swap(valPtr1,valPtr2) ;

	
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float sortTime ;
	cudaEventElapsedTime(&sortTime , start, stop) ;

	cudaEventRecord(start, 0) ;

	cudaMemcpy(centroidIDsVec.data(), valPtr1+padding, sizeof(int)*numCentroids, cudaMemcpyDeviceToHost) ;

	// Free temporary memory. //
	cudaFree(blockReduceArray) ;
	cudaFree(valPtr1) ;
	cudaFree(keyPtr1) ;
	cudaFree(valPtr2) ;
	cudaFree(keyPtr2) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;

	float copyTime ;
	cudaEventElapsedTime(&copyTime , start, stop) ;

	times.push_back(sortTime/(1E3)) ;
	times.push_back((sortTime+transformTime)/1E3) ;
	times.push_back((sortTime+transformTime+copyTime)/1E3) ;
}

/* 
 * ===  MEMBER FUNCTION : ImplRadixSort ===============================================
 *         Name:  deAllocate
 *    Arguments:  float * centroidPos - Centroid position location.
 *                int * centroidIDs - Centroid ids location.
 *  Description:  Frees data sotred at pointers.
 * =====================================================================================
 */

void ImplRadixSort::deAllocate(float * centroidPos, int * centroidIDs) {
	cudaFree(centroidPos) ;
	cudaFree(centroidIDs) ;
}

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

static __global__ void upsweepReduce(const int * __restrict__ keys, int * blockOffsetsArray, const int numElements, int digitPos, const int numTiles) {

	// Calculate offset and if block is empty terminate. //
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

	// Set tile totals to zero. //
	if (threadIdx.x < RADIXSIZE) {
		tileTotal[threadIdx.x] = 0 ;
	}
	
	// Process tiles in serial. //
	for (int k = 0 ; k < numTiles ; ++k) {
		if (globalOffset < numElements) {
			numFlags = 0  ;
			// Decode keys. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int digit = (keys[globalOffset+i*WARPSIZE] >> digitPos) & RADIXMASK ;
				numFlags += (1 << PARA_BIT_SIZE*digit) ;
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
					int temp = (numFlags >> (i * PARA_BIT_SIZE)) & PARA_BIT_MASK ;
					#pragma unroll
					for (int j = WARPSIZE_HALF ; j > 0 ; j>>=1) {
						temp += __shfl_down(temp,j) ;
					}
					if (threadIdx.x == 0) {
						tileTotal[i] += temp ;
					}
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

static __global__ void topLevelScan(const int * __restrict__ input, int * output) {
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int threadID = threadIdx.x ;
	int warpID = threadID/WARPSIZE ;
	int offset = blockIdx.x*blockDim.x*RADIXSIZE + warpID*WARPSIZE*RADIXSIZE ;

	// Reads spaced by WARPSIZE incrementes in order to promote read coalescing. //
	int localVal1 = input[offset+laneID] ;
	int localVal2 = input[offset+laneID+32] ;
	int localVal3 = input[offset+laneID+64] ;
	int localVal4 = input[offset+laneID+96] ;

	// Perform 4 manually unrolled warp reductions. //
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

	// Prefix sum values across single warp. //
	localVal2 += __shfl(localVal1,WARPSIZE_MIN_1) ;
	localVal3 += __shfl(localVal2,WARPSIZE_MIN_1) ;
	localVal4 += __shfl(localVal3,WARPSIZE_MIN_1) ;

	 __shared__ int warpReduceVals[NUM_BLOCKS/WARPSIZE] ;

	 // Output warp reduction to shared memory. //
	if (laneID == WARPSIZE_MIN_1) {
		warpReduceVals[warpID] = localVal4 ;
	}
	__syncthreads() ;


	// Interwarp prefix summation on a single warp. //
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

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  downsweepScan
 *    Arguments:  const int * keysIn - Array of keys to be read in.
 *                int * keysOut - Array to scatter keys into.
 *                const int * valuesIn - Array of values to be read in.
 *                int * valuesOut - Array to scatter values into.
 *                const int * blockOffsetsArray - Array containing digit block offsets.
 *                int numElements - Number of keys.
 *                int digitPos - The digit position to decode.
 *                const int numTiles - Number of tiles to process in serial.
 *  Description:  Performs a prefix scan of the digit flag vectors and adds the block
 *                offsets calculated in the top level scan. Does an interal shared memory
 *                shuffle to boost coalesced writing to global memory.
 * =====================================================================================
 */

static __global__ void downsweepScan(const int * __restrict__ keysIn, int * keysOut, 
		const int * __restrict__ valuesIn, int * valuesOut, const int * __restrict__ blockOffsetsArray, 
		const int numElements, int digitPos, const int numTiles) {

	// If block is empty terminate. //
	int blockOffset = (NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD * blockIdx.x) * numTiles ;
	if (blockOffset >= numElements) {
		return ;
	}

	// Declarations. //
	__shared__ int reductionStorage[NUM_WARPS*RADIXSIZE] ;
	__shared__ int digitTotals[RADIXSIZE] ;
	__shared__ int digitOffsets[RADIXSIZE] ;
	__shared__ int globalOffsets[RADIXSIZE] ;
	__shared__ int keysInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;
	//__shared__ int valuesInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;


	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int globalOffset = blockOffset  + laneID + warpID * WARPSIZE * NUM_KEYS_PER_THREAD ;
	int currentKey[NUM_KEYS_PER_THREAD] ;
	int numFlags[NUM_KEYS_PER_THREAD] ;
	int digit[NUM_KEYS_PER_THREAD] ;

	if (threadIdx.x < RADIXSIZE) {
		globalOffsets[threadIdx.x] = blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] ;
	}

	__syncthreads() ;
	

	// Early exit routine. //
	for (int i = 0 ; i < RADIXSIZE-2 ; ++i) {
		if (globalOffsets[i+1] == numElements) {
			if (globalOffsets[i+1] - globalOffsets[i] + blockOffset == numElements) {
				for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
					keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
					valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
				}
				return ;
			}
		}
	}
	int temp = globalOffsets[RADIXSIZE-1] ;
	if (temp == blockOffset){
		for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
			keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
			valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
		}
		return ;
	} else if (temp == numElements) {
		if (temp - globalOffsets[RADIXSIZE-2] + blockOffset == numElements) {
			for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
				keysOut[globalOffset+ j*WARPSIZE] = keysIn[globalOffset+ j*WARPSIZE] ;
				valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
			}
			return ;
		}
	}

	// Process each tile sequentially. //
	for (int k = 0 ; k < numTiles ; ++k) {
		if (globalOffset < numElements) {
			// Load and decode keys plus store in shared memory. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				currentKey[i] = keysIn[globalOffset+i*WARPSIZE] ;
				digit[i] = (currentKey[i]>>digitPos) & RADIXMASK ;
				numFlags[i] = (1 << PARA_BIT_SIZE*digit[i]) ;
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
				reductionStorage[warpID] = numFlags[NUM_KEYS_PER_THREAD-1] ;
			}
			__syncthreads() ;

			// Do a final interwarp prefix sum. //
			temp = 0 ;
			if (warpID == 0) {
				if (laneID < NUM_WARPS) {
					int temp2 = reductionStorage[laneID] ;
					// Load warp digit totals. //
					#pragma unroll
					for (int i = 0 ; i < RADIXSIZE ; ++i) {
						temp = (temp2 >> i*PARA_BIT_SIZE) & PARA_BIT_MASK ;
						temp = warpIncPrefixSum(temp, laneID, NUM_WARPS) ;
						// Save digit totals for the block. //
						if (laneID == (NUM_WARPS - 1)) {
							digitTotals[i] = temp  ;
						}
						reductionStorage[i*NUM_WARPS+laneID] = temp ;
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
				int digOffset = (digit[i] == 0 ? 0 : digitOffsets[digit[i]-1]) ;
				int warpOffset = (warpID == 0 ? 0:reductionStorage[(digit[i]*NUM_WARPS+warpID-1)]) ;
				int localFlag = (numFlags[i] >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK ;
				newOffset[i]  = localFlag + digOffset + warpOffset - 1 ;
				keysInShr[newOffset[i]] = currentKey[i] ;
			}

			__syncthreads() ;

			int globalWriteLoc[4] ;
			// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				int currentKey = keysInShr[readLoc] ;
				int newDigit = (currentKey >>digitPos) & RADIXMASK ;
				int writeOffset = (newDigit == 0 ? 0 : digitOffsets[newDigit-1]) ;
				globalWriteLoc[i] = globalOffsets[newDigit] + readLoc - writeOffset ;
				keysOut[globalWriteLoc[i]] = currentKey ;
			}

			
			__syncthreads() ;

			// Shuffle keys and values in shared memory per tile. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				keysInShr[newOffset[i]] = valuesIn[globalOffset+i*WARPSIZE] ;
			}

			__syncthreads() ;

			// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				valuesOut[globalWriteLoc[i]] = keysInShr[readLoc] ;
			}
		
			// Add tile offsets to 
			if (threadIdx.x < RADIXSIZE) {
				globalOffsets[threadIdx.x] += (digitTotals[threadIdx.x]) ;
			} 
		} else {
			return ;
		}

		globalOffset += NUM_THREADS_PER_BLOCK*NUM_KEYS_PER_THREAD ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  downsweepScanFinal
 *    Arguments:  const int * keysIn - Array of keys to be read in.
 *                int * keysOut - Array to scatter keys into.
 *                const int * valuesIn - Array of values to be read in.
 *                int * valuesOut - Array to scatter values into.
 *                const int * blockOffsetsArray - Array containing digit block offsets.
 *                int numElements - Number of keys.
 *                int digitPos - The digit position to decode.
 *                const int numTiles - Number of tiles to process in serial.
 *  Description:  Performs a prefix scan of the digit flag vectors and adds the block
 *                offsets calculated in the top level scan. Does an interal shared memory
 *                shuffle to boost coalesced writing to global memory. Avoids writing
 *                keys as it's the final kernel.
 * =====================================================================================
 */

static __global__ void downsweepScanFinal(const int * __restrict__ keysIn, const int * __restrict__ valuesIn, 
		int * valuesOut, const int * __restrict__ blockOffsetsArray, const int numElements, 
		int digitPos, const int numTiles) {

	// If the block is empty terminate. //
	int blockOffset = (NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD * blockIdx.x) * numTiles ;
	if (blockOffset >= numElements) {
		return ;
	}


	// Declarations. //
	__shared__ int reductionStorage[NUM_WARPS*RADIXSIZE] ;
	__shared__ int digitTotals[RADIXSIZE] ;
	__shared__ int digitOffsets[RADIXSIZE] ;
	__shared__ int globalOffsets[RADIXSIZE] ;
	__shared__ int keysInShr[NUM_THREADS_PER_BLOCK * NUM_KEYS_PER_THREAD] ;

	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int globalOffset = blockOffset  + laneID + warpID * WARPSIZE * NUM_KEYS_PER_THREAD ;
	int currentKey[NUM_KEYS_PER_THREAD] ;
	int numFlags[NUM_KEYS_PER_THREAD] ;
	int digit[NUM_KEYS_PER_THREAD] ;

	if (threadIdx.x < RADIXSIZE) {
		globalOffsets[threadIdx.x] = blockOffsetsArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] ;
	}

	__syncthreads() ;
	
	// Early exit. //
	for (int i = 0 ; i < RADIXSIZE-2 ; ++i) {
		if (globalOffsets[i+1] == numElements) {
			if (globalOffsets[i+1] - globalOffsets[i] + blockOffset == numElements) {
				for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
					valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
				}
				return ;
			}
		}
	}
	int temp = globalOffsets[RADIXSIZE-1] ;
	if (temp == blockOffset){
		for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
			valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
		}
		return ;
	} else if (temp == numElements) {
		if (temp - globalOffsets[RADIXSIZE-2] + blockOffset == numElements) {
			for (int j = 0 ; j < NUM_KEYS_PER_THREAD ; ++j) {
				valuesOut[globalOffset+ j*WARPSIZE] = valuesIn[globalOffset+ j*WARPSIZE] ;
			}
			return ;
		}
	}

	// Process each tile sequentially. //
	for (int k = 0 ; k < numTiles ; ++k) {
		if (globalOffset < numElements) {
			// Load and decode keys plus store in shared memory. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				currentKey[i] = keysIn[globalOffset+i*WARPSIZE] ;
				digit[i] = (currentKey[i]>>digitPos) & RADIXMASK ;
				numFlags[i] = (1 << PARA_BIT_SIZE*digit[i]) ;
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
				reductionStorage[warpID] = numFlags[NUM_KEYS_PER_THREAD-1] ;
			}
			__syncthreads() ;

			// Do a final interwarp prefix sum. //
			temp = 0 ;
			if (warpID == 0) {
				if (laneID < NUM_WARPS) {
					int temp2 = reductionStorage[laneID] ;
					// Load warp digit totals. //
					#pragma unroll
					for (int i = 0 ; i < RADIXSIZE ; ++i) {
						temp = (temp2 >> i*PARA_BIT_SIZE) & PARA_BIT_MASK ;
						temp = warpIncPrefixSum(temp, laneID, NUM_WARPS) ;
						// Save digit totals for the block. //
						if (laneID == (NUM_WARPS - 1)) {
							digitTotals[i] = temp  ;
						}
						reductionStorage[i*NUM_WARPS+laneID] = temp ;
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

			// Shuffle keys in shared memory per tile. Keys and values shuffled seperately to save memory.  //
			int newOffset[4] ;
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int digOffset = (digit[i] == 0 ? 0 : digitOffsets[digit[i]-1]) ;
				int warpOffset = (warpID == 0 ? 0:reductionStorage[(digit[i]*NUM_WARPS+warpID-1)]) ;
				int localFlag = (numFlags[i] >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK ;
				newOffset[i]  = localFlag + digOffset + warpOffset - 1 ;
				keysInShr[newOffset[i]] = currentKey[i] ;
			}

			__syncthreads() ;

			int globalWriteLoc[4] ;
			// Calculate global write position for values. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				int currentKey = keysInShr[readLoc] ;
				int newDigit = (currentKey >>digitPos) & RADIXMASK ;
				int writeOffset = (newDigit == 0 ? 0 : digitOffsets[newDigit-1]) ;
				globalWriteLoc[i] = globalOffsets[newDigit] + readLoc - writeOffset ;
			}
			
			__syncthreads() ;

			// Shuffle values into shared memory per tile. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				keysInShr[newOffset[i]] = valuesIn[globalOffset+i*WARPSIZE] ;
			}

			__syncthreads() ;

			// Scatter values based on global write location. //
			#pragma unroll
			for (int i = 0 ; i < NUM_KEYS_PER_THREAD ; ++i) {
				int readLoc = laneID+i*WARPSIZE + warpID*WARPSIZE*NUM_KEYS_PER_THREAD ;
				valuesOut[globalWriteLoc[i]] = keysInShr[readLoc] ;
			}
		
			// Add tile digit counts to globalOffset. //
			if (threadIdx.x < RADIXSIZE) {
				globalOffsets[threadIdx.x] += (digitTotals[threadIdx.x]) ;
			} 
		} else {
			return ;
		}

		globalOffset += NUM_THREADS_PER_BLOCK*NUM_KEYS_PER_THREAD ;
	}
}

