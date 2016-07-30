/*
 * =====================================================================================
 *
 *       Filename:  merrel_radix_gpu_sort.cu
 *
 *    Description:  CUDA code for shared memory based radix sort on gpu.
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

#include <stdio.h>
#include <cmath>
#include <math.h>

#include "../../inc/cpp_inc/merrel_radix_gpu_sort_funcs.hpp"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../inc/cu_inc/test_utils.cuh"
#include "../../inc/cu_inc/prefix_sums.cuh"

#define WARPSIZE 32
#define WARPSIZE_HALF 16
#define WARPSIZE_MIN_1 31
#define RADIXSIZE 4
#define RADIXMASK 3


#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 128
#define NUM_TILES_PER_BLOCK 1
#define NUM_WARPS NUM_THREADS_PER_BLOCK/WARPSIZE

#define VEC_SIZE 4
#define PARA_BIT_SIZE 8
#define NUM_BIT_PARTITIONS 32/PARA_BIT_SIZE
#define PARA_BIT_MASK_0 (1 << PARA_BIT_SIZE)-1
#define PARA_BIT_MASK_1 PARA_BIT_MASK_0 << PARA_BIT_SIZE
#define PARA_BIT_MASK_2 PARA_BIT_MASK_1 << PARA_BIT_SIZE
#define PARA_BIT_MASK_3 PARA_BIT_MASK_2 << PARA_BIT_SIZE
#define SUB_TILE_SIZE (128 / VEC_SIZE)
#define SUB_TILE_SIZE_MIN_1 (SUB_TILE_SIZE-1)
#define NUM_WARPS_PER_SUB_TILE SUB_TILE_SIZE / WARPSIZE
#define NUM_SUB_TILES (NUM_THREADS_PER_BLOCK / SUB_TILE_SIZE)

typedef union {
	int4 vec ;
	int a[4] ;
} U4 ;

typedef union {
	int2 vec ;
	int a[2] ;
} U2 ;


#define NUM_THREADS_REDUCE NUM_BLOCKS

static __device__ inline void printBinary(int num) {
	int mask = 1 ;
	for (int i = 0 ; i < sizeof(num)*8 ; ++i) {
		printf("%d", (num>>(31-i) & mask)) ;
	}
	printf("\n");
}


static __global__ void upsweepReduce(const int * __restrict__ keys, int * reduceArray, const int numElements, const int digitPos) {
	// Declarations. //
	__shared__ int warpVals[NUM_WARPS] ;
	__shared__ int subTileVals[NUM_SUB_TILES] ;

	// Get IDs. //
	int warpID  = threadIdx.x / WARPSIZE ;
	int laneID = threadIdx.x & WARPSIZE_MIN_1 ;
	int subTileID = threadIdx.x / SUB_TILE_SIZE ;
	int subLaneID = threadIdx.x & SUB_TILE_SIZE_MIN_1 ;
	int subWarpID = warpID % (NUM_WARPS_PER_SUB_TILE) ;
	__shared__ int tileTotal[RADIXSIZE] ;
	int blockOffset = (NUM_THREADS_PER_BLOCK * blockIdx.x) * NUM_TILES_PER_BLOCK ;
	int globalOffset = blockOffset + threadIdx.x ;
	// Allocate array containing flag reductions. //
	int numFlags ;

	if (threadIdx.x < RADIXSIZE) {
		tileTotal[threadIdx.x] = 0 ;
	}
	
	#pragma unroll
	for (int k = 0 ; k < NUM_TILES_PER_BLOCK ; ++k) {
		numFlags = 0 ;
		// Decode keys. //
		if ((VEC_SIZE*globalOffset) < numElements) {
			U4 keyVec ;
			keyVec.vec = reinterpret_cast<const int4*>(keys)[globalOffset] ;
			#pragma unroll
			for (int i = 0 ; i < VEC_SIZE ; ++i) {
				int digit = (keyVec.a[i]>>digitPos) & RADIXMASK ;
				numFlags += (1 << PARA_BIT_SIZE*digit) ;
			}
		}

		__syncthreads() ;
		// Warp level reduction. //
		#pragma unroll
		for (int j = WARPSIZE_HALF ; j > 0 ; j>>=1) {
			numFlags += __shfl_down(numFlags,j) ;
		}
		if (laneID == 0) {
			warpVals[warpID] = numFlags ;
		}

		__syncthreads() ;

		// Get data into first warp in subtile. //
		numFlags = (subLaneID < NUM_WARPS_PER_SUB_TILE) ? 
			warpVals[NUM_WARPS_PER_SUB_TILE*subTileID+subLaneID] : 0 ;

		// Final subTile reduction. //
		if (subWarpID == 0) {
			#pragma unroll
			for (int j = NUM_WARPS_PER_SUB_TILE ; j > 0 ; j>>=1) {
				numFlags += __shfl_down(numFlags,j) ;
			}
			if (subLaneID == 0) {
				subTileVals[subTileID] = numFlags ;
			}
		}


		__syncthreads() ;

		// Get data into first warp in subtile. //
		numFlags = (threadIdx.x < NUM_SUB_TILES) ? 
			subTileVals[laneID] : 0 ;
		// Final warp reduction. //
		if (warpID == 0) {
			#pragma unroll
			for (int i = 0 ; i < RADIXSIZE ; ++i) {
				int temp = (numFlags >> i*PARA_BIT_SIZE) & PARA_BIT_MASK_0 ;
				#pragma unroll
				for (int j = NUM_WARPS ; j > 0 ; j>>=1) {
					temp += __shfl_down(temp,j) ;
				}
				if (threadIdx.x == 0) {
					tileTotal[i] += temp ;
				}
			}
		}

		__syncthreads() ;
		
		globalOffset += NUM_THREADS_PER_BLOCK ;
	}

	__syncthreads() ;

	if (threadIdx.x < RADIXSIZE) {
		reduceArray[threadIdx.x*NUM_BLOCKS+blockIdx.x] = tileTotal[threadIdx.x] ;
	}

}

__device__ inline int warpIncPrefixSum(int localVal, int laneID, int numThreads) {
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

__device__ inline int warpExPrefixSum(int localVal, int laneID, int numThreads) {
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


static __global__ void downsweepScan(const int * __restrict__ keysIn, int * keysOut, const int * __restrict__ valuesIn, int * valuesOut, int * reduceArray, const int numElements, const int digitPos) {
	// Declarations. //`
	__shared__ int warpVals[NUM_WARPS] ;
	__shared__ int subTileVals[NUM_SUB_TILES*RADIXSIZE] ;
	__shared__ int digitTotals[RADIXSIZE] ;
	__shared__ int digitOffsets[RADIXSIZE] ;
	__shared__ int seedValues[RADIXSIZE] ;


	__shared__ int keysInShr[NUM_THREADS_PER_BLOCK*VEC_SIZE] ;
	__shared__ int valuesInShr[NUM_THREADS_PER_BLOCK*VEC_SIZE] ;

	// Get IDs. //
	int localOffset = threadIdx.x ;
	int warpID  = localOffset / WARPSIZE ;
	int laneID = localOffset & WARPSIZE_MIN_1 ;
	int subTileID = threadIdx.x / SUB_TILE_SIZE ;
	int subLaneID = laneID ;
	int subWarpID = warpID % (NUM_WARPS_PER_SUB_TILE) ;
	int blockOffset = (NUM_THREADS_PER_BLOCK * blockIdx.x) * NUM_TILES_PER_BLOCK ;
	int globalOffset = blockOffset + threadIdx.x ;
	int digit[VEC_SIZE] ;
	int localFlags[VEC_SIZE] ;
	int orig ; 
	int numFlags ;
	U4 keyVec ;
	U4 valVec ;

	if (localOffset < RADIXSIZE) {
		seedValues[localOffset] = reduceArray[localOffset*NUM_BLOCKS+blockIdx.x] ;
	}

	__syncthreads() ;

	// Early exit. //
	#pragma unroll
	for (int i = 0 ; i < RADIXSIZE-1 ; ++i) {
		if (seedValues[i+1] - seedValues[i] + blockOffset == numElements) {
			return ;
		}
	}
	if (seedValues[RADIXSIZE-1] - blockOffset == 0) {
		return ;
	}


	// Process each tile sequentially. //
	#pragma unroll
	for (int k = 0 ; k < NUM_TILES_PER_BLOCK ; ++k) {

		numFlags = 0 ;
		
		// Load and decode keys plus store in shared memory. //
		if ((VEC_SIZE*globalOffset) < numElements) {
			keyVec.vec = reinterpret_cast<const int4*>(keysIn)[globalOffset] ;
			valVec.vec = reinterpret_cast<const int4*>(valuesIn)[globalOffset] ;
			#pragma unroll
			for (int i = 0 ; i < VEC_SIZE ; ++i) {
				digit[i] = (keyVec.a[i]>>digitPos) & RADIXMASK ;
				localFlags[i] = (1 << PARA_BIT_SIZE*digit[i]) ;
			}
			// Serial reduction. //
			#pragma unroll
			for (int i = 1 ; i < VEC_SIZE ; ++i) {
				localFlags[i] += localFlags[i-1] ;
			}
			numFlags = localFlags[VEC_SIZE-1] ;
			orig = numFlags ;


			#pragma unroll
			for (int i = VEC_SIZE-1 ; i >= 1 ; --i) {
				localFlags[i] = localFlags[i-1] ;
			}
			localFlags[0] = 0 ;
	
			#pragma unroll
			for (int i = 0 ; i < VEC_SIZE ; ++i) {
				keysInShr[VEC_SIZE*localOffset+i] = keyVec.a[i] ;
				valuesInShr[VEC_SIZE*localOffset+i] = valVec.a[i] ;
			}
		}



		numFlags = warpIncPrefixSum(numFlags, laneID, WARPSIZE) ;
		// Save warp digit total. //
		if (subLaneID == WARPSIZE_MIN_1) {
			warpVals[warpID] = numFlags ;
		}

		__syncthreads() ;

		// Do a sub Tile prefix sum. //
		int temp = 0 ;
		// Get data into first warp in subtile. //
		// Final subTile reduction. //
		if (subWarpID == 0) {
			if (subLaneID < NUM_WARPS_PER_SUB_TILE) {
				// Load warp digit totals. //
				temp = warpVals[NUM_WARPS_PER_SUB_TILE*subTileID+subLaneID] ;
				temp = warpIncPrefixSum(temp, subLaneID, NUM_WARPS_PER_SUB_TILE) ;
				// Save digit totals for the block. //
				if (subLaneID == (NUM_WARPS_PER_SUB_TILE - 1)) {
					#pragma unroll
					for (int i = 0 ; i < RADIXSIZE ; ++i) {
						subTileVals[i*NUM_SUB_TILES+subTileID] = (temp >> i*PARA_BIT_SIZE) & PARA_BIT_MASK_0 ;
					}
				} 
				// Save the new prefix summed warp totals. //
				warpVals[NUM_WARPS_PER_SUB_TILE*subTileID+subLaneID] = temp ;
			}
		}


		__syncthreads() ;

		temp = (subWarpID == 0 ? 0:warpVals[NUM_WARPS_PER_SUB_TILE*(subTileID)+(subWarpID-1)]) ;
		numFlags += temp ;
		#pragma unroll
		for (int i = 0 ; i < VEC_SIZE ; ++i) {
			localFlags[i] += numFlags ;
		}


		temp = 0 ;
		// Do a final interwarp prefix sum. //
		if (warpID == 0) {
			if (laneID < NUM_SUB_TILES) {
				// Load warp digit totals. //
				#pragma unroll
				for (int i = 0 ; i < RADIXSIZE ; ++i) {
					temp = subTileVals[i*NUM_SUB_TILES+laneID]  ;
					temp = warpIncPrefixSum(temp, laneID, NUM_SUB_TILES) ;
					// Save digit totals for the block. //
					if (laneID == (NUM_SUB_TILES - 1)) {
						digitTotals[i] = temp ;
					} 
					if (laneID == 0) {
						subTileVals[i*NUM_SUB_TILES+laneID] = 0 ;
					}
					if (laneID < NUM_SUB_TILES-1) {
						subTileVals[i*NUM_SUB_TILES+laneID+1] = temp ;
					}
				}
			}
		}

		__syncthreads() ;
			// Scan block digit totals. //
		temp = 0 ;
		if (warpID == 0) {
			if (laneID < RADIXSIZE) {
				// Load warp digit totals. //
				temp = digitTotals[laneID] ;
				temp = warpIncPrefixSum(temp, laneID, RADIXSIZE) ;
				// Save the new prefix summed warp digit totals. //
				digitOffsets[laneID] = temp ;
			}
		}

		__syncthreads() ;


		// Shuffle keys and values in shared memory per tile. //
		if (VEC_SIZE*globalOffset < numElements) {
			#pragma unroll
			for (int i = 0 ; i < VEC_SIZE ; ++i) {
				int digOffset = (digit[i] == 0 ? 0 : digitOffsets[digit[i]-1]) ;
				int localPos = (localFlags[i] >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK_0 ;
				int origOff = (orig >> (digit[i]*PARA_BIT_SIZE)) & PARA_BIT_MASK_0 ;
				int subTileOffset = subTileVals[digit[i]*NUM_SUB_TILES+subTileID] ;
				int newOffset  = localPos - origOff + digOffset + subTileOffset ;
				keysInShr[newOffset] = keyVec.a[i] ;
				valuesInShr[newOffset] = valVec.a[i] ;
			}
		}

		__syncthreads() ;

		
		/*
		
		if (threadIdx.x == 0) {
			int prev = (keysInShr[0] >> digitPos) & RADIXMASK ;
			for (int i = 1 ; i < NUM_THREADS_PER_BLOCK * VEC_SIZE ; ++i) {
				int cur = (keysInShr[i] >> digitPos) & RADIXMASK ;
				if (prev > cur) {
					if (blockIdx.x == 0) {
						printf("%d %d %d\n", k);
					}
					 // printf("%d %d %d %d %d\n", blockIdx.x, k, i, prev, cur);
				}
				prev = cur ;
			}
		} 
 */
		/*
		if (threadIdx.x == 0 && blockIdx.x == 0 && k == 0) {
			for (int i = 0 ; i < VEC_SIZE * NUM_THREADS_PER_BLOCK ; ++i) {
				printf("%d : %d : %d\n",i, (keysInShr[i]>>digitPos) & RADIXMASK, digitPos);
			}
		} */

	//	__syncthreads() ;


		// Scatter keys based on local prefix sum, tile prefix sum and block prefix sum. //
		if (VEC_SIZE*globalOffset < numElements) {
			#pragma unroll
			for (int i = 0 ; i < VEC_SIZE ; ++i) {
				keyVec.a[i] = keysInShr[VEC_SIZE*localOffset+i] ;
				valVec.a[i] = valuesInShr[VEC_SIZE*localOffset+i] ;
				int newDigit = (keyVec.a[i] >> digitPos) & RADIXMASK ;
				int writeOffset = (newDigit == 0 ? 0 : digitOffsets[newDigit-1]) ;
				int globalWriteLoc = seedValues[newDigit] + VEC_SIZE*localOffset + i - writeOffset ;
				keysOut[globalWriteLoc] = keyVec.a[i] ;
				valuesOut[globalWriteLoc] = valVec.a[i] ;
			}
		}

		__syncthreads() ;

		if (threadIdx.x < RADIXSIZE) {
			seedValues[threadIdx.x] += digitTotals[threadIdx.x] ;
		}


		globalOffset += NUM_THREADS_PER_BLOCK ;
		
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  intraWarpScan
 *    Arguments:  int * digitFlags - Array storing the digit flags.
 *  Description:  Performs a prefix scan using warp shuffles. This has the benefit of
 *                not requiring thread synchronisation. Unrolled for RADIXSIZE of 4.
 *                Taken from paper on light prefix sum.
 * =====================================================================================
 */

static __global__ void intraWarpScan(int * input, int * output) {
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

}		/* -----  end of function intraWarpScan  ----- */

static __global__ void printTopArray(int * topArray, int size) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;

	if (globalID == 0) {
		for (int i = 0 ; i < size ; ++i) {
			printf(" T : %d\n", topArray[i]);
		}
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sort
 *    Arguments:  int * keys - Keys array.
 *                int * values - Values array.
 *                const int numElements - Number of elements in keys.
 *  Description:  Sorts the key and values array. Uses a global radix sort.
 * =====================================================================================
 */

static void sort(int * keys, int * values, const int numElements) {

	//const int numKeysPerThread = (numElements/(NUM_THREADS_PER_BLOCK * NUM_BLOCKS)) + 
	//	((numElements % (NUM_THREADS_PER_BLOCK * NUM_BLOCKS)) == 0 ? 0 : 1) ;

	int * blockReduceArray = NULL ;
	cudaMalloc((void**) &blockReduceArray, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;
	cudaMemset(blockReduceArray, 0, sizeof(int)*(NUM_BLOCKS*RADIXSIZE+1)) ;

	dim3 reductionBlock(NUM_THREADS_PER_BLOCK) ;
	dim3 reductionGrid(NUM_BLOCKS) ;



	// Create buffer for keys and values. //
	int * keyPtr1 = keys ;
	int * keyPtr2 = NULL ;
	int * valPtr1 = values ;
	int * valPtr2 = NULL ;

	cudaMalloc((void**) &keyPtr2, sizeof(int)*numElements) ;
	cudaMalloc((void**) &valPtr2, sizeof(int)*numElements) ;

	// Give threads more shmem. //
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) ;

	for (int i = 0 ; i < 30; i+=2) {
		upsweepReduce<<<reductionGrid,reductionBlock>>>(keyPtr1,blockReduceArray+1,numElements,i) ;
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		intraWarpScan<<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		downsweepScan<<<reductionGrid,reductionBlock>>>(keyPtr1, keyPtr2, valPtr1, valPtr2, blockReduceArray, numElements, i) ;
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		std::swap(keyPtr1,keyPtr2) ;
		std::swap(valPtr1,valPtr2) ;
	}


	if (values != valPtr1) {
		cudaMemcpy(values, valPtr1, sizeof(int)*numElements, cudaMemcpyDeviceToDevice) ;
		cudaFree(valPtr1) ;
	} else {
		cudaFree(valPtr2) ;
	}
	if (keys != keyPtr1) {
		cudaMemcpy(keys, keyPtr1, sizeof(int)*numElements, cudaMemcpyDeviceToDevice) ;
		cudaFree(keyPtr1) ;
	} else {
		cudaFree(keyPtr2) ;
	}

	cudaFree(blockReduceArray) ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaBasicRadixSortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Triangles to be sorted.
 *                std::vector<Camera> & cameras - Cameras to be sorted relative to.
 *  Description:  Uses a basic global implementation of radix sort on the GPU to sort
 *                the triangles relative to the cameras.
 * =====================================================================================
 */

void cudaMerrelRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	// Vectorise Triangle data. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;

	const int numTriangles = triangles.size() ;
	const int numCameras = cameras.size() ;

	for (int i = 0 ; i < numTriangles ; ++i) {
		triIds[i] = i ;
		const float * coords = triangles[i].getCoords() ;
		triCo[i] = coords[0] ;
		triCo[i+triangles.size()] = coords[1] ;
		triCo[i+2*triangles.size()] = coords[2] ;
	}

	// Vectorise camera co-ordinates. //
	std::vector<float> camCo(3*cameras.size()) ;
	for (int i = 0 ; i < numCameras ; ++i) {
		const float * coords = cameras[i].getCoords() ;
		camCo[i] = coords[0] ;
		camCo[i+cameras.size()] = coords[1] ;
		camCo[i+2*cameras.size()] = coords[2] ;
	}

	float * gpuTriCo = NULL ;
	float * gpuCamCo = NULL ;
	float * gpuDistancesSq = NULL ;
	int * gpuTriIds = NULL ;

	cudaMalloc((void **) &gpuTriCo, sizeof(float)*triCo.size()) ;
	cudaMalloc((void **) &gpuCamCo, sizeof(float)*camCo.size()) ;
	cudaMalloc((void **) &gpuTriIds, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*triangles.size()) ;

	cudaMemcpy(gpuTriCo, triCo.data(), sizeof(float)*triCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuCamCo, camCo.data(), sizeof(float)*camCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuTriIds, triIds.data(), sizeof(int)*triIds.size(), cudaMemcpyHostToDevice) ;

	dim3 distanceBlock(WARPSIZE) ;
	dim3 distanceGrid(numTriangles/distanceBlock.x + (!(numTriangles%distanceBlock.x)?0:1)) ;

	for (int i = 0 ; i < numCameras ; ++i) {
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, gpuTriIds, numTriangles, numCameras) ;
		sort((int*)gpuDistancesSq,gpuTriIds,numTriangles) ;
		//reshuffleGPUData<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuTriCoTemp, gpuTriIds, numTriangles) ;
		cudaMemcpy(triIds.data(), gpuTriIds, sizeof(int)*triIds.size(), cudaMemcpyDeviceToHost) ;
		// Read back new indices. //
	}

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}

	// Free gpu data. // 
	cudaFree(gpuTriCo) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuTriIds) ;
	cudaFree(gpuDistancesSq) ;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaBasicRadixSortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles to sort.
 *                std::vector<Camera> & cameras - Vector of cameras to sort relative to.
 *		          std::vector<float> & times - Vector of times used for benchmarking.
 *  Description:  Uses a basic global implementation of radix sort on the GPU to sort
 *                the triangles relative to the cameras. Benchmarks and saves times.
 * =====================================================================================
 */

void cudaMerrelRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {
	// Timing. //
	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	std::vector<float> newTimes ;
	// Vectorise Triangle data. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;

	const int numTriangles = triangles.size() ;
	const int numCameras = cameras.size() ;

	for (int i = 0 ; i < numTriangles ; ++i) {
		triIds[i] = i ;
		const float * coords = triangles[i].getCoords() ;
		triCo[i] = coords[0] ;
		triCo[i+triangles.size()] = coords[1] ;
		triCo[i+2*triangles.size()] = coords[2] ;
	}

	// Vectorise camera co-ordinates. //
	std::vector<float> camCo(3*cameras.size()) ;
	for (int i = 0 ; i < numCameras ; ++i) {
		const float * coords = cameras[i].getCoords() ;
		camCo[i] = coords[0] ;
		camCo[i+cameras.size()] = coords[1] ;
		camCo[i+2*cameras.size()] = coords[2] ;
	}

	float * gpuTriCo = NULL ;
	float * gpuCamCo = NULL ;
	float * gpuDistancesSq = NULL ;
	int * gpuTriIds = NULL ;
	// Memory allocation to GPU. //
	cudaMalloc((void **) &gpuTriCo, sizeof(float)*triCo.size()) ;
	cudaMalloc((void **) &gpuCamCo, sizeof(float)*camCo.size()) ;
	cudaMalloc((void **) &gpuTriIds, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*triangles.size()) ;

	// Memory transfer to GPU. //
	cudaMemcpy(gpuTriCo, triCo.data(), sizeof(float)*triCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuCamCo, camCo.data(), sizeof(float)*camCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuTriIds, triIds.data(), sizeof(int)*triIds.size(), cudaMemcpyHostToDevice) ;

	dim3 distanceBlock(WARPSIZE) ;
	dim3 distanceGrid(numTriangles/distanceBlock.x + (!(numTriangles%distanceBlock.x)?0:1)) ;

	for (int i = 0 ; i < numCameras ; ++i) {
		cudaEventRecord(start, 0) ;
		// Transform triangles to distance vector. //
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, 
				gpuTriIds, numTriangles, numCameras) ;
		cudaEventRecord(stop, 0) ;
		cudaEventSynchronize(stop) ;
		float transformTime ;
		cudaEventElapsedTime(&transformTime , start, stop) ;

		cudaEventRecord(start, 0) ;
		sort((int*)gpuDistancesSq,gpuTriIds,numTriangles) ;
		cudaEventRecord(stop, 0) ;
		cudaEventSynchronize(stop);
		float sortTime ;
		cudaEventElapsedTime(&sortTime , start, stop) ;

		// Read back new indices. //
		cudaEventRecord(start, 0) ;
		cudaMemcpy(triIds.data(), gpuTriIds, sizeof(int)*triIds.size(), cudaMemcpyDeviceToHost) ;
		cudaEventRecord(stop, 0) ;
		cudaEventSynchronize(stop);
		float transferTime ;
		cudaEventElapsedTime(&transferTime , start, stop) ;

		float totalTime = transformTime + transferTime + sortTime ;
		float incTransTime = sortTime + transformTime ;
		float sortOnlyTime = sortTime  ;
		newTimes.push_back(totalTime/1E3) ;
		newTimes.push_back(incTransTime/1E3) ;
		newTimes.push_back(sortOnlyTime/1E3) ;
	}

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}

	times = newTimes ;

	cudaFree(gpuTriCo) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuTriIds) ;
	cudaFree(gpuDistancesSq) ;
}