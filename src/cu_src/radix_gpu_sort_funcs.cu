/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.cu
 *
 *    Description:  CUDA code for radix sort on gpu.
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

#include "../../inc/cpp_inc/radix_gpu_sort_funcs.hpp"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../inc/cu_inc/cuda_transforms.cuh"

#include "../../../cub-1.5.2/cub/cub.cuh"

#define WARPSIZE 32
#define RADIXSIZE 4


__global__ void prefixSum(int * indices, int * globalValues, int * localPrefixSumArray , 
		int * blockSumArray, int numTriangles, int bits) ;
__global__ void scatteredWrite(int * indices, int * globalValues, int * localPrefixSumArray, 
		int * blockSumArray, int numTriangles, int bits) ;

static void cudaRadixSortKernels(float * gpuDistancesSq, int * gpuTriIds, const int numTriangles, 
		dim3 & distanceBlock, dim3 & distanceGrid) {
	int * blockSumArray = NULL ;
	int * localPrefixSumArray = NULL ;
	cudaMalloc((void **) &blockSumArray, sizeof(int)*distanceGrid.x*RADIXSIZE) ;
	cudaMalloc((void **) &localPrefixSumArray, sizeof(int)*distanceGrid.x*distanceBlock.x) ;
	const int memRequired = distanceBlock.x*sizeof(int)*7 ;
	for (int i = 0 ; i < 2 ; i+=2) {
		prefixSum<<<distanceGrid,distanceBlock,memRequired>>>(gpuTriIds, (int*)gpuDistancesSq, localPrefixSumArray, blockSumArray, numTriangles, i) ;
		// Determine temporary device storage requirements
		int num_items = RADIXSIZE * distanceGrid.x ;
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockSumArray, blockSumArray, num_items);

		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockSumArray, blockSumArray, num_items);

		cudaFree(d_temp_storage) ;
		scatteredWrite<<<distanceGrid,distanceBlock,3*distanceBlock.x*sizeof(int)>>>(gpuTriIds, (int*)gpuDistancesSq, localPrefixSumArray, blockSumArray, numTriangles, i) ;
		gpuErrchk(cudaPeekAtLastError()) ;
		gpuErrchk(cudaDeviceSynchronize()) ;
	}
}

void cudaRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	
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
		cudaRadixSortKernels(gpuDistancesSq,gpuTriIds,numTriangles,distanceBlock,distanceGrid) ;
	}

	// Read back new indices. //
	cudaMemcpy(triIds.data(), gpuTriIds, sizeof(int)*triIds.size(), cudaMemcpyDeviceToHost) ;

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}
}


void cudaRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {
	;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  prefixSum
 *    Arguments:  int * indices - Indices array.
 *                float * values - Values array.
 *  Description:  Generates 2^d radix bitmasks and calculates their prefix sum for a
 *                given block of indices and values. 
 * =====================================================================================
 */

__global__ void prefixSum(int * indices, int * globalValues, int * localPrefixSumArray ,  int * blockSumArray, int numTriangles, int bits) {

	extern __shared__ int compositeArray[] ;
	int * count = &compositeArray[0] ;
	int * blockValues = &compositeArray[RADIXSIZE*blockDim.x] ;
	int * blockIndices = &compositeArray[(RADIXSIZE+1)*blockDim.x] ;
	int * localPrefixSum = &compositeArray[(RADIXSIZE+2)*blockDim.x] ;

	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	int threadID = threadIdx.x ;


	// Initialise shared mem to zero. //
	for (int i = 0 ; i < RADIXSIZE ; ++i) {
		count[RADIXSIZE*threadID+i] = 0 ;
	}

	// Prefix sum. //
	int digit =  0 ;
	if (globalID < numTriangles) {
		blockValues[threadID] = globalValues[globalID] ;
		blockIndices[threadID] = indices[globalID] ;
		digit = ((blockValues[threadID] >> bits) & 3) ;
		count[RADIXSIZE*threadID+digit] = 1 ;
	} else {
		blockValues[threadID] = 0 ;
		blockIndices[threadID] = 0 ;
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

	
	__syncthreads() ;

	if (threadID < RADIXSIZE) { 
		count[RADIXSIZE*(blockDim.x-1)+threadID] = 0 ;
	} // clear the last element  

	__syncthreads() ;

	if (threadID % 2 == 0) {
		for (int i = 1 ; i < blockDim.x ; i*=2) {
			offset >>= 1 ;
			__syncthreads() ;
			if (threadID < 2*i) {
				int a = offset*(threadID+1)-1;  
				int b = offset*(threadID+2)-1;  
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

	localPrefixSum[threadID] = count[RADIXSIZE*threadID+digit] ;
	blockSumArray[threadID*gridDim.x+blockIdx.x] = count[RADIXSIZE*(blockDim.x-1)+threadID] ;

	__syncthreads() ;

	

	// Offset counts such that radix val 01 is offset by the max val of count 00. //
	for (int i = 1 ; i < RADIXSIZE ; ++i) {
		count[RADIXSIZE*threadID+i] += count[RADIXSIZE*(blockDim.x-1)+(i-1)] ;
	}
	__syncthreads() ;

	if (globalID < numTriangles) {
		// Shuffle. //
		int tempv = blockValues[threadID] ;
		int tempi = blockIndices[threadID] ;
		int tempLocalPrefixSum = localPrefixSum[threadID] ;
		__syncthreads() ;
		int newPos = count[RADIXSIZE*threadID+digit] ;
		blockValues[newPos] = tempv ;
		blockIndices[newPos] = tempi ;
		localPrefixSum[newPos] = tempLocalPrefixSum ;
		__syncthreads() ;
		globalValues[globalID] = blockValues[threadID] ;
		indices[globalID] = blockIndices[threadID] ;
		localPrefixSumArray[globalID] = localPrefixSum[threadID] ;
	}
}

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

