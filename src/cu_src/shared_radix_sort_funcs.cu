/*
 * =====================================================================================
 *
 *       Filename:  shared_radix_gpu_sort.cu
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

#include "../../inc/cpp_inc/shared_radix_gpu_sort_funcs.hpp"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../inc/cu_inc/test_utils.cuh"
#include "../../inc/cu_inc/prefix_sums.cuh"

#define WARPSIZE 32
#define NUMTHREADSDEC 128
#define NUMTHREADSRED 512
#define NUMTHREADSBS 1024
#define RADIXSIZE 4
#define RADIXMASK 3

#define NUM_BANKS WARPSIZE/2
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  flagDecode
 *    Arguments:  int * keys - Keys array.
 *                int * digitFlags - Array storing the digit flags.
 *                const int numElements - Number of elements in keys.
 *                int bitOffset - The offset of the bits to be decoded. Right to left.
 *  Description:  Generates the flag vectors for the given bits. 
 * =====================================================================================
 */

static __global__ void flagDecode(int * keys, int * digitFlags, const int numElements, int bitOffset) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID < numElements) {
		// Decode digit. //
		int digitVal = (keys[globalID]>>bitOffset) & (RADIXMASK) ;
		digitFlags[digitVal*numElements+globalID] = 1 ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  calcBlockSumArray
 *    Arguments:  int * localSumArray - The array containing local unsummed values. (Per
 *                block.
 *                int * blockSumArray - The array containing the reduced sum of all the
 *                per block values. Each element is for its corresponding block.
 *                int numThreadsReq - The number of threads required to process array
 *                globally.
 *  Description:  Fused local prefix sum and global block sum array kernels.
 * =====================================================================================
 */

__global__ void calcBlockSumArray(int * localSumArray, int * blockSumArray) {

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
		blockSumArray[blockIdx.x] = sharedSum[n-1+CONFLICT_FREE_OFFSET(n-1)] ;
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
 *         Name:  updateLocalPrefixes
 *    Arguments:  int * localSumArray - Array of local sums 
 *                int * blockSumArray - Array containing the block offset.
 *                int numElements - Number of elements in array.
 *  Description:  Add corresponding block offset to each member of the block.
 * =====================================================================================
 */

static void __global__ updateLocalPrefixes(int * localSumArray, int * blockSumArray, int numElements) {
	int threadID = threadIdx.x ;
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	__shared__ int blockOffset ;

	if (2*globalID < numElements) {
		if (threadID == 0) {
			blockOffset = blockSumArray[blockIdx.x] ;
		}
		__syncthreads() ;
		localSumArray[2*globalID] += blockOffset ;
		localSumArray[2*globalID+1] += blockOffset ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  shuffle
 *    Arguments:  int * keyPtrIn - Input keys.
 *                int * keyPtrOut - Ouput keys.
 *                int * valPtrIn - Input values. 
 *                int * valPtrOut - Output values.  
 *                int * digitFlags - Prefix Summed Flag array. 
 *                const int numElements - Number of keys.
 *                int bitOffset - The offset of the bits to be decoded. Right to left.
 *  Description:  Shuffle keys and values to there new sorted positions for the given
 *                bit offset. The new locations are in the digitFlags array.
 * =====================================================================================
 */

static __global__ void shuffle(int * keyPtrIn, int * keyPtrOut, int * valPtrIn, int * valPtrOut, 
		int * digitFlags, const int numElements, int bitOffset) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID < numElements) {
		int key = keyPtrIn[globalID] ;
		// Decode digit. //
		int digitVal = (key >> bitOffset) & (RADIXMASK) ;
		// Calculate write location //
		int location = digitFlags[numElements*digitVal+globalID] ;
		keyPtrOut[location] = key ;
		valPtrOut[location] = valPtrIn[globalID] ;
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
	dim3 blockDimensionsDecode(NUMTHREADSDEC) ;
	dim3 gridDimensionsDecode(numElements/(blockDimensionsDecode.x) + 
			((numElements%(blockDimensionsDecode.x))?1:0)) ;

	dim3 blockDimensionsLocalScan(NUMTHREADSRED) ;
	dim3 gridDimensionsLocalScan((RADIXSIZE*(numElements))/(blockDimensionsLocalScan.x*2) + 
			((((RADIXSIZE*numElements))%(blockDimensionsLocalScan.x*2))?1:0)) ;


	dim3 blockDimensionsPR(NUMTHREADSRED) ;
	dim3 gridDimensionsPR((RADIXSIZE*(numElements))/(blockDimensionsPR.x) + 
			((((RADIXSIZE*numElements))%(blockDimensionsPR.x))?1:0)) ;

	dim3 blockDimensionsBlockScan(NUMTHREADSBS) ;
	dim3 gridDimensionsBlockScan((gridDimensionsLocalScan.x)/(blockDimensionsBlockScan.x*2) + 
			(((gridDimensionsLocalScan.x)%(blockDimensionsBlockScan.x*2))?1:0)) ;

	// Allocate memory for prefix sum buffers. //
	int * digitFlags = NULL ;
	int * blockSumArray = NULL ; 
	const int digitFlagSize = 2*(gridDimensionsLocalScan.x * blockDimensionsLocalScan.x) ;
	const int blockSumArraySize = 2*(gridDimensionsBlockScan.x * blockDimensionsBlockScan.x) ; 
	cudaMalloc((void**) &digitFlags, sizeof(int)*digitFlagSize) ;
	cudaMalloc((void**) &blockSumArray, sizeof(int)*blockSumArraySize) ;

	// Create buffer for keys and values. //
	int * keyPtr1 = keys ;
	int * keyPtr2 = NULL ;
	int * valPtr1 = values ;
	int * valPtr2 = NULL ;
	cudaMalloc((void**) &keyPtr2, sizeof(int)*numElements) ;
	cudaMalloc((void**) &valPtr2, sizeof(int)*numElements) ;

	for (int i = 0 ; i < 30 ; i+=2) {
		cudaMemset(digitFlags, 0, sizeof(int)*digitFlagSize) ;
		flagDecode<<<gridDimensionsDecode,blockDimensionsDecode>>>(keyPtr1, digitFlags, numElements, i) ;
		calcBlockSumArray<<<gridDimensionsLocalScan,blockDimensionsLocalScan,blockDimensionsLocalScan.x*2*sizeof(int)>>>
			(digitFlags, blockSumArray) ;
		calcExclusivePrefixSum<<<gridDimensionsBlockScan, blockDimensionsBlockScan, blockDimensionsBlockScan.x*2*sizeof(int)>>>(
				blockSumArray) ;
		updateLocalPrefixes<<<gridDimensionsLocalScan, blockDimensionsLocalScan>>>(digitFlags, blockSumArray, digitFlagSize) ;
		//printPrefixValues<<< gridDimensionsPR, blockDimensionsPR>>>(digitFlags+1024, 1024) ;
		// Shuffle data to new locations. //
		shuffle<<<gridDimensionsDecode,blockDimensionsDecode>>>(keyPtr1,keyPtr2,valPtr1,valPtr2,digitFlags,numElements,i) ;
		//checkSortedGlobal<<< gridDimensionsDecode, blockDimensionsDecode>>>(keyPtr2,numElements, i+2) ;
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

	cudaFree(digitFlags) ;
	cudaFree(blockSumArray) ;
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

void cudaSharedRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
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

void cudaSharedRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
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
