/*
 * =====================================================================================
 *
 *       Filename:  radix_sort.cu
 *
 *    Description:  CUDA code for basic global based radix sort on gpu.
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

#include "../../inc/cpp_inc/basic_radix_gpu_sort_funcs.hpp"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../inc/cu_inc/test_utils.cuh"

#define WARPSIZE 32
#define NUMTHREADSDEC 256
#define NUMTHREADSRED 256
#define RADIXSIZE 4
#define RADIXMASK 3

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
 *         Name:  blockPrefixScan
 *    Arguments:  int * digitFlagsIn - Digit flags array used for reading.
 *                int * digitFlagsOut - Digit flags array used for writing. 
 *                int depth - Depth of the reduction
 *  Description:  Performs a block wide prefix sum step in a naive manner. Depth informs
 *                the current level of the recursion.
 * =====================================================================================
 */

static __global__ void blockPrefixScan(int * digitFlagsIn, int * digitFlagsOut, const int numElements, int depth) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID < depth) {
		digitFlagsOut[globalID] = digitFlagsIn[globalID] ;
	}
	else if (globalID < RADIXSIZE*numElements) {
		digitFlagsOut[globalID] = digitFlagsIn[globalID-depth] + digitFlagsIn[globalID] ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  exclusiveSetup
 *    Arguments:  int * digitFlagsIn - Input digit flags.
 *                int * digitFlagsOut - Output digit flags.
 *                const int numElements - Number of keys.
 *  Description:  Seeds digitFlags with a left shift in order to generate an 
 *                exclusive scan.
 * =====================================================================================
 */

static __global__ void exclusiveSetup(int * digitFlagsIn, int * digitFlagsOut, const int numElements) {
	int globalID = threadIdx.x + blockDim.x * blockIdx.x ;
	if (globalID < numElements*RADIXSIZE) {
		if (globalID == 0) {
			digitFlagsOut[globalID] = 0 ;
		} else {
			digitFlagsOut[globalID] = digitFlagsIn[globalID-1] ;
		}
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
			(!(numElements%(blockDimensionsDecode.x))?0:1)) ;

	dim3 blockDimensionsScan(NUMTHREADSRED) ;
	dim3 gridDimensionsScan((RADIXSIZE*(numElements))/(blockDimensionsScan.x + 
				(!(((RADIXSIZE*numElements))%(blockDimensionsScan.x))?0:1))) ;

	// Allocate memory for prefix sum buffers. //
	int * digitFlags1 = NULL ;
	int * digitFlags2 = NULL ;
	cudaMalloc((void**) &digitFlags1, sizeof(int)*numElements*RADIXSIZE) ;
	cudaMalloc((void**) &digitFlags2, sizeof(int)*numElements*RADIXSIZE) ;

	// Create buffer for keys and values. //
	int * keyPtr1 = keys ;
	int * keyPtr2 = NULL ;
	int * valPtr1 = values ;
	int * valPtr2 = NULL ;
	cudaMalloc((void**) &keyPtr2, sizeof(int)*numElements) ;
	cudaMalloc((void**) &valPtr2, sizeof(int)*numElements) ;

	// Use a naive global implementation of Hillis and Steele algorithm. //
	int highestPower2 = (1 << (int) (std::ceil(log2(float(numElements*RADIXSIZE))))) ;
	for (int i = 0 ; i < 30 ; i+=2) {
		cudaMemset(digitFlags1, 0, sizeof(int)*numElements*RADIXSIZE) ;
		flagDecode<<<gridDimensionsDecode,blockDimensionsDecode>>>(keyPtr1, digitFlags1, numElements, i) ;
		exclusiveSetup<<<gridDimensionsScan,blockDimensionsScan>>>(digitFlags1,digitFlags2,numElements) ;

		// Perform "device synchronised" block wide prefix sum. //
		for (int j = 1 ; j < highestPower2 ; j<<=1) {
			blockPrefixScan<<<gridDimensionsScan, blockDimensionsScan>>>(digitFlags2,digitFlags1,numElements,j) ; 
			cudaDeviceSynchronize() ;
			std::swap(digitFlags2,digitFlags1) ;
		}
		// Shuffle data to new locations. //
		shuffle<<<gridDimensionsDecode,blockDimensionsDecode>>>(keyPtr1,keyPtr2,valPtr1,valPtr2,digitFlags2,numElements,i) ;
		
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

	cudaFree(digitFlags1) ;
	cudaFree(digitFlags2) ;
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

void cudaBasicRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
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

void cudaBasicRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {
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
		cudaEventRecord(start, 0) ;
		// Transform triangles to distance vector. //
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, gpuTriIds, numTriangles, numCameras) ;
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
