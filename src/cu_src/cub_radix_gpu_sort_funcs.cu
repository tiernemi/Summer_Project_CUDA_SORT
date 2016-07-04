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

#include "../../inc/cpp_inc/cub_radix_gpu_sort_funcs.hpp"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../../cub-1.5.2/cub/cub.cuh"

#define WARPSIZE 32

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaCubRadixSortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles to be sorted.
 *                std::vector<Camera> & cameras - Vector of cameras to be sorted relative
 *                to.
 *  Description:  Uses the CUB librray radix sort to sort the key value pairs.
 * =====================================================================================
 */

void cudaCubRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	// Vectorise Triangle data. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;

	const int numTriangles = triangles.size() ;
	const int numCameras = cameras.size() ;

	// Vectorise triangle co-ordinates. //
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
	
	// Allocate and transfer to GPU memory. //
	float * gpuTriCo = NULL ;
	float * gpuCamCo = NULL ;
	float * gpuDistancesSq = NULL ;
	int * gpuTriIds = NULL ;
	int * gpuTriIdsOut = NULL ;
	float * gpuDistancesSqOut = NULL ;
	cudaMalloc((void **) &gpuTriCo, sizeof(float)*triCo.size()) ;
	cudaMalloc((void **) &gpuCamCo, sizeof(float)*camCo.size()) ;
	cudaMalloc((void **) &gpuTriIds, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuTriIdsOut, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*triangles.size()) ;
	cudaMalloc((void **) &gpuDistancesSqOut, sizeof(float)*triangles.size()) ;

	cudaMemcpy(gpuTriCo, triCo.data(), sizeof(float)*triCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuCamCo, camCo.data(), sizeof(float)*camCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuTriIds, triIds.data(), sizeof(int)*triIds.size(), cudaMemcpyHostToDevice) ;

	// Temporary storage needed by cub radix sort. //
	void * tempStorage = NULL ;
	size_t tempStorageBytes = 0 ;
	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSq, 
			gpuDistancesSqOut, gpuTriIds, gpuTriIdsOut, numTriangles) ;
	cudaMalloc(&tempStorage, tempStorageBytes) ;

	// Block dimensions for transforms. //
	dim3 distanceBlock(WARPSIZE) ;
	dim3 distanceGrid(numTriangles/distanceBlock.x + (!(numTriangles%distanceBlock.x)?0:1)) ;

	// For each camera, transforms and use the cub radix sort to sort. //
	for (int i = 0 ; i < numCameras ; ++i) {
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, gpuTriIds, numTriangles, numCameras) ;
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSq, 
			gpuDistancesSqOut, gpuTriIds, gpuTriIdsOut, numTriangles) ;

		cudaMemcpy(triIds.data(), gpuTriIdsOut, sizeof(int)*triIds.size(), cudaMemcpyDeviceToHost) ;
	}

	// Free GPU memory. //
	cudaFree(gpuTriIds) ;
	cudaFree(gpuTriCo) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuDistancesSq) ;
	cudaFree(tempStorage) ;

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaCubRadixSortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles to be sorted.
 *                std::vector<Camera> & cameras - Vector of cameras to be sorted relative
 *                to.
 *		          std::vector<float> & times - Vector used to save benchmarking times.
 *  Description:  Uses the CUB librray radix sort to sort the key value pairs.
 * =====================================================================================
 */

void cudaCubRadixSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
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

	// Vectorise triangle co-ordinates. //
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
	
	// Allocate and transfer to GPU memory. //
	float * gpuTriCo = NULL ;
	float * gpuCamCo = NULL ;
	float * gpuDistancesSq = NULL ;
	int * gpuTriIds = NULL ;
	int * gpuTriIdsOut = NULL ;
	float * gpuDistancesSqOut = NULL ;
	cudaMalloc((void **) &gpuTriCo, sizeof(float)*triCo.size()) ;
	cudaMalloc((void **) &gpuCamCo, sizeof(float)*camCo.size()) ;
	cudaMalloc((void **) &gpuTriIds, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuTriIdsOut, sizeof(int)*triIds.size()) ;
	cudaMalloc((void **) &gpuDistancesSq, sizeof(float)*triangles.size()) ;
	cudaMalloc((void **) &gpuDistancesSqOut, sizeof(float)*triangles.size()) ;

	cudaMemcpy(gpuTriCo, triCo.data(), sizeof(float)*triCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuCamCo, camCo.data(), sizeof(float)*camCo.size(), cudaMemcpyHostToDevice) ;
	cudaMemcpy(gpuTriIds, triIds.data(), sizeof(int)*triIds.size(), cudaMemcpyHostToDevice) ;

	// Temporary storage needed by cub radix sort. //
	void * tempStorage = NULL ;
	size_t tempStorageBytes = 0 ;
	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSq, 
			gpuDistancesSqOut, gpuTriIds, gpuTriIdsOut, numTriangles) ;
	cudaMalloc(&tempStorage, tempStorageBytes) ;

	// Block dimensions for transforms. //
	dim3 distanceBlock(WARPSIZE) ;
	dim3 distanceGrid(numTriangles/distanceBlock.x + (!(numTriangles%distanceBlock.x)?0:1)) ;

	// For each camera, transforms and use the cub radix sort to sort. //
	for (int i = 0 ; i < numCameras ; ++i) {
		cudaEventRecord(start, 0) ;
		
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, gpuTriIds, numTriangles, numCameras) ;

		cudaEventRecord(stop, 0) ;
		cudaEventSynchronize(stop) ;
		float transformTime ;
		cudaEventElapsedTime(&transformTime , start, stop) ;

		cudaEventRecord(start, 0);
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSq, 
			gpuDistancesSqOut, gpuTriIds, gpuTriIdsOut, numTriangles) ;
		cudaEventRecord(stop, 0) ;
		cudaEventSynchronize(stop);
		float sortTime ;
		cudaEventElapsedTime(&sortTime , start, stop) ;

		cudaEventRecord(start, 0) ;
		// Read back new indices. /
		cudaMemcpy(triIds.data(), gpuTriIdsOut, sizeof(int)*triIds.size(), cudaMemcpyDeviceToHost) ;
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


	// Free GPU memory. //
	cudaFree(gpuTriIds) ;
	cudaFree(gpuTriCo) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuDistancesSq) ;
	cudaFree(tempStorage) ;

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}

	times = newTimes ;
}
