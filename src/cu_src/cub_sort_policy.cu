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

#include "../../inc/cpp_inc/cub_sort_policy.hpp"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/cuda_error.cuh"
#include "../../../cub-1.5.2/cub/cub.cuh"

#define WARPSIZE 32


std::pair<float*,int*> CUBSort::allocate(const std::vector<Centroid> & centroids) {
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

void CUBSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) {

	float * gpuCamCo = NULL ;
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	cudaMalloc((void**) &gpuCamCo, sizeof(float)*3) ;
	cudaMemcpy(gpuCamCo,camCo.data(),sizeof(float)*3,cudaMemcpyHostToDevice) ;

	// Allocate buffer memory. //
	const int numCentroids = centroidIDsVec.size() ;
	float * gpuDistancesSqBuf1 = NULL ;
	float * gpuDistancesSqBuf2 = NULL ;
	int * gpuCentroidsIDsBuf1 = centroidIDs ;
	int * gpuCentroidsIDsBuf2 = NULL ;
	cudaMalloc((void **) &gpuDistancesSqBuf1, sizeof(float)*numCentroids) ;
	cudaMalloc((void **) &gpuDistancesSqBuf2, sizeof(float)*numCentroids) ;
	cudaMalloc((void **) &gpuCentroidsIDsBuf2, sizeof(int)*numCentroids) ;

	dim3 distanceBlock(1024) ;	
	dim3 distanceGrid(numCentroids/distanceBlock.x + (!(numCentroids%distanceBlock.x)?0:1)) ;
	cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(centroidPos, gpuCamCo, gpuDistancesSqBuf1, numCentroids) ;

	// Temporary storage needed by cub radix sort. //
	void * tempStorage = NULL ;
	size_t tempStorageBytes = 0 ;
	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSqBuf1, 
			gpuDistancesSqBuf2, gpuCentroidsIDsBuf1, gpuCentroidsIDsBuf2, numCentroids) ;
	cudaMalloc(&tempStorage, tempStorageBytes) ;

	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSqBuf1, 
		gpuDistancesSqBuf2, gpuCentroidsIDsBuf1, gpuCentroidsIDsBuf2, numCentroids) ;

	cudaMemcpy(centroidIDsVec.data(), gpuCentroidsIDsBuf2, sizeof(int)*numCentroids, cudaMemcpyDeviceToHost) ;

	cudaFree(gpuCentroidsIDsBuf2) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuDistancesSqBuf1) ;
	cudaFree(gpuDistancesSqBuf2) ;
	cudaFree(tempStorage) ;
}

void CUBSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos, std::vector<float> & times) {

	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	
	float * gpuCamCo = NULL ;
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	cudaMalloc((void**) &gpuCamCo, sizeof(float)*3) ;
	cudaMemcpy(gpuCamCo,camCo.data(),sizeof(float)*3,cudaMemcpyHostToDevice) ;

	// Allocate buffer memory. //
	const int numCentroids = centroidIDsVec.size() ;
	float * gpuDistancesSqBuf1 = NULL ;
	float * gpuDistancesSqBuf2 = NULL ;
	int * gpuCentroidsIDsBuf1 = centroidIDs ;
	int * gpuCentroidsIDsBuf2 = NULL ;
	cudaMalloc((void **) &gpuDistancesSqBuf1, sizeof(float)*numCentroids) ;
	cudaMalloc((void **) &gpuDistancesSqBuf2, sizeof(float)*numCentroids) ;
	cudaMalloc((void **) &gpuCentroidsIDsBuf2, sizeof(int)*numCentroids) ;

	dim3 distanceBlock(1024) ;	
	dim3 distanceGrid(numCentroids/distanceBlock.x + (!(numCentroids%distanceBlock.x)?0:1)) ;
	cudaEventRecord(start, 0) ;
	cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(centroidPos, gpuCamCo, gpuDistancesSqBuf1, numCentroids) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float transformTime ;
	cudaEventElapsedTime(&transformTime , start, stop) ;

	// Temporary storage needed by cub radix sort. //
	void * tempStorage = NULL ;
	size_t tempStorageBytes = 0 ;
	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSqBuf1, 
			gpuDistancesSqBuf2, gpuCentroidsIDsBuf1, gpuCentroidsIDsBuf2, numCentroids) ;
	cudaMalloc(&tempStorage, tempStorageBytes) ;

	cudaEventRecord(start, 0) ;
	cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gpuDistancesSqBuf1, 
		gpuDistancesSqBuf2, gpuCentroidsIDsBuf1, gpuCentroidsIDsBuf2, numCentroids) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float sortTime ;
	cudaEventElapsedTime(&sortTime , start, stop) ;

	cudaEventRecord(start, 0) ;
	cudaMemcpy(centroidIDsVec.data(), gpuCentroidsIDsBuf2, sizeof(int)*numCentroids, cudaMemcpyDeviceToHost) ;

	cudaFree(gpuCentroidsIDsBuf2) ;
	cudaFree(gpuCamCo) ;
	cudaFree(gpuDistancesSqBuf1) ;
	cudaFree(gpuDistancesSqBuf2) ;
	cudaFree(tempStorage) ;

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float copyTime ;
	cudaEventElapsedTime(&copyTime , start, stop) ;

	times.push_back(sortTime/1E3) ;
	times.push_back((sortTime+transformTime)/1E3) ;
	times.push_back((sortTime+transformTime+copyTime)/1E3) ;
}


void CUBSort::deAllocate(float * centroidPos, int * centroidIDs) {
	cudaFree(centroidPos) ;
	cudaFree(centroidIDs) ;
}
