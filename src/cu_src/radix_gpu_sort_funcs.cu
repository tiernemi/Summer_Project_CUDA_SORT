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
#include "../../inc/cu_inc/prefix_sum.cuh"
#include "../../inc/cu_inc/cuda_transforms.cuh"
#include "../../inc/cu_inc/scattered_write.cuh"

#include "../../../cub-1.5.2/cub/cub.cuh"

#define WARPSIZE 32
#define RADIXSIZE 4

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
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, numTriangles, numCameras) ;
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
