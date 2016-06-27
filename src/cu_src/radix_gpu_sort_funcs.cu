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

#define WARPSIZE 32

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaCalcDistanceSq
 *    Arguments:  float * gpuTriCo - Pointer to gpu triangle co-ordinates.
 *                float * gpuCamCo - Pointer to camera co-ordinates.
 *                float * gpuDistances - Pointer to distances vector which will be
 *                written into an eventually sorted.
 *                int numTriangles - Number of triangles in data set.
 *                int numCameras - Number of cameras to sort relative to.
 *  Description:  Calculates the distance squared of each point relative to the camera.
 *                Reads are clustered in such a way to promote coalesced reading.
 * =====================================================================================
 */

__global__ void cudaCalcDistanceSq(float * gpuTriCo, float * gpuCamCo, float * gpuDistancesSq, 
		int numTriangles, int numCameras) {

	// Grid stride loop. //
	for (int i = threadIdx.x + blockDim.x * blockIdx.x  ; i < numTriangles ; 
			i += blockDim.x*gridDim.x) {
		// Coalesced reading of triangles. //
		float xt = gpuTriCo[i] ;
		float yt = gpuTriCo[i+numTriangles] ;
		float zt = gpuTriCo[i+(numTriangles*2)] ;
		// Coalesced reading of cameras. //
		float xc = gpuCamCo[0] ;
		float yc = gpuCamCo[numCameras] ;
		float zc = gpuCamCo[2*numCameras] ;

		float diffx = xt - xc ;
		float diffy = yt - yc ;
		float diffz = zt - zc ;

		gpuDistancesSq[i] = diffx*diffx + diffy*diffy + diffz*diffz ;
	}
}


static void cudaRadixSortKernels(float * distancesSq, int * ids, int numTriangles) {
	;
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

	const int memRequired = distanceBlock.x*sizeof(int)*5 ;
	for (int i = 0 ; i < numCameras ; ++i) {
		cudaCalcDistanceSq<<<distanceGrid,distanceBlock>>>(gpuTriCo, gpuCamCo+i, gpuDistancesSq, numTriangles, numCameras) ;
		prefixSum<<<distanceGrid,distanceBlock, memRequired>>>(gpuTriIds,(int*)gpuDistancesSq, numTriangles, 3) ;
		gpuErrchk(cudaPeekAtLastError()) ;
		gpuErrchk(cudaDeviceSynchronize()) ;
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
