/*
 * =====================================================================================
 *
 *       Filename:  thrust_gpu_sort.cpp
 *
 *    Description:  Implementaion of radix sort on cpu.
 *
 *        Version:  1.0
 *        Created:  2016-06-15 12:42
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>

// Custom Headers //
#include "../../inc/bitonic_gpu_sort_funcs.hpp"
#include "../../inc/transforms.hpp"

int WARP_SIZE = 32 ;
int BLOCK_SIZE_MULT = 32 ;

__global__ void initialPhaseCompar(float * gpuDists, int numSectionCompars, int sectionLength, int totalSeqLength) {
	int comparID = blockDim.x*blockIdx.x + threadIdx.x ;
	if (comparID < totalSeqLength) {
		int sectionID = comparID / numSectionCompars ;
		int sectionComparID = comparID % numSectionCompars ;
		int comparOffset = sectionID*sectionLength + sectionComparID ;
		int comparStride = sectionLength - 1 - 2*sectionComparID ;

		float num1 = gpuDists[comparOffset] ; //__ldg(&gpuDists[comparOffset]) ;
		float num2 = gpuDists[comparOffset+comparStride] ;//__ldg(&gpuDists[comparOffset]) ;
		if (num1 > num2) {
			gpuDists[comparOffset] = num2 ;
			gpuDists[comparOffset+comparStride] = num1 ;
		}
	}
}

__global__ void postPhaseCompar(float * gpuDists, int numSectionCompars, int sectionLength, int totalSeqLength) {
	int comparID = blockDim.x*blockIdx.x + threadIdx.x ;
	if (comparID < totalSeqLength) {
		int sectionID = comparID / numSectionCompars ;
		int sectionComparID = comparID % numSectionCompars ;
		int comparOffset = sectionID*sectionLength + sectionComparID ;

		float num1 = gpuDists[comparOffset] ; //__ldg (&gpuDists[comparOffset]) ;
		float num2 = gpuDists[comparOffset+numSectionCompars] ;
		if (num1 > num2) {
			gpuDists[comparOffset] = num2 ;
			gpuDists[comparOffset+numSectionCompars] = num1 ;
		}
	}
}

void bitonicSort(float * gpuDists, int length) {
	int N = length/2 ;
	dim3 dimBlock(WARP_SIZE*BLOCK_SIZE_MULT) ;
	dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	for (int i = 2 ; i <= length ; i*=2) {
		initialPhaseCompar<<<dimGrid, dimBlock>>>(gpuDists, i/2, i, length) ;
		for (int j = (i/4) ; j >= 1  ; j/=2) {
			postPhaseCompar<<<dimGrid, dimBlock>>>(gpuDists, j, 2*j, length) ;
		}
	}
}

void cudaBitonicSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
/*
	const int numTriangles = triangles.size() ;
	const int numCameras = triangles.size() ;

	float * gpuDists ;
	cudaMalloc((void**) &gpuDists, sizeof(float) * numTriangles) ;
	int * gpuIDs ;
	cudaMalloc((void**) &gpuIDs, sizeof(int) * numTriangles) ;

	for (int i = 0 ; i < cameras.size() ; ++i) {
		Transforms::transformToDistVec(distances,triangles,cameras[i]) ;
		cudaMemcpy(gpuDists, distances.data(), sizeof(float) * dataSize,  cudaMemcpyHostToDevice) ;
		bitonicSort(gpuDists, dataSize) ;
		cudaThreadSynchronize() ;	// Wait for the GPU launched work to complete
		cudaMemcpy(cpuData, gpuDists, sizeof(float) * dataSize, cudaMemcpyDeviceToHost) ;

	}

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start ;
	printf("%lf\n", elapsed_seconds.count());
	// ................................................................ //


	cudaFree((void*) gpuDists) ;
//	cudaDeviceReset() ;

	return 0;
*/
}

void cudaBitonicSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {
}


