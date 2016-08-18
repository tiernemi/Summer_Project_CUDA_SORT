/*
 * =====================================================================================
 *
 *       Filename:  cuda_transforms.cu
 *
 *    Description:  Useful transforms needed before sorting can be carried out.
 *
 *        Version:  1.0
 *        Created:  2016-06-30 11:14
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "../../inc/cu_inc/cuda_transforms.cuh"
#include <stdio.h>

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaCalcDistanceSq
 *    Arguments:  float * gpuTriCo - Pointer to gpu triangle co-ordinates.
 *                float * gpuCamCo - Pointer to camera co-ordinates.
 *                float * gpuDistances - Pointer to distances vector which will be
 *                int numCentroids - Number of centroids in data set
 *  Description:  Calculates the distance squared of each point relative to the camera.
 *                Reads are clustered in such a way to promote coalesced reading.
 * =====================================================================================
 */

__global__ void cudaCalcDistanceSq(const float * __restrict__ gpuCenCo, const float * __restrict__ gpuCamCo, float * gpuDistancesSq, int numCentroids) {

	// Grid stride loop. //

	for (int i = threadIdx.x + blockDim.x * blockIdx.x  ; i < numCentroids ; 
			i += blockDim.x*gridDim.x) {
		// Coalesced reading of centroids. //
		float xt = gpuCenCo[i] ;
		float yt = gpuCenCo[i+numCentroids] ;
		float zt = gpuCenCo[i+(numCentroids*2)] ;
		// Coalesced reading of cameras. //
		float xc = gpuCamCo[0] ;
		float yc = gpuCamCo[1] ;
		float zc = gpuCamCo[2] ;

		float diffx = xt - xc ;
		float diffy = yt - yc ;
		float diffz = zt - zc ;

		gpuDistancesSq[i] = diffx*diffx + diffy*diffy + diffz*diffz ;
	}
}





