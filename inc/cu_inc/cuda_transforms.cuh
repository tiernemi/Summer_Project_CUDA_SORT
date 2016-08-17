#ifndef CUDA_TRANSFORMS_SUM_CU_T02OEQEL
#define CUDA_TRANSFORMS_SUM_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  prefix_sum.cuh
 *
 *    Description:  Header file for useful gpu trasnforms.
 *
 *        Version:  1.0
 *        Created:  2016-06-30 11:11
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

__global__ void cudaCalcDistanceSq(const float * __restrict__ gpuCenCo, const float * __restrict__ gpuCamCo, 
float * gpuDistancesSq,  int numCentroids) ;

#endif /* end of header gaurd CUDA_TRANSFORMS_SUM_CU_T02OEQEL */
