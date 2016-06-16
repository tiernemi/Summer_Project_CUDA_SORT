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
#include <iostream>
#include <string.h>
#include <stdlib.h>

// Thrust. //
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// Custom Headers //
#include "../../inc/thrust_gpu_sort_funcs.hpp"

thrust::device_vector<int> devKeys(ids.begin(),ids.end()) ;
thrust::device_vector<float> devVals(dists.begin(),dists.end()) ;

void transferDataToGPU(std::vector<int> & ids, std::vector<float> & dists)

void cudaSortDistances(std::vector<int> & ids, std::vector<float> & dists) {
	thrust::device_vector<int> devKeys(ids.begin(),ids.end()) ;
	thrust::device_vector<float> devVals(dists.begin(),dists.end()) ;

	thrust::device_ptr<int> devKeyPtr = devKeys.data();
	thrust::device_ptr<float> devValPtr = devVals.data();

	thrust::sort_by_key(devValPtr,devValPtr+devVals.size(),devKeyPtr) ;

	thrust::copy(devVals.begin(),devVals.end(),dists.begin()) ;
	thrust::copy(devKeys.begin(),devKeys.end(),ids.begin()) ;
}

void cudaSortDistances(std::vector<int> & ids, std::vector<float> & dists, float & sortTime) {
	cudaEvent_t start ;
	cudaEvent_t stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;
	thrust::device_vector<int> devKeys(ids.begin(),ids.end()) ;
	thrust::device_vector<float> devVals(dists.begin(),dists.end()) ;

	thrust::device_ptr<int> devKeyPtr = devKeys.data();
	thrust::device_ptr<float> devValPtr = devVals.data();

	thrust::sort_by_key(devValPtr,devValPtr+devVals.size(),devKeyPtr) ;

	thrust::copy(devVals.begin(),devVals.end(),dists.begin()) ;
	thrust::copy(devKeys.begin(),devKeys.end(),ids.begin()) ;

	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float elapsedTime; 
	cudaEventElapsedTime(&elapsedTime , start, stop);
	sortTime = elapsedTime*1E3 ;
}
