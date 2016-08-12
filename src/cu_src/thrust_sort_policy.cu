/*
 * =====================================================================================
 *
 *       Filename:  thrust_gpu_sort.cpp
 *
 *    Description:  Implementaion of thrust sort on gpu.
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

// Thrust. //
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>

// Custom Headers //
#include "../../inc/cpp_inc/thrust_sort_policy.hpp"
#include "../../inc/cpp_inc/strided_iterators.hpp"

typedef thrust::tuple<float,float,float> Tuple3f;
typedef thrust::device_vector<float>::iterator DevVecIteratorf ;
typedef thrust::device_vector<int>::iterator DevVecIteratori ;
typedef thrust::tuple<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator,
		thrust::device_vector<float>::iterator> TupleIt ;
typedef thrust::zip_iterator<TupleIt> ZipIteratorTuple ;

// This functor implements the dot product between 3d vectors
struct calcDistance : public thrust::binary_function<Tuple3f,Tuple3f,float> {
	__host__ __device__
	float operator()(const Tuple3f & a, const Tuple3f & b) const {
		float diff1 = thrust::get<0>(a) - thrust::get<0>(b) ;
		float diff2 = thrust::get<1>(a) - thrust::get<1>(b) ;
		float diff3 = thrust::get<2>(a) - thrust::get<2>(b) ;
		return diff1*diff1 + diff2*diff2 + diff3*diff3 ;
	}
} ;


std::pair<float*,int*> ThrustSort::allocate(const std::vector<Centroid> & centroids) {
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


void ThrustSort::sort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos) {
	const int numCentroids = centroidIDsVec.size() ;
	// Get the pointers of the x, y, z data for the triangles. //
	thrust::device_ptr<float> cenXPtrBegin = thrust::device_pointer_cast(centroidPos)  ;
	thrust::device_ptr<float> cenYPtrBegin = cenXPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenZPtrBegin = cenYPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenXPtrEnd = cenXPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenYPtrEnd = cenYPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenZPtrEnd = cenZPtrBegin + numCentroids ;

	// Pre process camera co-ordinates. //
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	
	// Initialise device vectors. //`
	thrust::device_vector<float> devCamCo(camCo.begin(),camCo.end()) ;
	thrust::device_vector<float> devDists(numCentroids) ;
	thrust::device_ptr<int> devIDs = thrust::device_malloc<int>(numCentroids);
	thrust::copy(centroidIDs,centroidIDs+numCentroids,devIDs) ;

	thrust::device_ptr<float> camXPtrBegin = devCamCo.data() ;
	thrust::device_ptr<float> camYPtrBegin = devCamCo.data()+1 ;
	thrust::device_ptr<float> camZPtrBegin = devCamCo.data()+2 ;
	thrust::device_ptr<float> camXPtrEnd = devCamCo.data() + 1 ;
	thrust::device_ptr<float> camYPtrEnd = devCamCo.data() + 2 ;
	thrust::device_ptr<float> camZPtrEnd = devCamCo.data() + 3 ;

	// Zip the x...y...z vector into tuples of x,y,z //
	ZipIteratorTuple zipTriBegin = zip(cenXPtrBegin, cenYPtrBegin, cenZPtrBegin) ;
	ZipIteratorTuple zipTriEnd = zip(cenXPtrEnd, cenYPtrEnd, cenZPtrEnd);
	ZipIteratorTuple zipCamBegin = zip(camXPtrBegin, camYPtrBegin, camZPtrBegin) ;
	ZipIteratorTuple zipCamEnd = zip(camXPtrEnd, camYPtrEnd, camZPtrEnd);

	// Get the device pointers for the device cenangle ids and device distance vector. //
	thrust::device_ptr<float> devKeyPtr = devDists.data() ;
	thrust::device_ptr<int> devValPtr = devIDs  ;

	// For each camera get distance and sort ids. //
	thrust::constant_iterator<Tuple3f> cam(*(zipCamBegin)) ;
	thrust::permutation_iterator<ZipIteratorTuple,DevVecIteratori> permIter(zipTriBegin,devValPtr) ;
	thrust::transform(thrust::device, permIter, permIter+numCentroids, cam, devDists.begin(), calcDistance());
	thrust::sort_by_key(thrust::device, devKeyPtr,devKeyPtr+numCentroids,devValPtr) ;
	// GPU copy back to CPU. //
	thrust::copy(devValPtr,devValPtr+numCentroids,centroidIDsVec.begin()) ;

	thrust::device_free(devIDs) ;
}

void ThrustSort::benchSort(const Camera & camera, std::vector<int> & centroidIDsVec, int * centroidIDs, float * centroidPos, std::vector<float> & times) {

	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;

	const int numCentroids = centroidIDsVec.size() ;
	// Get the pointers of the x, y, z data for the triangles. //
	thrust::device_ptr<float> cenXPtrBegin = thrust::device_pointer_cast(centroidPos)  ;
	thrust::device_ptr<float> cenYPtrBegin = cenXPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenZPtrBegin = cenYPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenXPtrEnd = cenXPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenYPtrEnd = cenYPtrBegin + numCentroids ;
	thrust::device_ptr<float> cenZPtrEnd = cenZPtrBegin + numCentroids ;

	// Pre process camera co-ordinates. //
	std::vector<float> camCo(3) ;
	const float * coords = camera.getCoords() ;
	camCo[0] = coords[0] ;
	camCo[1] = coords[1] ;
	camCo[2] = coords[2] ;
	
	// Initialise device vectors. //`
	thrust::device_vector<float> devCamCo(camCo.begin(),camCo.end()) ;
	thrust::device_vector<float> devDists(numCentroids) ;
	thrust::device_ptr<int> devIDs = thrust::device_malloc<int>(numCentroids);
	thrust::copy(centroidIDs,centroidIDs+numCentroids,devIDs) ;

	thrust::device_ptr<float> camXPtrBegin = devCamCo.data() ;
	thrust::device_ptr<float> camYPtrBegin = devCamCo.data()+1 ;
	thrust::device_ptr<float> camZPtrBegin = devCamCo.data()+2 ;
	thrust::device_ptr<float> camXPtrEnd = devCamCo.data() + 1 ;
	thrust::device_ptr<float> camYPtrEnd = devCamCo.data() + 2 ;
	thrust::device_ptr<float> camZPtrEnd = devCamCo.data() + 3 ;

	// Zip the x...y...z vector into tuples of x,y,z //
	ZipIteratorTuple zipTriBegin = zip(cenXPtrBegin, cenYPtrBegin, cenZPtrBegin) ;
	ZipIteratorTuple zipTriEnd = zip(cenXPtrEnd, cenYPtrEnd, cenZPtrEnd);
	ZipIteratorTuple zipCamBegin = zip(camXPtrBegin, camYPtrBegin, camZPtrBegin) ;
	ZipIteratorTuple zipCamEnd = zip(camXPtrEnd, camYPtrEnd, camZPtrEnd);

	// Get the device pointers for the device cenangle ids and device distance vector. //
	thrust::device_ptr<float> devKeyPtr = devDists.data() ;
	thrust::device_ptr<int> devValPtr = devIDs  ;

	// For each camera get distance and sort ids. //
	cudaEventRecord(start, 0) ;
	thrust::constant_iterator<Tuple3f> cam(*(zipCamBegin)) ;
	thrust::permutation_iterator<ZipIteratorTuple,DevVecIteratori> permIter(zipTriBegin,devValPtr) ;
	thrust::transform(thrust::device, permIter, permIter+numCentroids, cam, devDists.begin(), calcDistance());
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float transformTime ;
	cudaEventElapsedTime(&transformTime , start, stop) ;

	cudaEventRecord(start, 0) ;
	thrust::sort_by_key(thrust::device, devKeyPtr,devKeyPtr+numCentroids,devValPtr) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float sortTime ;
	cudaEventElapsedTime(&sortTime , start, stop) ;
	// GPU copy back to CPU. //
	cudaEventRecord(start, 0) ;
	thrust::copy(devValPtr,devValPtr+numCentroids,centroidIDsVec.begin()) ;
	thrust::device_free(devIDs) ;
	cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	float copyTime ;
	cudaEventElapsedTime(&copyTime , start, stop) ;

	times.push_back(sortTime/1E3) ;
	times.push_back((sortTime+transformTime)/1E3) ;
	times.push_back((sortTime+transformTime+copyTime)/1E3) ;
}



void ThrustSort::deAllocate(float * centroidPos, int * centroidIDs) {
	cudaFree(centroidPos) ;
	cudaFree(centroidIDs) ;
}
