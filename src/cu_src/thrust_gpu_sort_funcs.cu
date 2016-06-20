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
#include "../../inc/thrust_gpu_sort_funcs.hpp"
#include "../../inc/strided_iterators.hpp"

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

void cudaThrustSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {
	// Pre process triangle co-ordinates. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triIds[i] = i ;
		const float * coords = triangles[i].getCoords() ;
		triCo[i] = coords[0] ;
		triCo[i+triangles.size()] = coords[1] ;
		triCo[i+2*triangles.size()] = coords[2] ;
	}

	// Pre process camera co-ordinates. //
	std::vector<float> camCo(3*cameras.size()) ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		const float * coords = cameras[i].getCoords() ;
		camCo[i] = coords[0] ;
		camCo[i+cameras.size()] = coords[1] ;
		camCo[i+2*cameras.size()] = coords[2] ;
	}
	
	// Initialise device vectors. //`
	thrust::device_vector<float> devTriCo(triCo.begin(),triCo.end()) ;
	thrust::device_vector<float> devTriTemp(devTriCo.size()) ;
	thrust::device_vector<int> devTriIds(triIds.begin(),triIds.end()) ;
	thrust::device_vector<float> devCamCo(camCo.begin(),camCo.end()) ;
	thrust::device_vector<float> devDists(triCo.size()/3) ;

	// Number of triangles etc. //
	const int numTriangles = devTriCo.size()/3 ;
	const int numCameras = devCamCo.size()/3 ;

	// Get the pointers of the x, y, z data for the triangles. //
	thrust::device_ptr<float> triXPtrBegin = devTriCo.data() ;
	thrust::device_ptr<float> triYPtrBegin = devTriCo.data() + numTriangles ;
	thrust::device_ptr<float> triZPtrBegin = devTriCo.data() + 2*numTriangles ;
	thrust::device_ptr<float> triXPtrEnd = devTriCo.data() + numTriangles ;
	thrust::device_ptr<float> triYPtrEnd = devTriCo.data() + 2*numTriangles ;
	thrust::device_ptr<float> triZPtrEnd = devTriCo.data() + 3*numTriangles ;

	// Get the pointers of the x, y, z data for the camera. //
	thrust::device_ptr<float> camXPtrBegin = devCamCo.data() ;
	thrust::device_ptr<float> camYPtrBegin = devCamCo.data() + numCameras ;
	thrust::device_ptr<float> camZPtrBegin = devCamCo.data() + 2*numCameras ;
	thrust::device_ptr<float> camXPtrEnd = devCamCo.data() + numCameras ;
	thrust::device_ptr<float> camYPtrEnd = devCamCo.data() + 2*numCameras ;
	thrust::device_ptr<float> camZPtrEnd = devCamCo.data() + 3*numCameras ;

	// Zip the x...y...z vector into tuples of x,y,z //
	ZipIteratorTuple zipTriBegin = zip(triXPtrBegin, triYPtrBegin, triZPtrBegin) ;
	ZipIteratorTuple zipTriEnd = zip(triXPtrEnd, triYPtrEnd, triZPtrEnd);
	ZipIteratorTuple zipCamBegin = zip(camXPtrBegin, camYPtrBegin, camZPtrBegin) ;
	ZipIteratorTuple zipCamEnd = zip(camXPtrEnd, camYPtrEnd, camZPtrEnd);

	// Get the device pointers for the device triangle ids and device distance vector. //
	thrust::device_ptr<int> devKeyPtr = devTriIds.data();
	thrust::device_ptr<float> devValPtr = devDists.data();

	// For each camera get distance and sort ids. //
	for (int i = 0 ; i < numCameras ; ++i) {
		thrust::constant_iterator<Tuple3f> cam(*(zipCamBegin+i)) ;
		thrust::permutation_iterator<ZipIteratorTuple,DevVecIteratori> permIter(zipTriBegin,devTriIds.begin()) ;
		thrust::transform(thrust::device, permIter, permIter+numTriangles, cam, devDists.begin(), calcDistance());
		thrust::sort_by_key(thrust::device, devValPtr,devValPtr+numTriangles,devKeyPtr) ;
	}

	// GPU copy back to CPU. //
	thrust::copy(devTriIds.begin(),devTriIds.end(),triIds.begin()) ;

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaThrustSortTriangles
 *    Arguments:  std::vector<Triangle> & triangles - Vector of triangles to sort.
 *                std::vector<Camera> & cameras - Vector of cameras to sort relative to.
 *		          std::vector<float> & times - Vector of times 
 *
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

void cudaThrustSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	std::vector<float> newTimes ;

	// Pre process triangle co-ordinates. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triIds[i] = i ;
		const float * coords = triangles[i].getCoords() ;
		triCo[i] = coords[0] ;
		triCo[i+triangles.size()] = coords[1] ;
		triCo[i+2*triangles.size()] = coords[2] ;
	}

	// Pre process camera co-ordinates. //
	std::vector<float> camCo(3*cameras.size()) ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		const float * coords = cameras[i].getCoords() ;
		camCo[i] = coords[0] ;
		camCo[i+cameras.size()] = coords[1] ;
		camCo[i+2*cameras.size()] = coords[2] ;
	}
	
	// Initialise device vectors. //`
	thrust::device_vector<float> devTriCo(triCo.begin(),triCo.end()) ;
	thrust::device_vector<float> devTriTemp(devTriCo.size()) ;
	thrust::device_vector<int> devTriIds(triIds.begin(),triIds.end()) ;
	thrust::device_vector<float> devCamCo(camCo.begin(),camCo.end()) ;
	thrust::device_vector<float> devDists(triCo.size()/3) ;

	// Number of triangles etc. //
	const int numTriangles = devTriCo.size()/3 ;
	const int numCameras = devCamCo.size()/3 ;

	// Get the pointers of the x, y, z data for the triangles. //
	thrust::device_ptr<float> triXPtrBegin = devTriCo.data() ;
	thrust::device_ptr<float> triYPtrBegin = devTriCo.data() + numTriangles ;
	thrust::device_ptr<float> triZPtrBegin = devTriCo.data() + 2*numTriangles ;
	thrust::device_ptr<float> triXPtrEnd = devTriCo.data() + numTriangles ;
	thrust::device_ptr<float> triYPtrEnd = devTriCo.data() + 2*numTriangles ;
	thrust::device_ptr<float> triZPtrEnd = devTriCo.data() + 3*numTriangles ;

	// Get the pointers of the x, y, z data for the camera. //
	thrust::device_ptr<float> camXPtrBegin = devCamCo.data() ;
	thrust::device_ptr<float> camYPtrBegin = devCamCo.data() + numCameras ;
	thrust::device_ptr<float> camZPtrBegin = devCamCo.data() + 2*numCameras ;
	thrust::device_ptr<float> camXPtrEnd = devCamCo.data() + numCameras ;
	thrust::device_ptr<float> camYPtrEnd = devCamCo.data() + 2*numCameras ;
	thrust::device_ptr<float> camZPtrEnd = devCamCo.data() + 3*numCameras ;

	// Zip the x...y...z vector into tuples of x,y,z //
	ZipIteratorTuple zipTriBegin = zip(triXPtrBegin, triYPtrBegin, triZPtrBegin) ;
	ZipIteratorTuple zipTriEnd = zip(triXPtrEnd, triYPtrEnd, triZPtrEnd);
	ZipIteratorTuple zipCamBegin = zip(camXPtrBegin, camYPtrBegin, camZPtrBegin) ;
	ZipIteratorTuple zipCamEnd = zip(camXPtrEnd, camYPtrEnd, camZPtrEnd);

	// Get the device pointers for the device triangle ids and device distance vector. //
	thrust::device_ptr<int> devKeyPtr = devTriIds.data();
	thrust::device_ptr<float> devValPtr = devDists.data();

	// For each camera get distance and sort ids. //
	for (int i = 0 ; i < numCameras ; ++i) {
		thrust::constant_iterator<Tuple3f> cam(*(zipCamBegin+i)) ;
		thrust::permutation_iterator<ZipIteratorTuple,DevVecIteratori> permIter(zipTriBegin,devTriIds.begin()) ;
		thrust::transform(thrust::device, permIter, permIter+numTriangles, cam, devDists.begin(), calcDistance());
		cudaEventRecord(start, 0);
		thrust::sort_by_key(thrust::device, devValPtr,devValPtr+numTriangles,devKeyPtr) ;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime ;
		cudaEventElapsedTime(&elapsedTime , start, stop) ;
		newTimes.push_back(elapsedTime/1E3) ;
	}

	// GPU copy back to CPU. //
	thrust::copy(devTriIds.begin(),devTriIds.end(),triIds.begin()) ;

	// CPU Overwrite triangles. //
	std::vector<Triangle> temp = triangles ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triangles[i] = temp[triIds[i]] ;
	}

	times = newTimes ;
}

