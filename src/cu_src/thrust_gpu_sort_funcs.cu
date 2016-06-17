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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>

// Custom Headers //
#include "../../inc/thrust_gpu_sort_funcs.hpp"
#include "../../inc/strided_iterators.hpp"

typedef thrust::tuple<float,float,float> tuple3;
typedef StridedRange<float*> strTuple3f ;
typedef thrust::tuple<strTuple3f::iterator,strTuple3f::iterator,strTuple3f::iterator> iterTup3f ;
typedef thrust::zip_iterator<iterTup3f> zipTuple3f  ;


// This functor implements the dot product between 3d vectors
struct calcDistance : public thrust::binary_function<tuple3,tuple3,float> {
	__host__ __device__
	float operator()(const tuple3 & a, const tuple3 & b) const {
		float diff1 = thrust::get<0>(a) - thrust::get<0>(b) ;
		float diff2 = thrust::get<1>(a) - thrust::get<1>(b) ;
		float diff3 = thrust::get<2>(a) - thrust::get<2>(b) ;
		return sqrt(diff1 + diff2 + diff3) ;
	}
} ;

void cudaSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras) {

	// Pre process triangle co-ordinates. //
	std::vector<float> triCo(3*triangles.size()) ;
	std::vector<int> triIds(triangles.size()) ;
	for (unsigned int i = 0 ; i < triangles.size() ; ++i) {
		triIds[i] = triangles[i].getID() ;
		const float * coords = triangles[i].getCoords() ;
		triCo[3*i] = coords[0] ;
		triCo[3*i+1] = coords[1] ;
		triCo[3*i+2] = coords[2] ;
	}

	// Pre process camera co-ordinates. //
	std::vector<float> camCo(3*cameras.size()) ;
	for (unsigned int i = 0 ; i < cameras.size() ; ++i) {
		const float * coords = cameras[i].getCoords() ;
		camCo[3*i] = coords[0] ;
		camCo[3*i+1] = coords[1] ;
		camCo[3*i+2] = coords[2] ;
	}
	
	thrust::device_vector<float> devTriCo(triCo.begin(),triCo.end()) ;
	thrust::device_vector<int> devTriIds(triIds.begin(),triIds.end()) ;
	thrust::device_vector<float> devCamCo(camCo.begin(),camCo.end()) ;
	thrust::device_vector<float> devNewCo(triCo.size()/3) ;

	const int triStride = 3 ;
	const int numTri = devTriCo.size()/3 ;
	const int triSize = devTriCo.size() ;

	float * devTriPtr = thrust::raw_pointer_cast(devTriCo.data()) ;
	StridedRange<float*> firstT =  StridedRange<float*>(devTriPtr,   devTriPtr + 6,   3);
	StridedRange<float*> secondT = StridedRange<float*>(devTriPtr+1, devTriPtr + triSize+1-triStride+1, triStride);
	StridedRange<float*> thirdT =  StridedRange<float*>(devTriPtr+2, devTriPtr + triSize+1-triStride+2, triStride);
	zipTuple3f zipBeginItT = zip(firstT.begin(),secondT.begin(), thirdT.begin());
	zipTuple3f zipEndItT = zip(firstT.end(), secondT.end(), thirdT.end());

    // Finally, we pass the zip_iterators into transform() as if they
    // were 'normal' iterators for a device_vector<Float3>.
   // thrust::transform(A_first, A_last, thirdT.begin(), result.begin(), calcDistance());

	//std::cout << *thrust::device_ptr<float>(&(thrust::get<0>(*zipBeginItT))) << std::endl ;
	const int camStride = 3 ;
	const int numCams = devCamCo.size()/3 ;
	const int camSize = devCamCo.size() ;
	float * devCamPtr = thrust::raw_pointer_cast(devCamCo.data()) ;
	StridedRange<float*> firstC =  StridedRange<float*>(devCamPtr,   devCamPtr + camSize+1-camStride,  camStride);
	StridedRange<float*> secondC = StridedRange<float*>(devCamPtr+1, devCamPtr + camSize+1-camStride+1, camStride);
	StridedRange<float*> thirdC =  StridedRange<float*>(devCamPtr+2, devCamPtr + camSize+1-camStride+2, camStride);
	zipTuple3f zipBeginItC = zip(firstC.begin(),secondC.begin(), thirdC.begin());
	zipTuple3f zipEndItC = zip(firstC.end(), secondC.end(), thirdC.end());

	for (int i = 0 ; i < numCams ; ++i) {
		//thrust::constant_iterator<thrust::tuple<float,float,float> > camTuple = thrust::make_constant_iterator<thrust::tuple<float,float,float> >(*(zipBeginItT+i))  ;
		//thrust::transform(zipBeginItT, zipEndItT, zipBeginItT, devNewCo.begin(), calcDistance());
	}
	//thrust::constant_iterator<zipTuple3f> camTuple(zipBeginItT) ;
	//std::cout << *thrust::device_ptr<float>(&(thrust::get<0>(**camTuple))) << std::endl ;

}

void cudaSortTriangles(std::vector<Triangle> & triangles, std::vector<Camera> & cameras, 
		std::vector<float> & times) {

}

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
	float elapsedTime ; 
	cudaEventElapsedTime(&elapsedTime , start, stop) ;
	sortTime = elapsedTime*1E3 ;
}


