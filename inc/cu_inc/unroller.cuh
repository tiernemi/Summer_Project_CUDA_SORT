#ifndef UNROLLER
#define UNROLLER

#include <algorithm>
#include "prefix_sums.cuh"
#include "upsweep_reduce.cuh"
#include "top_level_scan.cuh"
#include "downsweep_scan.cuh"

// Useful MACROS. //
#define WARPSIZE 32
#define WARPSIZE_HALF 16
#define P2_WARPSIZE_HALF 5
#define WARPSIZE_MIN_1 31
#define RADIXSIZE 4
#define RADIXMASK 3
#define RADIXWIDTH 2
#define NUM_KEYS_PER_THREAD 4

#define NUM_BLOCKS 512
#define NUM_THREADS_PER_BLOCK 128
#define NUM_WARPS NUM_THREADS_PER_BLOCK/WARPSIZE
#define PARA_BIT_SIZE 8
#define PARA_BIT_MASK_0 255


template <int N>
struct _Uint { } ;

template <int N>
inline void unroll(int * & keyPtr1, int * & keyPtr2, int * & valPtr1, int * & valPtr2, int * blockReduceArray, int numCentroids, int numTiles, dim3 & dimGrid, dim3 & blockDim, _Uint<N> num) {
		
	unroll(keyPtr1,keyPtr2,valPtr1,valPtr2,blockReduceArray,numCentroids,numTiles,dimGrid,blockDim, _Uint<N-1>()) ;
	upsweepReduce<RADIXWIDTH*N><<<dimGrid,blockDim>>>(keyPtr1,blockReduceArray+1,numCentroids,numTiles) ;
	topLevelScan<RADIXWIDTH*N><<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
	downsweepScan<RADIXWIDTH*N><<<dimGrid,blockDim>>>(keyPtr1, keyPtr2, valPtr1, valPtr2, blockReduceArray, numCentroids, numTiles) ;
	std::swap(keyPtr1,keyPtr2) ;
	std::swap(valPtr1,valPtr2) ;
} ;

inline void unroll(int * & keyPtr1, int * & keyPtr2, int * & valPtr1, int * & valPtr2, int * blockReduceArray, int numCentroids, int numTiles, dim3 & dimGrid, dim3 & blockDim, _Uint<0> num) {
	upsweepReduce<0><<<dimGrid,blockDim>>>(keyPtr1,blockReduceArray+1,numCentroids,numTiles) ;
	topLevelScan<0><<<1,NUM_BLOCKS>>>(blockReduceArray+1,blockReduceArray+1) ;
	downsweepScanFinal<0><<<dimGrid,blockDim>>>(keyPtr1, valPtr1, valPtr2, blockReduceArray, numCentroids, numTiles) ;
	std::swap(valPtr1,valPtr2) ;
} ;



#endif
