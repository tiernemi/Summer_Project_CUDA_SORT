/*
 * =====================================================================================
 *
 *       Filename:  blockPrefix_sum.cu
 *
 *    Description:  CUDA code for blockPrefix sum on gpu. Prefix sum is an essential algorithmic
 *                  primitive in radix sorts. It's used to calculate offsets.
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

#include "stdio.h"

#define RADIXSIZE 4

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  blockPrefixSum
 *    Arguments:  int * indices - Indices array.
 *                float * values - Values array.
 *  Description:  Generates 2^d radix bitmasks and calculates their blockPrefix sum for a
 *                given block of indices and values. 
 * =====================================================================================
 */

__global__ void blockPrefixSum(int * indices, int * globalValues, int * blockSumArray) {
	// Fill in here. //
}
