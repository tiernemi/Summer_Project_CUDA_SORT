#ifndef SCATTERED_WRITE_CU_T02OEQEL
#define SCATTERED_WRITE_CU_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  scatter_write.cuh
 *
 *    Description:  Header file for scattered write.
 *
 *        Version:  1.0
 *        Created:  07/06/16 17:22:08
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

__global__ void scatteredWrite(int * indices, int * globalValues, int * localPrefixSumArray, int * blockSumArray, int size, int bits) ;

#endif /* end of header gaurd SCATTERED_WRITE_CU_T02OEQEL */
