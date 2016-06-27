#ifndef CUDA_ERROR_CUH_T02OEQEL
#define CUDA_ERROR_CUH_T02OEQEL

/*
 * =====================================================================================
 *
 *       Filename:  cuda_error.cuh
 *
 *    Description:  Header file for useful cuda error checking macros.
 *
 *        Version:  1.0
 *        Created:  07/06/16 17:22:08
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Shamelessly taken from stack overflow question 14038589 : talonmies

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif /* end of header gaurd CUDA_ERROR_HPP_T02OEQEL */
