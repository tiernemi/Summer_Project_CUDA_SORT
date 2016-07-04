#ifndef STRIDED_ITERATORS_HPP_UEWXJLCF
#define STRIDED_ITERATORS_HPP_UEWXJLCF

/*
 * =====================================================================================
 *
 *       Filename:  strided_iterators.hpp
 *
 *    Description:  Strided iterator template from m.s question 31161835 on stack
 *                  overflow. 
 *
 *        Version:  1.0
 *        Created:  17/06/16 11:01:37
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  
 *
 * =====================================================================================
 */

#include <thrust/tuple.h>

template <typename Iterators>
__host__ __device__
thrust::zip_iterator<thrust::tuple<Iterators, Iterators, Iterators> > zip(Iterators a, Iterators b, Iterators c) {
    return thrust::make_zip_iterator(thrust::make_tuple(a,b,c));
}

template <typename Iterator>
struct StridedRange {
	typedef typename thrust::iterator_difference<Iterator>::type difference_type ;
    struct stride_functor : public thrust::unary_function<difference_type,difference_type> {
        difference_type stride ;
        stride_functor(difference_type stride) : stride(stride) {}
        __host__ __device__
		difference_type operator()(const difference_type& i) const { 
			return stride * i ;
		}
	} ;

    typedef typename thrust::counting_iterator<difference_type> CountingIterator ;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator ;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator> PermutationIterator ;
    // type of the StridedRange iterator
    typedef PermutationIterator iterator ;
 public:
     // construct StridedRange for the range [first,last)
    StridedRange(Iterator first, Iterator last, difference_type stride) 
		: first(first), last(last), stride(stride) {} ;
    iterator begin(void) const {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    } ;
    iterator end(void) const {
        return begin() + ((last - first) + (stride - 1)) / stride;
    } ;
 protected:
    Iterator first ;
    Iterator last ;
    difference_type stride ;
};


#endif /* end of include guard: STRIDED_ITERATORS_HPP_UEWXJLCF */
