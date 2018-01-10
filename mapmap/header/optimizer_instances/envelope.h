/**
 * Copyright (C) 2017, Nick Heppert, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_ENVELOPE_H_
#define __MAPMAP_ENVELOPE_H_

#include "header/optimizer_instances/dp_node.h"

NS_MAPMAP_BEGIN

/**
 * Template for implementing linear-time DP optimizers as in

 * Pedro F. Felzenszwalb, and Daniel P. Huttenlocher,
 * "Efficient belief propagation for early vision."
 * International Journal of Computer Vision 70.1 (2006): 41-54.
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
class PairwiseCostsEnvelope
{
public:
    virtual ~PairwiseCostsEnvelope() {};

    virtual void compute_m_primes(
        DPNode<COSTTYPE, SIMDWIDTH> * node,
        const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
        const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
        const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
        _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
        _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
        _s_t<COSTTYPE, SIMDWIDTH> * scratch) = 0;
    virtual _s_t<COSTTYPE, SIMDWIDTH> cost_bound_d() = 0;

    virtual luint_t scratch_bytes_needed(
        DPNode<COSTTYPE, SIMDWIDTH> * node) = 0;
};

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
using PairwiseCostsEnvelope_ptr = std::shared_ptr<PairwiseCostsEnvelope<
    COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>>;

NS_MAPMAP_END

#endif /* __MAPMAP_ENVELOPE_H_ */
