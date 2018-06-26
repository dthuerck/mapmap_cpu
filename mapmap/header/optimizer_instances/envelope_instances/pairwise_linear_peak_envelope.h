/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_PAIRWISE_LINEAR_PEAK_ENVELOPE_H_
#define __MAPMAP_PAIRWISE_LINEAR_PEAK_ENVELOPE_H_

#include "header/defines.h"
#include "header/optimizer_instances/envelope.h"
#include "header/cost_instances/pairwise_truncated_linear.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
class PairwiseLinearPeakEnvelope :
    public PairwiseCostsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>
{
public:
    PairwiseLinearPeakEnvelope(const PairwiseLinearPeak<COSTTYPE,
        SIMDWIDTH> * base);
    ~PairwiseLinearPeakEnvelope();

    virtual void compute_m_primes(
        DPNode<COSTTYPE, SIMDWIDTH> * node,
        const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
        const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
        const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
        _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
        _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
        _s_t<COSTTYPE, SIMDWIDTH> * scratch);
    virtual _s_t<COSTTYPE, SIMDWIDTH> cost_bound_d();
    virtual luint_t scratch_bytes_needed(
        DPNode<COSTTYPE, SIMDWIDTH> * node);

protected:
    _s_t<COSTTYPE, SIMDWIDTH> m_label_diff_cap = (_s_t<COSTTYPE, SIMDWIDTH>) 2;
};

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
using PairwiseLinearPeakEnvelope_ptr =
    std::unique_ptr<PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY,
    PAIRWISE>>;

NS_MAPMAP_END

/* include function implementations */
#include "source/optimizer_instances/envelope_instances/pairwise_linear_peak_envelope.impl.h"

#endif /* __MAPMAP_PAIRWISE_LINEAR_PEAK_ENVELOPE_H_ */
