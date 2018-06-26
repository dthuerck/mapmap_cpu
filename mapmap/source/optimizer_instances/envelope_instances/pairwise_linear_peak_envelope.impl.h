/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/envelope_instances/pairwise_linear_peak_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
PairwiseLinearPeakEnvelope(
    const PairwiseLinearPeak<COSTTYPE, SIMDWIDTH> * base)
: m_label_diff_cap(base->get_label_diff_cap())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~PairwiseLinearPeakEnvelope()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_m_primes(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
    const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
    const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
    _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
    _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
    _s_t<COSTTYPE, SIMDWIDTH> * scratch)
{
    /* save initial m_prime for bw pass */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_prime_in(m_prime, m_prime +
        lbl_union_size);

    const _s_t<COSTTYPE, SIMDWIDTH> c = node->c_node.to_parent_weight *
        ((PAIRWISE *) node->c_pairwise)->get_c();
    const _s_t<COSTTYPE, SIMDWIDTH> d = m_label_diff_cap;

    /* initialize envelope */
    for(_iv_st<COSTTYPE, SIMDWIDTH> pos = 0; pos < lbl_union_size; ++pos)
    {
        m_prime[pos] += c * d;
    }

    /* forward pass */
    _s_t<COSTTYPE, SIMDWIDTH> save_val;
    _iv_st<COSTTYPE, SIMDWIDTH> save_ix;

    _iv_st<COSTTYPE, SIMDWIDTH> model_ix = 0;
    _iv_st<COSTTYPE, SIMDWIDTH> model_label = lbl_union[model_ix];
    _iv_st<COSTTYPE, SIMDWIDTH> test_ptr = model_ix + 1;
    _iv_st<COSTTYPE, SIMDWIDTH> test_label = lbl_union[test_ptr];
    for(_iv_st<COSTTYPE, SIMDWIDTH> pos = 0; pos < lbl_union_size; ++pos)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> pos_lbl = lbl_union[pos];

        /* catch up to model with minimum value */
        _s_t<COSTTYPE, SIMDWIDTH> model_h = std::max(m_prime_in[model_ix],
            m_prime_in[model_ix] + c * (d - std::abs(model_label - pos_lbl)));
        _s_t<COSTTYPE, SIMDWIDTH> next_h = std::max(m_prime_in[test_ptr],
            m_prime_in[test_ptr] + c * (d - std::abs(test_label - pos_lbl)));
        while(test_ptr < pos)
        {
            /* skip models that are bounded away */
            if(m_prime_in[model_ix] > m_prime_in[test_ptr])
            {
                /* use monotonous decrease of lower envelope */
                if(next_h > model_h)
                {
                    break;
                }

                /* only use another model if not cut out */
                model_ix = test_ptr;
                model_h = next_h;
                model_label = test_label;
            }


            /* cache values for next model */
            ++test_ptr;
            test_label = lbl_union[test_ptr];
            next_h = std::max(m_prime_in[test_ptr],
                m_prime_in[test_ptr] + c * (d - std::abs(test_label - pos_lbl)));
        }


        /* evaluate entry with model function and local function */
        const _s_t<COSTTYPE, SIMDWIDTH> pos_h = m_prime_in[pos] + c * d;

        /* save entry */
        save_val = (pos_h < model_h) ? pos_h : model_h;
        save_ix = (pos_h < model_h) ? pos : model_ix;

        if(save_val < m_prime[pos])
        {
            m_prime[pos] = save_val;
            m_prime_ix[pos] = save_ix;
        }
    }

    /* backward pass */
    model_ix = lbl_union_size - 1;
    model_label = lbl_union[model_ix];
    test_ptr = model_ix - 1;
    test_label = lbl_union[test_ptr];
    for(_iv_st<COSTTYPE, SIMDWIDTH> pos = lbl_union_size - 1; pos >= 0; --pos)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> pos_lbl = lbl_union[pos];

        /* catch up to model with minimum value */
        _s_t<COSTTYPE, SIMDWIDTH> model_h = std::max(m_prime_in[model_ix],
            m_prime_in[model_ix] + c * (d - std::abs(model_label - pos_lbl)));
        _s_t<COSTTYPE, SIMDWIDTH> next_h = std::max(m_prime_in[test_ptr],
            m_prime_in[test_ptr] + c * (d - std::abs(test_label - pos_lbl)));
        while(test_ptr > pos)
        {
            /* skip models that are bounded away */
            if(m_prime_in[model_ix] > m_prime_in[test_ptr])
            {
                /* use monotonous decrease of lower envelope */
                if(next_h > model_h)
                {
                    break;
                }

                /* only use another model if not cut out */
                model_ix = test_ptr;
                model_h = next_h;
                model_label = test_label;
            }

            /* cache values for next model */
            --test_ptr;
            test_label = lbl_union[test_ptr];
            next_h = std::max(m_prime_in[test_ptr],
                m_prime_in[test_ptr] + c * (d - std::abs(test_label - pos_lbl)));
        }


        /* evaluate entry with model function and local function */
        const _s_t<COSTTYPE, SIMDWIDTH> pos_h = m_prime_in[pos] + c * d;

        /* save entry */
        save_val = (pos_h < model_h) ? pos_h : model_h;
        save_ix = (pos_h < model_h) ? pos : model_ix;

        if(save_val < m_prime[pos])
        {
            m_prime[pos] = save_val;
            m_prime_ix[pos] = save_ix;
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
cost_bound_d()
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
luint_t
PairwiseLinearPeakEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    return 0;
}

NS_MAPMAP_END
