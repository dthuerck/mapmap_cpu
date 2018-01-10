/**
 * Copyright (C) 2017, Nick Heppert, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
PairwiseTruncatedLinearEnvelope(
    const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH> * base)
: m_label_diff_cap(base->get_label_diff_cap())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~PairwiseTruncatedLinearEnvelope()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_m_primes(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
    const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
    const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
    _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
    _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
    _s_t<COSTTYPE, SIMDWIDTH> * scratch)
{
    /* Foward pass */
    const _s_t<COSTTYPE, SIMDWIDTH> c = node->c_node.to_parent_weight *
        ((PAIRWISE *) node->c_pairwise)->get_c();

    _iv_st<COSTTYPE, SIMDWIDTH> last_label = lbl_union[0];
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 1; l_i < lbl_union_size; ++l_i)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> this_label = lbl_union[l_i];
        const _s_t<COSTTYPE, SIMDWIDTH> accum_c = _s_t<COSTTYPE, SIMDWIDTH>(
            this_label - last_label) * c;

        if(m_prime[l_i - 1] + accum_c < m_prime[l_i])
        {
            m_prime[l_i] = m_prime[l_i - 1] + accum_c;
            m_prime_ix[l_i] = m_prime_ix[l_i - 1];
        }

        last_label = this_label;
    }

    /* Backward pass */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = lbl_union_size - 2; l_i >= 0; --l_i)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> this_label = lbl_union[l_i];
        const _s_t<COSTTYPE, SIMDWIDTH> accum_c = _s_t<COSTTYPE, SIMDWIDTH>(
                last_label - this_label) * c;

        if(m_prime[l_i + 1] + accum_c < m_prime[l_i])
        {
            m_prime[l_i] = m_prime[l_i + 1] + accum_c;
            m_prime_ix[l_i] = m_prime_ix[l_i + 1];
        }

        last_label = this_label;
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
cost_bound_d()
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
luint_t
PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    return 0;
}

NS_MAPMAP_END
