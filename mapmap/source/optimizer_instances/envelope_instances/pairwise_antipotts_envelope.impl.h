/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/envelope_instances/pairwise_antipotts_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseAntipottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
PairwiseAntipottsEnvelope(
    const PairwiseAntipotts<COSTTYPE, SIMDWIDTH> * base)
: m_c(base->get_c())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseAntipottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~PairwiseAntipottsEnvelope()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
PairwiseAntipottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_m_primes(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
    const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
    const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
    _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
    _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
    _s_t<COSTTYPE, SIMDWIDTH> * scratch)
{
    const _s_t<COSTTYPE, SIMDWIDTH> c = node->c_node.to_parent_weight *
        ((PAIRWISE *) node->c_pairwise)->get_c();

    /* compute minimum cost and its alternate */
    _s_t<COSTTYPE, SIMDWIDTH> min_fp_cost =
        std::numeric_limits<COSTTYPE>::max();
    _s_t<COSTTYPE, SIMDWIDTH> alt_min_fp_cost =
        std::numeric_limits<COSTTYPE>::max();

    _iv_st<COSTTYPE, SIMDWIDTH> min_fp = 0;
    _iv_st<COSTTYPE, SIMDWIDTH> alt_min_fp = 0;

    /* assume invalid costs are FLT_MAX */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < lbl_union_size; ++l_i)
    {
        const _s_t<COSTTYPE, SIMDWIDTH> cost = m_prime[l_i];

        if(cost < min_fp_cost)
        {
            /* move to second minimum */
            alt_min_fp_cost = min_fp_cost;
            alt_min_fp = min_fp;

            min_fp_cost = cost;
            min_fp = l_i;
        }
        else if(cost < alt_min_fp_cost)
        {
            alt_min_fp_cost = cost;
            alt_min_fp = l_i;
        }
    }

    /* fill m_prime */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < lbl_union_size; ++l_i)
    {
        if(l_i == min_fp)
        {
            m_prime[l_i] = (m_prime[l_i] + c < alt_min_fp_cost) ?
                m_prime[l_i] + c : alt_min_fp_cost;
            m_prime_ix[l_i] = (m_prime[l_i] + c < alt_min_fp_cost) ?
                m_prime_ix[l_i] : alt_min_fp;
        }
        else
        {
            m_prime[l_i] = (m_prime[l_i] + c < min_fp_cost) ?
                m_prime[l_i] + c : min_fp_cost;
            m_prime_ix[l_i] = (m_prime[l_i] + c < min_fp_cost) ?
                m_prime_ix[l_i] : min_fp;
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
cost_bound_d()
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
luint_t
PairwiseAntipottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    return 0;
}

NS_MAPMAP_END
