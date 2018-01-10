/**
 * Copyright (C) 2017, Nick Heppert, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/envelope_instances/pairwise_potts_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
PairwisePottsEnvelope(
    const PairwisePotts<COSTTYPE, SIMDWIDTH> * base)
: m_c(base->get_c())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~PairwisePottsEnvelope()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_m_primes(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
    const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
    const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
    _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
    _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
    _s_t<COSTTYPE, SIMDWIDTH> * scratch)
{
    /* for Potts: just copy for parent's labels, i.e. m' = identity */
    return;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
cost_bound_d()
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
luint_t
PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    return 0;
}

NS_MAPMAP_END
