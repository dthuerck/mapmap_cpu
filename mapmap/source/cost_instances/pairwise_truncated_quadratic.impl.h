/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_truncated_quadratic.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedQuadratic()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedQuadratic(
    const _s_t<COSTTYPE, SIMDWIDTH>& c,
    const _s_t<COSTTYPE, SIMDWIDTH>& label_diff_cap)
: m_c(c),
  m_label_diff_cap(label_diff_cap)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedQuadratic(
    const std::initializer_list<_s_t<COSTTYPE, SIMDWIDTH>>& ps)
{
    if(ps.size() > 0)
        m_c = ps.begin()[0];

    if(ps.size() > 1)
        m_label_diff_cap = ps.begin()[1];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
~PairwiseTruncatedQuadratic()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
get_c()
const
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
get_label_diff_cap()
const
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(new
        PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>(m_c, m_label_diff_cap));
    return std::move(uptr);
}

/* ************************************************************************** */


template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    if(typeid(*costs) != typeid(PairwiseTruncatedQuadratic<COSTTYPE, 
        SIMDWIDTH>))
        return false;

    const PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH> * oth = 
        dynamic_cast<const PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>*>(
        costs);

    return (m_c == oth->get_c() && 
        m_label_diff_cap == oth->get_label_diff_cap());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    _iv_t<COSTTYPE, SIMDWIDTH> diff = iv_sub<COSTTYPE, SIMDWIDTH>(label_vec_1,
        label_vec_2);
    diff = iv_abs<COSTTYPE, SIMDWIDTH>(diff);

    const _v_t<COSTTYPE, SIMDWIDTH> qdiff = v_mult<COSTTYPE, SIMDWIDTH>(
        v_mult<COSTTYPE, SIMDWIDTH>(
        iv_convert_v<COSTTYPE, SIMDWIDTH>(diff),
        iv_convert_v<COSTTYPE, SIMDWIDTH>(diff)),
        v_init<COSTTYPE, SIMDWIDTH>(m_c));
    return v_min<COSTTYPE, SIMDWIDTH>(qdiff, v_init<COSTTYPE, SIMDWIDTH>(
        m_label_diff_cap));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2)
const
{
    throw new ModeNotSupportedException("PairwiseTruncatedQuadratic does " \
        "not support enumerable costs");
}

NS_MAPMAP_END
