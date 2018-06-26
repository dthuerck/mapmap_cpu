/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_linear_peak.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
PairwiseLinearPeak()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
PairwiseLinearPeak(
    const _s_t<COSTTYPE, SIMDWIDTH>& c,
    const _s_t<COSTTYPE, SIMDWIDTH>& label_diff_cap)
: m_c(c),
  m_label_diff_cap(label_diff_cap)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
PairwiseLinearPeak(
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
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
~PairwiseLinearPeak()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
get_c()
const
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
get_label_diff_cap()
const
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(new
        PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>(m_c, m_label_diff_cap));
    return std::move(uptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    if(typeid(*costs) != typeid(PairwiseLinearPeak<COSTTYPE, 
        SIMDWIDTH>))
        return false;

    const PairwiseLinearPeak<COSTTYPE, SIMDWIDTH> * oth = 
        dynamic_cast<const PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>*>(
        costs);

    return (m_c == oth->get_c() && 
        m_label_diff_cap == oth->get_label_diff_cap());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    _iv_t<COSTTYPE, SIMDWIDTH> diff = iv_sub<COSTTYPE, SIMDWIDTH>(label_vec_1,
        label_vec_2);
    diff = iv_abs<COSTTYPE, SIMDWIDTH>(diff);

    _v_t<COSTTYPE, SIMDWIDTH> vdiff = v_sub<COSTTYPE, SIMDWIDTH>(
        v_init<COSTTYPE, SIMDWIDTH>(m_label_diff_cap),
        iv_convert_v<COSTTYPE, SIMDWIDTH>(diff));
    vdiff = v_mult<COSTTYPE, SIMDWIDTH>(vdiff,
        v_init<COSTTYPE, SIMDWIDTH>(m_c));
    return v_max<COSTTYPE, SIMDWIDTH>(vdiff, v_init<COSTTYPE, SIMDWIDTH>(0));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2)
const
{
    throw new ModeNotSupportedException("PairwiseLinearPeak does not support " \
        "enumerable costs");
}

NS_MAPMAP_END
