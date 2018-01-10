/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_truncated_linear.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedLinear()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedLinear(
    const _s_t<COSTTYPE, SIMDWIDTH>& c,
    const _s_t<COSTTYPE, SIMDWIDTH>& label_diff_cap)
: m_c(c),
  m_label_diff_cap(label_diff_cap)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
PairwiseTruncatedLinear(
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
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
~PairwiseTruncatedLinear()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
get_c()
const
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
get_label_diff_cap()
const
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(new
        PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>(m_c, m_label_diff_cap));
    return std::move(uptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    if(typeid(*costs) != typeid(PairwiseTruncatedLinear<COSTTYPE, 
        SIMDWIDTH>))
        return false;

    const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH> * oth = 
        dynamic_cast<const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>*>(
        costs);

    return (m_c == oth->get_c() && 
        m_label_diff_cap == oth->get_label_diff_cap());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    _iv_t<COSTTYPE, SIMDWIDTH> diff = iv_sub<COSTTYPE, SIMDWIDTH>(label_vec_1,
        label_vec_2);
    diff = iv_abs<COSTTYPE, SIMDWIDTH>(diff);

    const _v_t<COSTTYPE, SIMDWIDTH> vdiff = v_mult<COSTTYPE, SIMDWIDTH>(
        iv_convert_v<COSTTYPE, SIMDWIDTH>(diff), v_init<COSTTYPE, SIMDWIDTH>(
        m_c));
    return v_min<COSTTYPE, SIMDWIDTH>(vdiff, v_init<COSTTYPE, SIMDWIDTH>(
        m_label_diff_cap));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2)
const
{
    throw new ModeNotSupportedException("PairwiseTruncatedLinear does not " \
        "support enumerable costs");
}

NS_MAPMAP_END
