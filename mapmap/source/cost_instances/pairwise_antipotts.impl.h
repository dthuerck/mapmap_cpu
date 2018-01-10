/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_antipotts.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
PairwiseAntipotts()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
PairwiseAntipotts(
    const _s_t<COSTTYPE, SIMDWIDTH>& c)
: m_c(c)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
PairwiseAntipotts(
    const std::initializer_list<_s_t<COSTTYPE, SIMDWIDTH>>& ps)
{
    if(ps.size() > 0)
        m_c = ps.begin()[0];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
~PairwiseAntipotts()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_c()
const
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_label_diff_cap()
const
{
    return 0;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(new
        PairwiseAntipotts<COSTTYPE, SIMDWIDTH>(m_c));
    return std::move(uptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    if(typeid(*costs) != typeid(PairwiseAntipotts<COSTTYPE,
        SIMDWIDTH>))
        return false;

    const PairwiseAntipotts<COSTTYPE, SIMDWIDTH> * oth =
        dynamic_cast<const PairwiseAntipotts<COSTTYPE, SIMDWIDTH>*>(
        costs);

    return (m_c == oth->get_c());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    _v_t<COSTTYPE, SIMDWIDTH> lv1 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_1);
    _v_t<COSTTYPE, SIMDWIDTH> lv2 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_2);
    _v_t<COSTTYPE, SIMDWIDTH> eq = v_eq<COSTTYPE, SIMDWIDTH>(lv1, lv2);

    return v_and<COSTTYPE, SIMDWIDTH>(v_init<COSTTYPE, SIMDWIDTH>(m_c), eq);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2)
const
{
    throw new ModeNotSupportedException("PairwiseAntipotts does not support " \
        "enumerable costs");
}

NS_MAPMAP_END