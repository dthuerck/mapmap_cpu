/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_potts.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwisePotts<COSTTYPE, SIMDWIDTH>::
PairwisePotts()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwisePotts<COSTTYPE, SIMDWIDTH>::
PairwisePotts(
    const _s_t<COSTTYPE, SIMDWIDTH>& c)
: m_c(c)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwisePotts<COSTTYPE, SIMDWIDTH>::
PairwisePotts(
    const std::initializer_list<_s_t<COSTTYPE, SIMDWIDTH>>& ps)
{
    if(ps.size() > 0)
        m_c = ps.begin()[0];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwisePotts<COSTTYPE, SIMDWIDTH>::
~PairwisePotts()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwisePotts<COSTTYPE, SIMDWIDTH>::
get_c()
const
{
    return m_c;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwisePotts<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwisePotts<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(new
        PairwisePotts<COSTTYPE, SIMDWIDTH>(m_c));
    return std::move(uptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwisePotts<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    if(typeid(*costs) != typeid(PairwisePotts<COSTTYPE, 
        SIMDWIDTH>))
        return false;

    const PairwisePotts<COSTTYPE, SIMDWIDTH> * oth = 
        dynamic_cast<const PairwisePotts<COSTTYPE, SIMDWIDTH>*>(
        costs);

    return (m_c == oth->get_c());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwisePotts<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    _v_t<COSTTYPE, SIMDWIDTH> lv1 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_1);
    _v_t<COSTTYPE, SIMDWIDTH> lv2 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_2);
    _v_t<COSTTYPE, SIMDWIDTH> neq = v_eq<COSTTYPE, SIMDWIDTH>(lv1, lv2);
    neq = v_not<COSTTYPE, SIMDWIDTH>(neq);

    return v_and<COSTTYPE, SIMDWIDTH>(v_init<COSTTYPE, SIMDWIDTH>(m_c), neq);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwisePotts<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2)
const
{
    throw new ModeNotSupportedException("PairwisePotts does not support " \
        "enumerable costs");
}

NS_MAPMAP_END
