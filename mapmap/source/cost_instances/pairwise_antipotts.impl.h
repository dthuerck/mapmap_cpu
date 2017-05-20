/**
 * Copyright (C) 2016, Daniel Thuerck
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
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
~PairwiseAntipotts()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
bool
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
node_dependent()
const
{
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_binary_costs(
    const luint_t& node_id_1, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
    const luint_t& node_id_2, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
throw()
{
    /** 
     * accessing location-independent costs by this method results 
     * in an error 
     */
    throw ModeNotSupportedException("PairwiseTable::get_binary_costs("
        "luint_t, _iv_t, luint_t, _iv_t): Node-dependent cost query "
        "not supported for node-independent costs.");
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseAntipotts<COSTTYPE, SIMDWIDTH>::
get_binary_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
throw()
{
    _v_t<COSTTYPE, SIMDWIDTH> lv1 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_1);
    _v_t<COSTTYPE, SIMDWIDTH> lv2 = iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(
        label_vec_2);
    _v_t<COSTTYPE, SIMDWIDTH> eq = v_eq<COSTTYPE, SIMDWIDTH>(lv1, lv2);

    return v_and<COSTTYPE, SIMDWIDTH>(v_init<COSTTYPE, SIMDWIDTH>(m_c), eq);
}

NS_MAPMAP_END
