/**
 * Copyright (C) 2016, Daniel Thuerck
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
PairwiseTruncatedLinear(
    const _s_t<COSTTYPE, SIMDWIDTH>& label_diff_cap)
: m_label_diff_cap(label_diff_cap)
{

}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
~PairwiseTruncatedLinear()
{

}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
node_dependent()
const
{
    return false;
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
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

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>::
get_binary_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
throw()
{
    _iv_t<COSTTYPE, SIMDWIDTH> diff = iv_sub<COSTTYPE, SIMDWIDTH>(label_vec_1,
        label_vec_2);
    diff = iv_abs<COSTTYPE, SIMDWIDTH>(diff);

    const _v_t<COSTTYPE, SIMDWIDTH> vdiff = iv_convert_v<COSTTYPE, SIMDWIDTH>(
        diff);
    return v_min<COSTTYPE, SIMDWIDTH>(vdiff, v_init<COSTTYPE, SIMDWIDTH>(
        m_label_diff_cap));
}

NS_MAPMAP_END
