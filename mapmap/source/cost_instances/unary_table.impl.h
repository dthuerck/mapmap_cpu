/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <cstring>
#include <cassert>
#include <iostream>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/parallel_do.h"

#include "header/cost_instances/unary_table.h"
#include "header/parallel_templates.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
UnaryTable<COSTTYPE, SIMDWIDTH>::
UnaryTable(
    const luint_t node_id,
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
: m_node_id(node_id),
  m_label_set(label_set),
  m_cost_table(label_set->label_set_size(node_id) + SIMDWIDTH, 0)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
UnaryTable<COSTTYPE, SIMDWIDTH>::
~UnaryTable()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
UnaryTable<COSTTYPE, SIMDWIDTH>::
set_costs(
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& costs)
{
    std::copy(costs.begin(), costs.end(), m_cost_table.begin());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH> *
UnaryTable<COSTTYPE, SIMDWIDTH>::
get_raw_costs()
{
    return m_cost_table.data();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
UnaryTable<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
UnaryTable<COSTTYPE, SIMDWIDTH>::
get_unary_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec)
const
{
    /* use stack array to access parts of the SSE register */
    _s_t<COSTTYPE, SIMDWIDTH> result[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> label_vec_d[SIMDWIDTH];

    iv_store<COSTTYPE, SIMDWIDTH>(label_vec, label_vec_d);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        result[i] = (COSTTYPE) 0;

    /* find all labels in the list and assemble in new vector */
    const _iv_st<COSTTYPE, SIMDWIDTH> lset_size =
        m_label_set->label_set_size(m_node_id);

    uint_t done = 0;
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_id = 0; l_id < lset_size; ++l_id)
    {
        /* stop searching if all labels have been found */
        if(done == SIMDWIDTH)
            break;

        const _iv_st<COSTTYPE, SIMDWIDTH> l_id_label =
            m_label_set->label_from_offset(m_node_id, l_id);

        for(uint_t j = 0; j < SIMDWIDTH; ++j)
        {
            if(l_id_label == label_vec_d[j])
            {
                result[j] = m_cost_table[l_id];
                ++done;
            }
        }
    }

    return v_load<COSTTYPE, SIMDWIDTH>(result);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
UnaryTable<COSTTYPE, SIMDWIDTH>::
get_unary_costs_enum_offset(
    const _iv_st<COSTTYPE, SIMDWIDTH>& offset)
const
{
    const _s_t<COSTTYPE, SIMDWIDTH> * ptr = &m_cost_table[offset];
    return v_load<COSTTYPE, SIMDWIDTH>(ptr);
}

NS_MAPMAP_END
