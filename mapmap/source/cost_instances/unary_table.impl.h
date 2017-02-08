/**
 * Copyright (C) 2016, Daniel Thuerck
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
    const Graph<COSTTYPE> * graph, 
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
: m_graph(graph),
  m_labelset(label_set),
  m_offsets(graph->num_nodes(), 0)
{
    /* compute offsets for padded label table rows */
    const luint_t table_size = align_offsets(graph->num_nodes(), label_set);

    /* resize table accordingly */
    m_cost_table = std::vector<COSTTYPE>(table_size, 0);
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
UnaryTable<COSTTYPE, SIMDWIDTH>::
~UnaryTable()
{
    
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
UnaryTable<COSTTYPE, SIMDWIDTH>::
set_costs_for_node(
    const luint_t& node_id, 
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& costs)
{
    if(node_id >= m_offsets.size())
        return;

    const uint_t len = std::min((_iv_st<COSTTYPE, SIMDWIDTH>) costs.size(), 
        m_labelset->label_set_size(node_id));
    std::memcpy(&m_cost_table[m_offsets[node_id]], &costs[0], len * 
        sizeof(_s_t<COSTTYPE, SIMDWIDTH>));
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
UnaryTable<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return true;
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
UnaryTable<COSTTYPE, SIMDWIDTH>::
get_unary_costs(
    const luint_t& node_id, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec)
const
throw()
{
    if(node_id >= m_offsets.size())
        return v_init<COSTTYPE, SIMDWIDTH>();

    /* use stack array to access parts of the SSE register */
    _s_t<COSTTYPE, SIMDWIDTH> result[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> label_vec_d[SIMDWIDTH];

    iv_store<COSTTYPE, SIMDWIDTH>(label_vec, label_vec_d);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        result[i] = (COSTTYPE) 0;

    /* find all labels in the list and assemble in new vector */
    const _iv_st<COSTTYPE, SIMDWIDTH> lset_size = 
        m_labelset->label_set_size(node_id);

    uint_t done = 0;
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_id = 0; l_id < lset_size; ++l_id)
    {
        /* stop searching if all labels have been found */
        if(done == SIMDWIDTH)
            break;

        const _iv_st<COSTTYPE, SIMDWIDTH> l_id_label = 
            m_labelset->label_from_offset(node_id, l_id);

        for(uint_t j = 0; j < SIMDWIDTH; ++j)
        {
            if(l_id_label == label_vec_d[j])
            {
                result[j] = m_cost_table[m_offsets[node_id] + l_id];
                ++done;
            }
        }
    }

    return v_load<COSTTYPE, SIMDWIDTH>(result);
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
UnaryTable<COSTTYPE, SIMDWIDTH>::
get_unary_costs_enum_offset(
    const luint_t& node_id, 
    const _iv_st<COSTTYPE, SIMDWIDTH>& offset)
const
throw()
{
    if(node_id >= m_offsets.size())
        return v_init<COSTTYPE, SIMDWIDTH>();
   
    const COSTTYPE * ptr = &m_cost_table[m_offsets[node_id]] + offset;
    return v_load<COSTTYPE, SIMDWIDTH>(ptr);
}

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
luint_t
UnaryTable<COSTTYPE, SIMDWIDTH>::
align_offsets(
    const luint_t& num_nodes,
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
{
    /* calculate aligned offsets */
    const luint_t chunk = v_alignment<COSTTYPE, SIMDWIDTH>();

    std::vector<luint_t> padded_label_set(num_nodes, 0);
    tbb::blocked_range<luint_t> node_range(0, num_nodes);

    /* make sure label set size for each node is divisible by the alignment */
    tbb::parallel_for(node_range, [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t node_id = r.begin(); node_id != r.end(); ++node_id)
        {
            const uint_t unpadded_labels = label_set->label_set_size(node_id);
            padded_label_set[node_id] = chunk * DIV_UP(unpadded_labels,
                chunk);
        }
    });

    /* scan to get aligned offsets */
    PlusScan<luint_t, luint_t> ex_scan(&padded_label_set[0], &m_offsets[0]);
    tbb::parallel_scan(node_range, ex_scan);

    return (padded_label_set.back() + m_offsets.back());
}

NS_MAPMAP_END
