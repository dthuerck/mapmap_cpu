/**
 * Copyright (C) 2016, Daniel Thuerck
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_table.h"

#include <limits>
#include <iostream>

#include "header/parallel_templates.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
PairwiseTable(
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set,
    const Graph<COSTTYPE> * graph,
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table)
: m_node_dependent(true),
  m_packed_table(packed_table),
  m_label_set(label_set),
  m_graph(graph)
{
    construct_table(true);
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
PairwiseTable(
    const _iv_st<COSTTYPE, SIMDWIDTH>& num_labels,
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table)
: m_node_dependent(false),
  m_num_labels(num_labels),
  m_packed_table(packed_table)
{
    construct_table(false);
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
~PairwiseTable()
{

}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTable<COSTTYPE, SIMDWIDTH>::
node_dependent()
const
{
    return m_node_dependent;
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
PairwiseTable<COSTTYPE, SIMDWIDTH>::
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
    if(!m_node_dependent)
        throw ModeNotSupportedException("PairwiseTable::get_binary_costs("
            "luint_t, _iv_t, luint_t, _iv_t): Node-dependent cost query "
            "not supported for node-independent costs.");

    /* access packed table at specific location by querying edges */
    _s_t<COSTTYPE, SIMDWIDTH> costs[SIMDWIDTH];
   
    /* find edge (id) in adjacency table */
    for(const luint_t& e_id : m_graph->inc_edges(node_id_1))
    {
        const GraphEdge<COSTTYPE>& edge = m_graph->edges()[e_id];

        if((edge.node_a == node_id_1 && edge.node_b == node_id_2) || 
            (edge.node_a == node_id_2 && edge.node_b == node_id_1))
        {
            /* if nodes are queried in wrong order (n1 > n2), swap */
            bool swap = (edge.node_a == node_id_2); 

            /* edge was found, now search table entries separately */
            for(uint_t i = 0; i < SIMDWIDTH; ++i)
            {
                _iv_st<COSTTYPE, SIMDWIDTH> lbl_1 = 
                    iv_extract<COSTTYPE, SIMDWIDTH>(label_vec_1, i);
                _iv_st<COSTTYPE, SIMDWIDTH> lbl_2 = 
                    iv_extract<COSTTYPE, SIMDWIDTH>(label_vec_2, i);

                /* find offsets in both nodes' label sets */
                const _iv_st<COSTTYPE, SIMDWIDTH> limit =
                    std::numeric_limits<_iv_st<COSTTYPE, SIMDWIDTH>>::max(); 
                _iv_st<COSTTYPE, SIMDWIDTH> offset_lbl_1 = limit;
                _iv_st<COSTTYPE, SIMDWIDTH> offset_lbl_2 = limit;

                const _iv_st<COSTTYPE, SIMDWIDTH> node_1_lbl_set_size = 
                    m_label_set->label_set_size(node_id_1);
                const _iv_st<COSTTYPE, SIMDWIDTH> node_2_lbl_set_size = 
                    m_label_set->label_set_size(node_id_2);

                _iv_st<COSTTYPE, SIMDWIDTH> j;
                for(j = 0; j < node_1_lbl_set_size; ++j)
                {
                    if(m_label_set->label_from_offset(node_id_1, j) == lbl_1)
                    {
                        offset_lbl_1 = j;
                        break;
                    }
                }
                for(j = 0; j < node_2_lbl_set_size; ++j)
                {
                    if(m_label_set->label_from_offset(node_id_2, j) == lbl_2)
                    {
                        offset_lbl_2 = j;
                        break;
                    }
                }

                if(offset_lbl_1 < limit && offset_lbl_2 < limit)
                {
                    costs[i] = m_packed_table[m_edge_offset[e_id] + 
                        (swap ? offset_lbl_2 : offset_lbl_1) * 
                        (swap ? node_1_lbl_set_size : node_2_lbl_set_size) + 
                        (swap ? offset_lbl_1 : offset_lbl_2) ];
                }
            }

            return v_load<COSTTYPE, SIMDWIDTH>(costs);
        }
    }

    /* If edge not found, return zero costs to maintain correctness */
    return v_init<COSTTYPE, SIMDWIDTH>();
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTable<COSTTYPE, SIMDWIDTH>:: 
get_binary_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
throw()
{
    /* accessing location-specific costs by this method results in an error */
    if(m_node_dependent)
        throw ModeNotSupportedException("PairwiseTable::get_binary_costs("
            "_iv_t, _iv_t): Node-independent cost query not supported for "
            "node-dependent costs.");

    /* directly query full table */
    _iv_t<COSTTYPE, SIMDWIDTH> addresses = 
        iv_init<COSTTYPE, SIMDWIDTH>(m_num_labels);
    addresses = iv_mult<COSTTYPE, SIMDWIDTH>(label_vec_1, addresses);
    addresses = iv_add<COSTTYPE, SIMDWIDTH>(label_vec_2, addresses);

    /** 
     * In lack of a pre-AVX2 gather instruction, we need to load
     * costs seperately 
     */
    _s_t<COSTTYPE, SIMDWIDTH> costs[SIMDWIDTH];
    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        const uint_t offset = iv_extract<COSTTYPE, SIMDWIDTH>(addresses, i);
        costs[i] = m_packed_table[offset];
    }

    return v_load<COSTTYPE, SIMDWIDTH>(costs);
}

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void 
PairwiseTable<COSTTYPE, SIMDWIDTH>::
construct_table(
    const bool node_dependent)
{
    /* not node dependent: use user-provided pre-packed table */
    if(!node_dependent)
        return;

    /* assume already packed input data, only calculate offsets */
    const luint_t num_edges = m_graph->edges().size(); 

    std::vector<luint_t> edge_sizes(num_edges, 0);
    m_edge_offset.clear();
    m_edge_offset.resize(num_edges);

    /* compute table sizes */
    tbb::blocked_range<luint_t> edge_range(0, num_edges);
    tbb::parallel_for(edge_range, 
    [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t e_id = r.begin(); e_id != r.end(); ++e_id)
        { 
            const GraphEdge<COSTTYPE>& edge = m_graph->edges()[e_id];

            const uint_t lset_a_size = m_label_set->label_set_size(edge.node_a);
            const uint_t lset_b_size = m_label_set->label_set_size(edge.node_b);

            edge_sizes[e_id] = (luint_t) (lset_a_size * lset_b_size);
        }
    });

    /* do an exclusive scan to determine offsets */
    PlusScan<luint_t, luint_t> ex_scan(&edge_sizes[0], &m_edge_offset[0]);
    tbb::parallel_scan(edge_range, ex_scan);
}

NS_MAPMAP_END
