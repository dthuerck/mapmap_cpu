/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_COST_INSTANCES_PAIRWISE_TABLE_H_
#define __MAPMAP_HEADER_COST_INSTANCES_PAIRWISE_TABLE_H_

#include <memory>
#include <vector>

#include "header/defines.h"
#include "header/graph.h"
#include "header/costs.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwiseTable : public PairwiseCosts<COSTTYPE, SIMDWIDTH>
{
public:
    PairwiseTable(const LabelSet<COSTTYPE, SIMDWIDTH> * label_set, 
        const Graph<COSTTYPE> * graph,
        const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table);
    PairwiseTable(const _iv_st<COSTTYPE, SIMDWIDTH>& num_labels,
        const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table);
    ~PairwiseTable();

    virtual bool node_dependent() const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(
        const luint_t& node_id_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
        const luint_t& node_id_2, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw();
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw();

protected:
    void construct_table(const bool node_dependent);

protected:
    bool m_node_dependent;
    _iv_st<COSTTYPE, SIMDWIDTH> m_num_labels;
    
    std::vector<luint_t> m_edge_offset;
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_packed_table;
    
    const LabelSet<COSTTYPE, SIMDWIDTH> * m_label_set;
    const Graph<COSTTYPE> * m_graph;
};

NS_MAPMAP_END

#include "source/cost_instances/pairwise_table.impl.h"

#endif /* __MAPMAP_HEADER_COST_INSTANCES_PAIRWISE_TABLE_H_ */
