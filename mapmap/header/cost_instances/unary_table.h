/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_COST_INSTANCES_UNARY_TABLE_H_
#define __MAPMAP_COST_INSTANCES_UNARY_TABLE_H_

#include <vector>

#include "header/costs.h"
#include "header/graph.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class UnaryTable : public UnaryCosts<COSTTYPE, SIMDWIDTH>
{
public:
    UnaryTable(const Graph<COSTTYPE> * graph, 
        const LabelSet<COSTTYPE, SIMDWIDTH> * label_set);
    ~UnaryTable();

    /* Builder functions */
    void set_costs_for_node(const luint_t& node_id, 
        const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& costs);

    /* Interface functions */
    virtual bool supports_enumerable_costs() const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs(const luint_t& node_id, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec) const throw();
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs_enum_offset(
        const luint_t& node_id, const _iv_st<COSTTYPE, SIMDWIDTH>& offset) 
        const throw();

protected:
    luint_t align_offsets(const luint_t& num_nodes, 
        const LabelSet<COSTTYPE, SIMDWIDTH> * label_set);

protected:
    const Graph<COSTTYPE> * m_graph;
    const LabelSet<COSTTYPE, SIMDWIDTH> * m_labelset;
    std::vector<luint_t> m_offsets;
    std::vector<COSTTYPE> m_cost_table;
};

NS_MAPMAP_END

#include "source/cost_instances/unary_table.impl.h"

#endif /* __MAPMAP_COST_INSTANCES_UNARY_TABLE_H_ */
