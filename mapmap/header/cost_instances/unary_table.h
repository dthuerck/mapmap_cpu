/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_UNARY_TABLE_H_
#define __MAPMAP_UNARY_TABLE_H_

#include <vector>

#include "header/costs.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class UnaryTable : public UnaryCosts<COSTTYPE, SIMDWIDTH>
{
public:
    UnaryTable(
        const luint_t node_id, 
        const LabelSet<COSTTYPE, SIMDWIDTH> * label_set);
    ~UnaryTable();

    /* Builder functions */
    void set_costs(const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& costs);
    _s_t<COSTTYPE, SIMDWIDTH> * get_raw_costs();

    /* Interface functions */
    virtual bool supports_enumerable_costs() const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec) const;
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs_enum_offset(
        const _iv_st<COSTTYPE, SIMDWIDTH>& offset) const;

protected:
    const luint_t m_node_id;
    const LabelSet<COSTTYPE, SIMDWIDTH> * m_label_set;

    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_cost_table;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/cost_instances/unary_table.impl.h"

#endif /* __MAPMAP_UNARY_TABLE_H_ */
