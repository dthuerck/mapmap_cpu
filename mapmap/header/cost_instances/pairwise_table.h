/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_PAIRWISE_TABLE_H_
#define __MAPMAP_PAIRWISE_TABLE_H_

#include <memory>
#include <vector>

#include "header/defines.h"
#include "header/costs.h"
#include "header/graph.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwiseTable : public PairwiseCosts<COSTTYPE, SIMDWIDTH>
{
public:
    /* these two constructors allocate memory */
    PairwiseTable(
        const luint_t node_a,
        const luint_t node_b,
        const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set);
    PairwiseTable(
        const luint_t node_a,
        const luint_t node_b,
        const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set,
        const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table);
        
    /* this - third - constructor only saves a raw pointer */
    PairwiseTable(
        const luint_t node_a,
        const luint_t node_b,
        const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set,
        _s_t<COSTTYPE, SIMDWIDTH> * packed_table);
    ~PairwiseTable();

    void set_costs(const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table);
    _s_t<COSTTYPE, SIMDWIDTH> * get_raw_costs();

    virtual std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> copy() const;

    virtual bool supports_enumerable_costs() const;
    virtual bool eq(const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs) const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_pairwise_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const;
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_pairwise_costs_enum_offsets(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2) const;

protected:
    const luint_t m_node_a;
    const luint_t m_node_b;
    const LabelSet<COSTTYPE, SIMDWIDTH> * m_lbl_set;

    _s_t<COSTTYPE, SIMDWIDTH> * m_packed_table;
    std::unique_ptr<_s_t<COSTTYPE, SIMDWIDTH>[]> m_packed_table_storage;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/cost_instances/pairwise_table.impl.h"

#endif /* __MAPMAP_PAIRWISE_TABLE_H_ */
