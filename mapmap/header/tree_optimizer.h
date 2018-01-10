/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_TREE_OPTIMIZER_H_
#define __MAPMAP_TREE_OPTIMIZER_H_

#include <memory>
#include <vector>

#include "header/defines.h"
#include "header/tree.h"
#include "header/vector_types.h"
#include "header/costs.h"
#include "header/cost_bundle.h"
#include "header/graph.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class TreeOptimizer
{
public:
    TreeOptimizer();
    ~TreeOptimizer();

    /* prerequisites for optimization */
    void set_graph(const Graph<COSTTYPE> * graph);
    void set_tree(const Tree<COSTTYPE> * tree);
    void set_label_set(const LabelSet<COSTTYPE, SIMDWIDTH> * label_set);
    void set_costs(const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle);
    void use_dependencies(const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>&
        current_solution);

    /* evaluate objective for a solution */
    _s_t<COSTTYPE, SIMDWIDTH> objective(
        const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution);

    /* optimization procedure - returns new objective value and solution */
    virtual _s_t<COSTTYPE, SIMDWIDTH> optimize(
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution) throw() = 0;

protected:
    bool data_complete();

protected:
    /* optimization input data */
    const Graph<COSTTYPE> * m_graph;
    const Tree<COSTTYPE> * m_tree;
    const LabelSet<COSTTYPE, SIMDWIDTH> * m_label_set;
    const CostBundle<COSTTYPE, SIMDWIDTH> * m_cbundle;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_current_assignment;

    /* status bits - check for data completeness */
    bool m_has_graph;
    bool m_has_tree;
    bool m_has_label_set;
    bool m_has_costs;
    bool m_uses_dependencies;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/tree_optimizer.impl.h"

#endif /* __MAPMAP_TREE_OPTIMIZER_H_ */
