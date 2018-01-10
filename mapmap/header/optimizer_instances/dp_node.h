/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_OPTIMIZER_INSTANCES_DP_NODE_H_
#define __MAPMAP_HEADER_OPTIMIZER_INSTANCES_DP_NODE_H_

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/vector_math.h"
#include "header/parallel_templates.h"
#include "header/costs.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
struct DPNode
{
    /* input data */
    TreeNode<COSTTYPE> c_node;
    const Graph<COSTTYPE> * c_graph;
    const LabelSet<COSTTYPE, SIMDWIDTH> * c_labels;

    const UnaryCosts<COSTTYPE, SIMDWIDTH> * c_unary;
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * c_pairwise;

    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>*> * c_child_values;
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>*> * c_child_labels;

    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> * c_assignment;

    std::vector<const PairwiseCosts<COSTTYPE, SIMDWIDTH>*> c_dep_costs;
    bool respect_dependencies;

    /* output data */
    _s_t<COSTTYPE, SIMDWIDTH> * c_opt_values;
    _iv_st<COSTTYPE, SIMDWIDTH> * c_opt_labels;

    /* scratch space, externally allocated */
    _s_t<COSTTYPE, SIMDWIDTH> * c_scratch;
};

NS_MAPMAP_END

#endif /* __MAPMAP_HEADER_OPTIMIZER_INSTANCES_DP_NODE_H_ */