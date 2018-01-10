/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_DP_NODE_SOLVER_FACTORY_H_
#define __MAPMAP_DP_NODE_SOLVER_FACTORY_H_

#include "header/defines.h"
#include "header/costs.h"

#include "header/optimizer_instances/envelope.h"
#include "header/optimizer_instances/dp_node_solver.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class DPNodeSolverFactory
{
public:
    static DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH> get_enumerative_solver(
        DPNode<COSTTYPE, SIMDWIDTH> * node);
    static DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH> get_solver(
        DPNode<COSTTYPE, SIMDWIDTH> * node);
};

NS_MAPMAP_END

/* include function implementations */
#include "source/optimizer_instances/dp_node_solver_factory.impl.h"

#endif /* __MAPMAP_DP_NODE_SOLVER_FACTORY_H_ */