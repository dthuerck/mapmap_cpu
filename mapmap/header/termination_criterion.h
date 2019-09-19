/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_TERMINATION_CRITERION_H_
#define __MAPMAP_TERMINATION_CRITERION_H_

#include <vector>
#include <memory>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * ***************************** SolverHistory *********************************
 * *****************************************************************************
 */

enum SolverMode
{
    SOLVER_ACYCLIC,
    SOLVER_SPANNINGTREE,
    SOLVER_MULTILEVEL
};

template<typename COSTTYPE, uint_t SIMDWIDTH>
struct SolverHistory
{
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>> * energy_history;
    const std::vector<luint_t> * time_history;
    const std::vector<SolverMode> * mode_history;

    luint_t acyclic_iterations;
    luint_t spanningtree_iterations;
    luint_t multilevel_iterations;
};

/**
 * *****************************************************************************
 * ************************** TerminationCriterion *****************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
class TerminationCriterion
{
public:
    TerminationCriterion();
    virtual ~TerminationCriterion();

    virtual bool check_termination(const SolverHistory<COSTTYPE,
        SIMDWIDTH> * history) = 0;
};

NS_MAPMAP_END

#include "source/termination_criterion.impl.h"

#endif /* __MAPMAP_TERMINATION_CRITERION_H_ */
