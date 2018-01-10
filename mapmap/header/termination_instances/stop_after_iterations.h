/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_STOP_AFTER_ITERATIONS_H_
#define __MAPMAP_STOP_AFTER_ITERATIONS_H_

#include "header/termination_criterion.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class StopAfterIterations : public TerminationCriterion<COSTTYPE, SIMDWIDTH>
{
public:
    StopAfterIterations(
        const luint_t max_iterations,
        const bool count_acyclic,
        const bool count_spanningtree,
        const bool multilevel);
    ~StopAfterIterations();

    virtual bool check_termination(const SolverHistory<COSTTYPE,
        SIMDWIDTH> * history);

protected:
    luint_t m_max_iterations;
    bool m_count_acyclic;
    bool m_count_spanningtree;
    bool m_count_multilevel;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/termination_instances/stop_after_iterations.impl.h"

#endif /* __MAPMAP_STOP_AFTER_ITERATIONS_H_ */
