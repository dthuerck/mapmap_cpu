/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_STOP_WHEN_RETURNS_DIMINISH_H_
#define __MAPMAP_STOP_WHEN_RETURNS_DIMINISH_H_

#include "header/termination_criterion.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class StopWhenReturnsDiminish : public TerminationCriterion<COSTTYPE, SIMDWIDTH>
{
public:
    StopWhenReturnsDiminish(const luint_t iteration_span,
        const _s_t<COSTTYPE, SIMDWIDTH> improv_threshold);
    ~StopWhenReturnsDiminish();

    virtual bool check_termination(const SolverHistory<COSTTYPE,
        SIMDWIDTH> * history);

protected:
    luint_t m_iteration_span;
    _s_t<COSTTYPE, SIMDWIDTH> m_improv_threshold;
};

NS_MAPMAP_END

#include "source/termination_instances/stop_when_returns_diminish.impl.h"

#endif /* __MAPMAP_STOP_WHEN_RETURNS_DIMINISH_H_ */