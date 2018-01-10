/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_STOP_AFTER_TIME_H_
#define __MAPMAP_STOP_AFTER_TIME_H_

#include "header/termination_criterion.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class StopAfterTime : public TerminationCriterion<COSTTYPE, SIMDWIDTH>
{
public:
    StopAfterTime(const luint_t max_seconds,
        const bool compute_before_iteration);
    ~StopAfterTime();

    virtual bool check_termination(const SolverHistory<COSTTYPE,
        SIMDWIDTH> * history);

protected:
    luint_t m_max_seconds;
    bool m_compute_before_iteration;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/termination_instances/stop_after_time.impl.h"

#endif /* __MAPMAP_STOP_AFTER_TIME_H_ */
