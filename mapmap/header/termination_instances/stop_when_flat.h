/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_STOP_WHEN_FLAT_H_
#define __MAPMAP_STOP_WHEN_FLAT_H_

#include "header/termination_criterion.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class StopWhenFlat : public TerminationCriterion<COSTTYPE, SIMDWIDTH>
{
public:
    StopWhenFlat(const luint_t num_allowed_flat_iterations,
        const bool reset_after_improvement);
    ~StopWhenFlat();

    virtual bool check_termination(const SolverHistory<COSTTYPE,
        SIMDWIDTH> * history);

protected:
    luint_t m_num_allowed_flat_iterations;
    bool m_reset_after_improvement;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/termination_instances/stop_when_flat.impl.h"

#endif /* __MAPMAP_STOP_WHEN_FLAT_H_ */
