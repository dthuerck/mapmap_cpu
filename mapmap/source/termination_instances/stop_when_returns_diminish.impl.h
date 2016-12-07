/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_TERMINATION_INSTANCES_STOP_WHEN_RETURNS_DIMINISH_IMPL_H_
#define __MAPMAP_HEADER_TERMINATION_INSTANCES_STOP_WHEN_RETURNS_DIMINISH_IMPL_H_

#include "header/termination_instances/stop_when_returns_diminish.h"

#include <iostream>

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
StopWhenReturnsDiminish<COSTTYPE, SIMDWIDTH>::
StopWhenReturnsDiminish(
    const luint_t iteration_span,
    const _s_t<COSTTYPE, SIMDWIDTH> improv_threshold)
: m_iteration_span(iteration_span),
  m_improv_threshold(improv_threshold)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
StopWhenReturnsDiminish<COSTTYPE, SIMDWIDTH>::
~StopWhenReturnsDiminish()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
bool
StopWhenReturnsDiminish<COSTTYPE, SIMDWIDTH>::
check_termination(
    const SolverHistory<COSTTYPE, SIMDWIDTH> * history)
{
    _s_t<COSTTYPE, SIMDWIDTH> oldest_val, newest_val; 
    const luint_t hist_size = history->energy_history->size();

    /* have at least the minimum number of iterations done */
    if(hist_size < m_iteration_span)
        return false;

    /* determine improvement in the last couple of iterations */
    newest_val = history->energy_history->back();

    luint_t oldest_pos = 0;
    if(hist_size > m_iteration_span)
        oldest_pos = (hist_size - 1) - m_iteration_span;

    oldest_val = (*history->energy_history)[oldest_pos];

    const _s_t<COSTTYPE, SIMDWIDTH> improv =
        (oldest_val - newest_val) / oldest_val;

    return (improv < m_improv_threshold);
}

NS_MAPMAP_END

#endif 
/* __MAPMAP_HEADER_TERMINATION_INSTANCES_STOP_WHEN_RETURNS_DIMINISH_IMPL_H_ */