/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/termination_instances/stop_after_time.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopAfterTime<COSTTYPE, SIMDWIDTH>::
StopAfterTime(
    const luint_t max_seconds,
    const bool compute_before_iteration)
: m_max_seconds(max_seconds),
  m_compute_before_iteration(compute_before_iteration)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopAfterTime<COSTTYPE, SIMDWIDTH>::
~StopAfterTime()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
StopAfterTime<COSTTYPE, SIMDWIDTH>::
check_termination(
    const SolverHistory<COSTTYPE, SIMDWIDTH> * history)
{
    /* calculate average iteration length */
    const luint_t num_iteration = history->acyclic_iterations +
        history->spanningtree_iterations +
        history->multilevel_iterations;
    const luint_t runtime = history->time_history->back();

    const COSTTYPE avg_iteration = runtime / (COSTTYPE) num_iteration;

    return (runtime + (m_compute_before_iteration ? avg_iteration : 0) >=
        m_max_seconds);
}

NS_MAPMAP_END
