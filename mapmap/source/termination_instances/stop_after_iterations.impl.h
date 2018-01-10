/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/termination_instances/stop_after_iterations.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopAfterIterations<COSTTYPE, SIMDWIDTH>::
StopAfterIterations(
    const luint_t max_iterations,
    const bool count_acyclic,
    const bool count_spanningtree,
    const bool count_multilevel)
: m_max_iterations(max_iterations),
  m_count_acyclic(count_acyclic),
  m_count_spanningtree(count_spanningtree),
  m_count_multilevel(count_multilevel)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopAfterIterations<COSTTYPE, SIMDWIDTH>::
~StopAfterIterations()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
StopAfterIterations<COSTTYPE, SIMDWIDTH>::
check_termination(
    const SolverHistory<COSTTYPE, SIMDWIDTH> * history)
{
    luint_t iterations_done =
        (m_count_acyclic ? history->acyclic_iterations : 0) +
        (m_count_spanningtree ? history->spanningtree_iterations : 0) +
        (m_count_multilevel ? history->multilevel_iterations : 0);

    return (iterations_done >= m_max_iterations);
}

NS_MAPMAP_END
