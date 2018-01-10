/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/termination_instances/stop_when_flat.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopWhenFlat<COSTTYPE, SIMDWIDTH>::
StopWhenFlat(
    const luint_t num_allowed_flat_iterations,
    const bool reset_after_improvement)
: m_num_allowed_flat_iterations(num_allowed_flat_iterations),
  m_reset_after_improvement(reset_after_improvement)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
StopWhenFlat<COSTTYPE, SIMDWIDTH>::
~StopWhenFlat()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
StopWhenFlat<COSTTYPE, SIMDWIDTH>::
check_termination(
    const SolverHistory<COSTTYPE, SIMDWIDTH> * history)
{
    const luint_t num_energy_iterations = history->energy_history->size();

    if(num_energy_iterations < 2)
        return false;

    /* count flat iterations */
    luint_t flat_iterations = 0;
    for(luint_t j = num_energy_iterations - 1; j > 0; --j)
    {
        const bool is_flat = ((*history->energy_history)[j] ==
            (*history->energy_history)[j - 1]);
        const bool is_improvement = ((*history->energy_history)[j] <=
            (*history->energy_history)[j - 1]);

        if(is_flat)
            ++flat_iterations;
        if(is_improvement && m_reset_after_improvement)
            break;
    }

    return (flat_iterations >= m_num_allowed_flat_iterations);
}

NS_MAPMAP_END
