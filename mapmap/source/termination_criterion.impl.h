/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/termination_criterion.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
TerminationCriterion<COSTTYPE, SIMDWIDTH>::
TerminationCriterion()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
TerminationCriterion<COSTTYPE, SIMDWIDTH>::
~TerminationCriterion()
{
    
}

NS_MAPMAP_END
