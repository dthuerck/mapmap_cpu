/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/instance_factory.h"

#include "header/optimistic_tree_sampler.h"
#include "header/lock_free_tree_sampler.h"

template<typename COSTTYPE>
std::unique_ptr<TreeSampler<COSTTYPE>>
InstanceFactory<COSTTYPE>::
get_sampler_instance(
    const TREE_SAMPLER_ALGORITHM& alg,
    const Graph<COSTTYPE> * graph,
    const bool acyclic)
{
    if(alg == OPTIMISTIC_TREE_SAMPLER)
    {
        if(acyclic)
            return std::make_unique<TreeSampler<COSTTYPE, true>(
                new OptimisticTreeSampler<COSTTYPE, true>(graph));
        else
            return std::make_unique<TreeSampler<COSTTYPE, false>(
                new OptimisticTreeSampler<COSTTYPE, false>(graph));
    }

    if(alg == LOCK_FREE_TREE_SAMPLER)
    {
        if(acyclic)
            return std::make_unique<TreeSampler<COSTTYPE, true>(
                new LockFreeTreeSampler<COSTTYPE, true>(graph));
        else
            return std::make_unique<TreeSampler<COSTTYPE, false>(
                new LockFreeTreeSampler<COSTTYPE, false>(graph));
    }

    return nullptr;
}