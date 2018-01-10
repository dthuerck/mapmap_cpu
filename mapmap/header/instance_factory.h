/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_INSTANCE_FACTORY_H_
#define __MAPMAP_INSTANCE_FACTORY_H_

#include "header/defines.h"
#include "header/graph.h"

#include "header/tree_sampler.h"

#include <memory>

NS_MAPMAP_BEGIN

/* 0s mark the default algorithm */
enum TREE_SAMPLER_ALGORITHM
{
    OPTIMISTIC_TREE_SAMPLER = 0,
    LOCK_FREE_TREE_SAMPLER = 1
};

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
class InstanceFactory
{
public:
    static std::unique_ptr<TreeSampler<COSTTYPE, ACYCLIC>>
        get_sampler_instance(
            const TREE_SAMPLER_ALGORITHM& alg,
            Graph<COSTTYPE> * graph,
            const bool deterministic,
            const uint_t seed);
};

NS_MAPMAP_END

/* include function implementations */
#include "source/instance_factory.impl.h"

#endif /* __MAPMAP_INSTANCE_FACTORY_H_ */