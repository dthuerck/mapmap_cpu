/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_INSTANCE_FACTORY_H_
#define __MAPMAP_HEADER_INSTANCE_FACTORY_H_

#include "header/defines.h"

/* 0s mark the default algorithm */
enum TREE_SAMPLER_ALGORITHM
{
    OPTIMISTIC_TREE_SAMPLER = 0,
    LOCK_FREE_TREE_SAMPLER = 1
};

/* ************************************************************************** */

template<typename COSTTYPE>
class InstanceFactory
{
public:
    static std::unique_ptr<TreeSampler<COSTTYPE>>& get_sampler_instance(
        const TREE_SAMPLER_ALGORITHM& alg,
        const Graph<COSTTYPE> * graph,
        const bool acyclic);
};

#endif /* __MAPMAP_HEADER_INSTANCE_FACTORY_H_ */