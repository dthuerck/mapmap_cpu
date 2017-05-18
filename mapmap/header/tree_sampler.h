/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_TREE_SAMPLER_H_
#define __MAPMAP_HEADER_TREE_SAMPLER_H_

#include "header/defines.h"

#include "header/tree.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, bool ACYCLIC>
class TreeSampler
{
public:
    TreeSampler() {}
    ~TreeSampler() {}

    virtual void
        select_random_roots(const luint_t k, std::vector<luint_t>& roots) = 0;
    virtual std::unique_ptr<Tree<COSTTYPE>> sample(std::vector<luint_t>& roots,
        bool record_dependencies, bool relax = true) = 0;
};

NS_MAPMAP_END

#endif /* __MAPMAP_HEADER_TREE_SAMPLER_H_ */