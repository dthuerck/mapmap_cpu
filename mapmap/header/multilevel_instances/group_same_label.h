/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_GROUP_SAME_LABEL_H_
#define __MAPMAP_GROUP_SAME_LABEL_H_

#include <memory>
#include <vector>

#include "header/multilevel.h"

#include "tbb/mutex.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class GroupSameLabel :
    public MultilevelCriterion<COSTTYPE, SIMDWIDTH>
{
public:
    GroupSameLabel();
    ~GroupSameLabel();

    void group_nodes(std::vector<luint_t>& node_in_group,
        const LevelSet<COSTTYPE, SIMDWIDTH> * current_level,
        const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& current_solution,
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& projected_solution);
};

NS_MAPMAP_END

/* include function implementations */
#include "source/multilevel_instances/group_same_label.impl.h"

#endif /* __MAPMAP_GROUP_SAME_LABEL_H_ */
