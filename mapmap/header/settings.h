/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_MAPMAP_SETTINGS_H_
#define __MAPMAP_MAPMAP_SETTINGS_H_

#include <mapmap/header/defines.h>
#include <mapmap/header/vector_types.h>

NS_MAPMAP_BEGIN

/* 0s mark the default algorithm */
enum TREE_SAMPLER_ALGORITHM
{
    OPTIMISTIC_TREE_SAMPLER = 0,
    LOCK_FREE_TREE_SAMPLER = 1
};

/**
 * control flow struct (default values for flowgraph in paper)
 *
 * NOTE: despite settings concerning minimum iteration numbers, early
 * termination may be forced by a user-defined termination criterion
 */
struct mapMAP_control
{
    /* switch modes on/off */
    bool use_multilevel;
    bool use_spanning_tree;
    bool use_acyclic;

    /* multilevel settings */
    /* none */

    /* spanning tree settings */
    uint_t spanning_tree_multilevel_after_n_iterations;

    /* acyclic settings */
    bool force_acyclic; /* force using acyclic even if terminated */
    uint_t min_acyclic_iterations;
    bool relax_acyclic_maximal;

    /* settings for tree sampling */
    TREE_SAMPLER_ALGORITHM tree_algorithm;
    bool sample_deterministic;
    uint_t initial_seed;
};

NS_MAPMAP_END

#endif /* __MAPMAP_MAPMAP_SETTINGS_H_ */
