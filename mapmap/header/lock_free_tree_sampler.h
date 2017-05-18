/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_LOCK_FREE_TREE_SAMPLER_H_
#define __MAPMAP_HEADER_LOCK_FREE_TREE_SAMPLER_H_

#include "header/defines.h"
#include "header/graph.h"
#include "header/tree.h"
#include "header/tree_sampler.h"

#include "tbb/tbb_atomic.h"

/**
 * *****************************************************************************
 * **************************** ColoredQueueBunch ******************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
class ColoredQueueBunch
{
public:
    ColoredQueueBunch(const Graph<COSTTYPE> * graph);
    ~ColoredQueueBunch()

    void push_to(const luint_t qu, const luint_t elem);
    luint_t queue_size(const luint_t qu);
    luint_t queue_capacity(const luint_t qu);
    luint_t * queue(const luint_t qu);
    luint_t num_queues();

    void reset();

protected:
    void init();

protected:
    Graph<COSTTYPE> * m_graph;

    luint_t m_num_qu;
    std::vector<luint_t> m_data;
    std::vector<luint_t *> m_qu_start;
    std::vector<tbb::atomic<luint_t>> m_qu_pos;
};

/**
 * *****************************************************************************
 * *************************** LockFreeTreeSampler *****************************
 * *****************************************************************************
 */

/**
 * Note: assumes the input graph has previously been colored.
 */
template<typename COSTTYPE, bool ACYCLIC>
class LockFreeTreeSampler : public TreeSampler<COSTTYPE>
{
public:
    LockFreeTreeSampler(const Graph<COSTTYPE> * graph);
    ~LockFreeTreeSampler();

    void select_random_roots(const luint_t k, std::vector<luint_t>& roots);
    std::unique_ptr<Tree<COSTTYPE>> sample(std::vector<luint_t>& roots,
        bool record_dependencies, bool relax = true);

public:
    const uint_t p_chunk_size = 16;

protected:
    const Graph<COSTTYPE> * m_graph;

    ColoredQueueBunch<COSTTYPE> m_qu_a;
    ColoredQueueBunch<COSTTYPE> m_qu_b;
};


#endif /* __MAPMAP_HEADER_LOCK_FREE_TREE_SAMPLER_H_ */