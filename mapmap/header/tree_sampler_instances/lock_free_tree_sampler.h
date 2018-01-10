/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_LOCK_FREE_TREE_SAMPLER_H_
#define __MAPMAP_LOCK_FREE_TREE_SAMPLER_H_

#include "header/defines.h"
#include "header/graph.h"
#include "header/tree.h"
#include "header/tree_sampler.h"

#include "tbb/atomic.h"
#include "tbb/concurrent_vector.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * **************************** ColoredQueueBunch ******************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
class ColoredQueueBunch
{
public:
    ColoredQueueBunch(Graph<COSTTYPE> * graph);
    ColoredQueueBunch(Graph<COSTTYPE> * graph, const bool deterministic);
    ~ColoredQueueBunch();

    void push_to(const luint_t qu, const luint_t elem);
    void replace_queue(const luint_t qu, const luint_t * new_qu,
        const luint_t new_size);

    luint_t queue_size(const luint_t qu);
    luint_t queue_capacity(const luint_t qu);
    luint_t * queue(const luint_t qu);
    luint_t num_queues();
    luint_t size();

    void reset();

protected:
    void init();

protected:
    Graph<COSTTYPE> * m_graph;

    luint_t m_num_qu;
    std::vector<luint_t> m_data;
    std::vector<luint_t *> m_qu_start;
    std::vector<tbb::atomic<luint_t>> m_qu_pos;

    tbb::atomic<luint_t> m_size;
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
class LockFreeTreeSampler : public TreeSampler<COSTTYPE, ACYCLIC>
{
public:
    LockFreeTreeSampler(Graph<COSTTYPE> * graph);
    LockFreeTreeSampler(Graph<COSTTYPE> * graph, const bool
        deterministic, const uint_t initial_seed);
    ~LockFreeTreeSampler();

    void select_random_roots(const luint_t k, std::vector<luint_t>& roots);
    std::unique_ptr<Tree<COSTTYPE>> sample(std::vector<luint_t>& roots,
        bool record_dependencies, bool relax = true);

public:
    const uint_t p_chunk_size = 16;

protected:
    void sample_phase_I();
    void sample_phase_II();
    void sample_rescue();

protected:
    /* colored queue for input data */
    ColoredQueueBunch<COSTTYPE> m_qu;

    /* use iteration count as deterministic seed */
    uint_t m_it;

    /* save this rounds' active color */
    luint_t m_cur_col;

    /* markers for acyclic growing */
    std::vector<tbb::atomic<luint_t>> m_markers;

    /* determine which nodes have been added to the respective queue */
    std::vector<tbb::atomic<char>> m_queued;

    /* saves nodes added in last iteration */
    std::vector<luint_t> m_new;
    tbb::atomic<luint_t> m_new_size;

    /* temporary single-color ouput queue */
    std::vector<luint_t> m_qu_out;
    tbb::atomic<luint_t> m_qu_out_pos;

    /* counter: nodes remaining */
    tbb::atomic<luint_t> m_rem_nodes;

    /* counter: remaining degree per node */
    std::vector<luint_t> m_rem_degrees;

    /* save resulting tree */
    std::unique_ptr<Tree<COSTTYPE>> m_tree;

    /* how many potential tree-edges to save per vertex */
    const luint_t m_buf_edges = 4;

    /* limit for rescue roots */
    const luint_t m_max_rescue = 4;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/tree_sampler_instances/lock_free_tree_sampler.impl.h"

#endif /* __MAPMAP_LOCK_FREE_TREE_SAMPLER_H_ */