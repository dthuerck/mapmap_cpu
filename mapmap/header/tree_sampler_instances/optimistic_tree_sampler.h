/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_OPTIMISTIC_TREE_SAMPLER_H_
#define __MAPMAP_HEADER_OPTIMISTIC_TREE_SAMPLER_H_

#include <vector>
#include <memory>
#include <random>

#include "tbb/concurrent_vector.h"

#include "header/defines.h"
#include "header/graph.h"
#include "header/tree.h"
#include "header/vector_types.h"
#include "header/tree_sampler.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, bool ACYCLIC>
class OptimisticTreeSampler : public TreeSampler<COSTTYPE, ACYCLIC>
{
public:
    OptimisticTreeSampler(Graph<COSTTYPE> * graph);
    OptimisticTreeSampler(Graph<COSTTYPE> * graph, const bool
        deterministic, const uint_t initial_seed);
    ~OptimisticTreeSampler();

    /**
     * Select around k roots, satisfying the following conditions:
     * - each component is covered by at least one root,
     * - number of components <= number of roots <= number of nodes.
     *
     * After each component is covered, the remaining nodes
     * are sampled according to the components' size.
     *
     * Conflict checking is deferred to a later stage.
     */
    void select_random_roots(const luint_t k, std::vector<luint_t>& roots);

    /**
     * Samples a maximal (acyclic) coordinate set, given a set
     * of roots to grow from. To achieve maximality, additional roots
     * may be included.
     *
     * Note: If a component is not covered by the root set,
     * it is also left out of the tree.
     *
     * Transfers ownership to the caller.
     */
    std::unique_ptr<Tree<COSTTYPE>> sample(std::vector<luint_t>& roots,
        bool record_dependencies, bool relax = true);

public:
    const uint_t p_chunk_size = 16;

protected:
    void sample_phase_I();
    void sample_phase_II();
    void sample_phase_III();
    void sample_rescue();
    void compute_dependencies();

protected:
#if defined(BUILD_FOR_TEST)
    #include <gtest/gtest_prod.h>

    FRIEND_TEST(mapMAPTestCoordinateSet, TestIsAcyclic);
    FRIEND_TEST(mapMAPTestCoordinateSet, TestIsMaximal);
#endif

    /* iteration counter for seed generation */
    uint_t m_it;

    tbb::concurrent_vector<luint_t>* m_w_in;
    tbb::concurrent_vector<luint_t>* m_w_out;

    tbb::concurrent_vector<luint_t> m_w_a;
    tbb::concurrent_vector<luint_t> m_w_b;

    tbb::concurrent_vector<luint_t> m_w_new;
    tbb::concurrent_vector<std::pair<luint_t, luint_t>> m_w_conflict;

    std::unique_ptr<Tree<COSTTYPE>> m_tree;
    std::vector<luint_t> m_rem_degrees;
    std::vector<tbb::atomic<luint_t>> m_markers;
    std::vector<tbb::atomic<unsigned char>> m_node_locks;
    std::vector<tbb::atomic<unsigned char>> m_in_queue;
    tbb::atomic<luint_t> m_rem_nodes;
};

template<typename COSTTYPE, bool ACYCLIC>
using OptimisticTreeSampler_ptr =
    std::shared_ptr<OptimisticTreeSampler<COSTTYPE, ACYCLIC>>;

NS_MAPMAP_END

/* include function implementations */
#include "source/tree_sampler_instances/optimistic_tree_sampler.impl.h"

#endif /* __MAPMAP_HEADER_OPTIMISTIC_TREE_SAMPLER_H_ */
