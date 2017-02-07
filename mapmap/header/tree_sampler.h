/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_TREE_SAMPLER_H_
#define __MAPMAP_TREE_SAMPLER_H_

#include <vector>
#include <memory>
#include <random>

#include "tbb/concurrent_vector.h"

#include "header/defines.h"
#include "header/graph.h"
#include "header/tree.h"
#include "header/vector_types.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, bool ACYCLIC>
class TreeSampler
{
public:
    TreeSampler(const Graph<COSTTYPE> * graph);
    ~TreeSampler();

    /**
     * Select around k roots, satisfying the following conditions:
     * - each component is covered by at least one root,
     * - if ACYCLIC = true, roots cannot be adjacent nodes,
     * - number of components <= number of roots <= number of nodes.
     *
     * After each component is covered, the remaining nodes
     * are sampled according to the components' size.
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
        const bool record_dependencies);

public:
    const uint_t p_chunk_size = 16;

protected:
    void build_component_lists();
    void create_adj_acc();
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

    const Graph<COSTTYPE> * m_graph;
    std::vector<std::vector<luint_t>> m_component_lists;

    std::vector<luint_t> m_adj_offsets;
    std::vector<luint_t> m_adj;

    std::random_device m_rnd_dev;

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
};

template<typename COSTTYPE, bool ACYCLIC>
using TreeSampler_ptr = std::shared_ptr<TreeSampler<COSTTYPE, ACYCLIC>>;

NS_MAPMAP_END

#include "source/tree_sampler.impl.h"

#endif /* __MAPMAP_TREE_SAMPLER_H_ */
