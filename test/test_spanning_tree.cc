/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <set>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#include <tbb/tbb.h>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/tree_sampler.h"
#include "header/instance_factory.h"
#include "test/util_test.h"

NS_MAPMAP_BEGIN

class mapMAPTestSpanningTree :
    public testing::TestWithParam<TREE_SAMPLER_ALGORITHM>
{
public:
    using cost_t = float;

    const uint_t num_components = 4;
    const uint_t component_dim = 20;

public:
    mapMAPTestSpanningTree()
    {

    }

    ~mapMAPTestSpanningTree()
    {

    }

    void
    SetUp()
    {
        /* create a grid graph with a given number of connected components */
        m_graph = createComponentGrid<cost_t>(num_components, component_dim);

        /* execute component retrieval */
        m_graph->update_components();

        /* select one root per component */
        for (uint_t c = 0; c < num_components; ++c)
            m_roots.push_back(c * component_dim * component_dim);

        m_sampler = InstanceFactory<cost_t, false>::get_sampler_instance(
            GetParam(), m_graph.get());

        m_tree = m_sampler->sample(m_roots, true);

        /* retrieve tree's nodes */
        const luint_t num_nodes = m_graph->num_nodes();
        for(luint_t n = 0; n < num_nodes; ++n)
        {
            const TreeNode<cost_t> node = m_tree->node(n);
            if (node.is_root)
                m_actual_roots.push_back(n);
        }
    }

    void
    TearDown()
    {

    }

    std::unique_ptr<Graph<cost_t>> m_graph;
    std::unique_ptr<TreeSampler<cost_t, false>> m_sampler;
    std::vector<luint_t> m_roots;
    std::vector<luint_t> m_actual_roots;

    std::unique_ptr<Tree<cost_t>> m_tree;
};

TEST_P(mapMAPTestSpanningTree, TestHasExactlySpecifiedRoots)
{
    std::set<luint_t> found_roots;

    for (const luint_t& r : m_actual_roots)
    {
        const luint_t old_size = found_roots.size();

        found_roots.insert(r);

        /* make sure no root is found twice */
        ASSERT_EQ(found_roots.size(), old_size + 1);

        auto found_it = std::find(found_roots.begin(), found_roots.end(),
            r);

        /* assure this root was actually specified */
        ASSERT_NE(found_it, found_roots.end());
    }

    /* make sure all specified have been found */
    ASSERT_EQ(found_roots.size(), m_roots.size());
}

TEST_P(mapMAPTestSpanningTree, TestMatchingRootCover)
{
    std::vector<luint_t> roots;

    m_sampler->select_random_roots(num_components, roots);

    std::set<luint_t> components;
    for (const luint_t& r : roots)
        components.insert(m_graph->components()[r]);

    ASSERT_EQ(components.size(), num_components);
}

TEST_P(mapMAPTestSpanningTree, TestIncompleteRootCover)
{
    std::vector<luint_t> roots;

    m_sampler->select_random_roots(num_components - 1, roots);

    std::set<luint_t> components;
    for (const luint_t& r : roots)
        components.insert(m_graph->components()[r]);

    ASSERT_EQ(components.size(), num_components);
}

TEST_P(mapMAPTestSpanningTree, TestIsComplete)
{
    const luint_t num_nodes = num_components * component_dim *
        component_dim;
    std::vector<unsigned char> visited(num_nodes, 0u);

    for(const luint_t r : m_actual_roots)
        BFSWithCustomFunc<cost_t>(m_tree.get(), r,
            [&visited](const Tree<cost_t> * tree, const luint_t node_id)
            {
                visited[node_id] = 1u;
            });

    for (luint_t n = 0; n < num_nodes; ++n)
        ASSERT_GE(visited[n], 1u);
}

TEST_P(mapMAPTestSpanningTree, TestDoesNotViolateComponents)
{
    for(const luint_t r : m_actual_roots)
        BFSWithCustomFunc<cost_t>(m_tree.get(), r,
            [&](const Tree<cost_t> * tree, const luint_t node_id)
            {
                const TreeNode<cost_t> node = tree->node(node_id);
                const luint_t parent_node = node.parent_id;

                /**
                 * Simply check if each parent/child link
                 * connects only nodes from the same component.
                 */
                ASSERT_EQ(m_graph->components()[node_id],
                    m_graph->components()[parent_node]);

                for (luint_t i = 0; i < node.degree; ++i)
                {
                    ASSERT_EQ(m_graph->components()[node_id],
                        m_graph->components()[node.children_ids[i]]);
                }
            });
}

TEST_P(mapMAPTestSpanningTree, TestIsDAG)
{
    const luint_t num_nodes = num_components * component_dim *
        component_dim;
    std::vector<luint_t> visited(num_nodes, 0u);

    for(const luint_t r : m_actual_roots)
        BFSWithCustomFunc<cost_t>(m_tree.get(), r,
            [&visited](const Tree<cost_t> * tree, const luint_t node_id)
            {
                ++visited[node_id];
            });

    /* tree is DAG <=> each node visited at most once */
    for (luint_t n = 0; n < num_nodes; ++n)
        ASSERT_LE(visited[n], 1u);
}

TEST_P(mapMAPTestSpanningTree, TestDependenciesAreComplete)
{
    const luint_t num_nodes = num_components * component_dim *
        component_dim;

    for(luint_t i = 0; i < num_nodes; ++i)
    {
        /* collect parent, children and dependencies */
        const TreeNode<cost_t>& node = m_tree->node(i);

        std::set<luint_t> inc_set;

        /* parent - at least for non-roots */
        if(node.parent_id != i)
            inc_set.insert(node.parent_id);

        /* children */
        for(luint_t i = 0; i < node.degree; ++i)
        {
            const luint_t c = node.children_ids[i];
            ASSERT_TRUE(inc_set.find(c) == inc_set.end());
            inc_set.insert(c);
        }

        /* dependencies */
        for(luint_t i = 0; i < node.dependency_degree; ++i)
        {
            const luint_t d = node.dependency_ids[i];
            ASSERT_TRUE(inc_set.find(d) == inc_set.end());
            inc_set.insert(d);
        }

        /* check if the collected set covers the incidence list */
        ASSERT_EQ(inc_set.size(), m_graph->inc_edges(i).size());

        for(const luint_t& n : m_graph->inc_edges(i))
        {
            const GraphEdge<cost_t> n_edge = m_graph->edges()[n];
            const luint_t other_n = (n_edge.node_a == i) ? n_edge.node_b :
                n_edge.node_a;

            ASSERT_TRUE(inc_set.find(other_n) != inc_set.end());
        }
    }
}

INSTANTIATE_TEST_CASE_P(SpanningTreeTest,
    mapMAPTestSpanningTree,
    ::testing::Values(OPTIMISTIC_TREE_SAMPLER, LOCK_FREE_TREE_SAMPLER));

NS_MAPMAP_END