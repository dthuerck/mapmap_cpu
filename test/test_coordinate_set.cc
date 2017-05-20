/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/tree_sampler.h"
#include "header/instance_factory.h"
#include "test/util_test.h"

NS_MAPMAP_BEGIN

/*
 * General note: all properties that hold for spanning trees -- except for
 * covering all nodes, obviously -- also hold for (acyclic) coordinate sets,
 * thus these are not tested again here.
 * This test only tests the difference (acyclicity) to spanning trees.
 */

class mapMAPTestCoordinateSet :
    public testing::TestWithParam<TREE_SAMPLER_ALGORITHM>
{
public:
    using cost_t = float;

    const uint_t num_components = 2;
    const uint_t component_dim = 100;

public:
    mapMAPTestCoordinateSet()
    {

    }

    ~mapMAPTestCoordinateSet()
    {

    }

    void
    SetUp()
    {
        /* create a grid graph with a given number of connected components */
        m_graph = createComponentGrid<cost_t>(num_components, component_dim);

        /* select one root per component */
        for (uint_t c = 0; c < num_components; ++c)
            m_roots.push_back(c * component_dim * component_dim);

        m_sampler = InstanceFactory<cost_t, true>::get_sampler_instance(
            GetParam(), m_graph.get());

        m_tree = m_sampler->sample(m_roots, true, false);

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
    std::unique_ptr<TreeSampler<cost_t, true>> m_sampler;
    std::vector<luint_t> m_roots;
    std::vector<luint_t> m_actual_roots;

    std::unique_ptr<Tree<cost_t>> m_tree;
};

TEST_P(mapMAPTestCoordinateSet, TestIsAcyclic)
{
    /*
     * Test acyclicity locally: for each graph node in the tree, each
     * neighboured node that is in the tree must be either parent or
     * child of the node under test.
     */

    /* Phase I : mark all nodes included in the tree */
    const luint_t num_nodes = m_graph->num_nodes();
    std::vector<uint_t> in_tree(num_nodes, 0);
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        TreeNode<cost_t> node = m_tree->node(i);
        in_tree[i] = (node.is_in_tree ? 1u : 0u);
    }

    /* Phase II : check the mentioned criterion */
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        TreeNode<cost_t> node = m_tree->node(i);

        if(in_tree[i] == 0)
            continue;

        /* Retrieve all adjacent nodes in the tree */
        std::vector<luint_t> adjacent_in_tree;
        for(const luint_t& e_id : m_graph->inc_edges(i))
        {
            const GraphEdge<cost_t>& e = m_graph->edges()[e_id];
            const luint_t other_node = (e.node_a ==
                i ? e.node_b : e.node_a);

            if(in_tree[other_node] > 0u)
                adjacent_in_tree.push_back(other_node);
        }

        /* Check if retrieved nodes are parent or children */
        for(const luint_t& o_n : adjacent_in_tree)
        {
            auto found = std::find(&node.children_ids[0],
                &node.children_ids[node.degree], o_n);

            if(!(node.parent_id == o_n || found !=
                &node.children_ids[node.degree]))
            {
                std::cout << "Problem in " << o_n << " (parent " <<
                    m_tree->node(o_n).parent_id << ") with " <<
                    i << " (in turn with parent " <<
                    node.parent_id << ", degree " << node.degree << ")" << std::endl;

                std::cout << "Children " << i << ": ";
                for(luint_t c = 0; c < node.degree; ++c)
                    std::cout << node.children_ids[c] << " ";
                std::cout << std::endl;

                std::cout << "Children " << o_n << ": ";
                for(luint_t c = 0; c < m_tree->node(o_n).degree; ++c)
                    std::cout << m_tree->node(o_n).children_ids[c] << " ";
                std::cout << std::endl;
            }

            ASSERT_TRUE(node.parent_id == o_n || found !=
                &node.children_ids[node.degree]);
        }
    }
}

TEST_P(mapMAPTestCoordinateSet, TestIsMaximal)
{
    /* Phase I : mark all nodes included in the tree */
    const luint_t num_nodes = m_graph->num_nodes();
    std::vector<uint_t> in_tree(num_nodes, 0);
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        TreeNode<cost_t> node = m_tree->node(i);
        in_tree[i] = (node.is_in_tree ? 1u : 0u);
    }

    /*
     * Phase II : count neighboring nodes in tree for free nodes and
     * test that marker is >= 2 (otherwise the node could have been added
     * as coordinate.
     */
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        TreeNode<cost_t> node = m_tree->node(i);

        if (node.is_in_tree)
            continue;

        /* Count neighboring nodes which are in the tree */
        luint_t marker = 0;
        for (const luint_t& e_id : m_graph->inc_edges(i))
        {
            const GraphEdge<cost_t>& e = m_graph->edges()[e_id];
            const luint_t other_node = (e.node_a ==
                i ? e.node_b : e.node_a);
            TreeNode<cost_t> o_node = m_tree->node(other_node);

            if (o_node.is_in_tree)
                ++marker;
        }

        ASSERT_GE(marker, 2u);
    }
}

INSTANTIATE_TEST_CASE_P(AcyclicTest,
    mapMAPTestCoordinateSet,
    ::testing::Values(OPTIMISTIC_TREE_SAMPLER, LOCK_FREE_TREE_SAMPLER));

NS_MAPMAP_END
