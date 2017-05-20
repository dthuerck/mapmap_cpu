/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include "header/defines.h"
#include "header/color.h"
#include "test/util_test.h"

NS_MAPMAP_BEGIN

class mapMAPTestColoring : public testing::Test
{
public:
    using cost_t = float;

    const uint_t num_components = 4;
    const uint_t component_dim = 100;

public:
    mapMAPTestColoring()
    {

    }

    ~mapMAPTestColoring()
    {

    }

    void
    SetUp()
    {
        /* create a grid graph with a given number of connected components */
        m_graph = createComponentGrid<cost_t>(num_components, component_dim);

        /* Color graph */
        Color<cost_t> col(*m_graph);
        col.color_graph(m_coloring);
    }

    void
    TearDown()
    {
    }

    std::unique_ptr<Graph<cost_t>> m_graph;
    std::vector<luint_t> m_coloring;
};

TEST_F(mapMAPTestColoring, TestValid)
{
    /* Detect collisions between the coloring of any nodes */
    const luint_t num_nodes = m_graph->num_nodes();
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        const std::vector<luint_t>& edges = m_graph->inc_edges(i);

        for(const luint_t e_id : edges)
        {
            const GraphEdge<cost_t>& e = m_graph->edges()[e_id];
            ASSERT_NE(m_coloring[e.node_a], m_coloring[e.node_b]);
        }
    }
}

NS_MAPMAP_END