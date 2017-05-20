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
#include "header/graph.h"
#include "test/util_test.h"

NS_MAPMAP_BEGIN

class mapMAPTestGraph : public ::testing::Test
{
public:
    using cost_t = float;
    const uint_t num_components = 4;
    const uint_t component_dim = 10;

public:
    mapMAPTestGraph()
    {
    }

    ~mapMAPTestGraph()
    {
    }

    void
    SetUp()
    {
        m_graph = createComponentGrid<cost_t>(num_components, component_dim);
    }

    void 
    TearDown()
    {
    }

protected:
    std::unique_ptr<Graph<cost_t>> m_graph;
};

TEST_F(mapMAPTestGraph, TestInternalValues)
{
    ASSERT_EQ(m_graph->m_nodes.size(), num_components * 
        component_dim * component_dim);
    ASSERT_EQ(m_graph->m_edges.size(), num_components * 
        ((component_dim - 1) * component_dim + 
        component_dim * (component_dim - 1)));
}

TEST_F(mapMAPTestGraph, TestComponentSearch)
{
    m_graph->update_components();
    ASSERT_EQ(m_graph->num_components(), 4u);

    const std::vector<luint_t>& components = m_graph->components();
    std::vector<luint_t> cmp_ids(num_components);
    for (uint_t i = 0; i < num_components; ++i)
        cmp_ids[i] = components[i * component_dim * component_dim];

    for (uint_t i = 0; i < num_components; ++i)
    {
        for (uint_t j = 0; j < component_dim * component_dim; ++j)
        {
            ASSERT_EQ(components[i * component_dim * component_dim + j],
                cmp_ids[i]);
        }
    }
}

NS_MAPMAP_END