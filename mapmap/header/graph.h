/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_GRAPH_H_
#define __MAPMAP_GRAPH_H_

#include <vector>
#include <memory>
#include <exception>

#include "header/defines.h"
#include "header/vector_types.h"

NS_MAPMAP_BEGIN

struct GraphNode
{
    std::vector<luint_t> incident_edges;
};

template<typename WEIGHTTYPE>
struct GraphEdge
{
    luint_t node_a;
    luint_t node_b;
    scalar_t<WEIGHTTYPE> weight;
};

template<typename COSTTYPE>
class Graph
{
public:
    Graph(const luint_t num_nodes);
    ~Graph();

    void add_edge(const luint_t node_a, const luint_t node_b,
        const scalar_t<COSTTYPE> weight) throw();
    const std::vector<GraphNode>& nodes() const;
    const luint_t num_nodes() const;
    const std::vector<luint_t>& inc_edges(const luint_t node) const;
    const std::vector<GraphEdge<COSTTYPE>>& edges() const;
    const luint_t num_edges() const;

    void update_components();
    const std::vector<luint_t>& components() const;
    const luint_t num_components() const;

    /* store and use coloring information */
    void set_coloring(const std::vector<luint_t>& coloring);
    const std::vector<luint_t>& get_coloring();
    bool was_colored();

    /* necessary for deterministic processing */
    void sort_incidence_lists();

protected:
#if defined(BUILD_FOR_TEST)
    #include <gtest/gtest_prod.h>

    FRIEND_TEST(mapMAPTestGraph, TestInternalValues);
    FRIEND_TEST(mapMAPTestGraph, TestComponentSearch);
#endif
    luint_t m_num_nodes;
    std::vector<GraphNode> m_nodes;
    std::vector<GraphEdge<scalar_t<COSTTYPE>>> m_edges;
    std::vector<luint_t> m_components;
    luint_t m_num_components;

    std::vector<luint_t> m_coloring;
    bool m_was_colored;
};

NS_MAPMAP_END

#include "source/graph.impl.h"

#endif /* __MAPMAP_GRAPH_H_ */