/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/tree_sampler.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, bool ACYCLIC>
TreeSampler<COSTTYPE, ACYCLIC>::
TreeSampler(
    Graph<COSTTYPE> * graph)
: TreeSampler<COSTTYPE, ACYCLIC>(graph, false, 0)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
TreeSampler<COSTTYPE, ACYCLIC>::
TreeSampler(
    Graph<COSTTYPE> * graph,
    const bool deterministic,
    const uint_t initial_seed)
: m_graph(graph),
  m_rnd_dev(),
  m_deterministic(deterministic),
  m_initial_seed(initial_seed)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
TreeSampler<COSTTYPE, ACYCLIC>::
~TreeSampler()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
TreeSampler<COSTTYPE, ACYCLIC>::
build_component_lists()
{
    const luint_t num_nodes = m_graph->num_nodes();
    const luint_t num_components = m_graph->num_components();
    const std::vector<luint_t>& components = m_graph->components();

    m_component_lists.clear();
    m_component_lists.resize(num_components);

    for(luint_t i = 0; i < num_components; ++i)
        m_component_lists[i].reserve(num_nodes / num_components);

    for(luint_t n = 0; n < num_nodes; ++n)
        m_component_lists[components[n]].push_back(n);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
TreeSampler<COSTTYPE, ACYCLIC>::
create_adj_acc()
{
    /* make private copy of nodes' adjacency list */
    std::vector<luint_t> m_adj_size(m_graph->num_nodes());
    m_adj_offsets.resize(m_graph->num_nodes());
    tbb::blocked_range<luint_t> node_range(0, m_graph->num_nodes(), 128u);

    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                m_adj_size[i] = m_graph->inc_edges(i).size();
        });

    PlusScan<luint_t, luint_t> ex_scan(&m_adj_size[0], &m_adj_offsets[0]);
    tbb::parallel_scan(node_range, ex_scan);

    const luint_t adj_size = 2 * m_graph->edges().size();
    m_adj.resize(adj_size);

    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                const std::vector<luint_t>& edges = m_graph->inc_edges(i);
                std::memcpy(&m_adj[m_adj_offsets[i]],
                    &edges[0], edges.size() * sizeof(luint_t));
            }
        });
}

NS_MAPMAP_END