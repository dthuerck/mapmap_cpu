/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/tree.h"

#include <exception>
#include <functional>
#include <numeric>
#include <iostream>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_scan.h"

#include "header/parallel_templates.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * *********************************** Tree ************************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
Tree<COSTTYPE>::
Tree(
    const luint_t graph_num_nodes,
    const luint_t graph_num_edges)
: m_num_nodes(0),
  m_graph_nodes(graph_num_nodes),
  m_enable_modify(true),
  m_parent_id(graph_num_nodes, invalid_luint_t),
  m_parent_edge_id(graph_num_nodes, invalid_luint_t),
  m_degree(graph_num_nodes, (luint_t) 0u),
  m_children_offset(graph_num_nodes, (luint_t) 0u),
  m_children_list(graph_num_nodes, invalid_luint_t),
  m_weight(graph_num_nodes, (COSTTYPE) 1.0),
  m_dependency_degree(graph_num_nodes, (luint_t) 0u),
  m_dependency_offset(graph_num_nodes, (luint_t) 0u),
  m_dependency_list(graph_num_edges, invalid_luint_t),
  m_dependency_edge_id_list(graph_num_edges, invalid_luint_t),
  m_dependency_weight(graph_num_edges, (COSTTYPE) 1.0)
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
Tree<COSTTYPE>::
~Tree()
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
const TreeNode<COSTTYPE>
Tree<COSTTYPE>::
node(
    const luint_t& node_id)
const
{
    if (node_id >= m_graph_nodes)
        throw std::out_of_range("Node ID out of range.");

    TreeNode<COSTTYPE> node;

    node.node_id = node_id;
    node.parent_id = m_parent_id[node_id];
    node.to_parent_edge_id = m_parent_edge_id[node_id];
    node.to_parent_weight = m_weight[node_id];
    node.degree = m_degree[node_id];
    node.dependency_degree = m_dependency_degree[node_id];

    const luint_t offset = m_children_offset[node_id];
    node.children_ids = &m_children_list[0] + offset;
    const luint_t d_offset = m_dependency_offset[node_id];
    node.dependency_ids = &m_dependency_list[0] + d_offset;
    node.dependency_edge_ids = &m_dependency_edge_id_list[0] + d_offset;
    node.dependency_weights = &m_dependency_weight[0] + d_offset;

    node.is_in_tree = (node.parent_id != invalid_luint_t);

    return node;
}

/* ************************************************************************** */

template<typename COSTTYPE>
const luint_t
Tree<COSTTYPE>::
num_nodes()
const
{
    return m_num_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
const luint_t
Tree<COSTTYPE>::
num_graph_nodes()
const
{
    return m_graph_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
void
Tree<COSTTYPE>::
finalize(
    const bool compute_dependencies,
    const Graph<COSTTYPE> * graph)
{
    if(!m_enable_modify)
        return;

    tbb::blocked_range<luint_t> node_range(0, m_graph_nodes);

    /* Count number of nodes in the tree (parent_id != invalid) */
    m_num_nodes = tbb::parallel_reduce(node_range, (luint_t) 0,
        [&](const tbb::blocked_range<luint_t>& r, luint_t reduced) -> luint_t
        {
            luint_t my_contrib = reduced;

            for(luint_t i = r.begin(); i != r.end(); ++i)
                my_contrib += (m_parent_id[i] != invalid_luint_t);

            return my_contrib;
        },
        std::plus<luint_t>());

    /**
     * Collect children in parallel (from parent ids)
     */

    /* collect degree per node */
    Histogram<luint_t, luint_t> hist(&m_parent_id[0], m_graph_nodes);
    m_degree = hist(node_range);

    /* set 0 for nodes not in the tree and subtract 1 per root */
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                if(m_parent_id[i] == i)
                    --m_degree[i];

                if(m_parent_id[i] == invalid_luint_t)
                    m_degree[i] = 0;
            }
        });

    /* scan to get offsets */
    PlusScan<luint_t, luint_t> scan(&m_degree[0], &m_children_offset[0]);
    tbb::parallel_scan(node_range, scan);

    /* save children in list and copy node degree */
    std::vector<tbb::atomic<luint_t>> loc_offsets(m_graph_nodes);
    std::fill(loc_offsets.begin(), loc_offsets.end(), 0u);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                const luint_t my_parent = m_parent_id[i];

                /* do not record as child if node is a root */
                if ((my_parent != invalid_luint_t) && (my_parent != i))
                {
                    const luint_t my_local_offset = loc_offsets[my_parent].
                        fetch_and_increment();
                    const luint_t my_offset = m_children_offset[my_parent];

                    m_children_list[my_offset + my_local_offset] = i;
                }
            }
        });

    /**
     * Collect dependencies in parallel (infer from parent ids)
     */

    if(!compute_dependencies)
    {
        std::fill(m_dependency_degree.begin(), m_dependency_degree.end(), 0);
        return;
    }

    /**
     * determine dependency degree per node: node degree in graph,
     * subtract number of children and subtract 1 parent if node
     * is not the root
     *
     * Nodes that are not in the tree can't have dependencies and
     * shall be skipped.
     */
     tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& n)
        {
            for(luint_t i = n.begin(); i != n.end(); ++i)
            {
                m_dependency_degree[i] = 0;

                /* skip inactive nodes */
                if(m_parent_id[i] == invalid_luint_t)
                    continue;

                m_dependency_degree[i] += graph->inc_edges(i).size();
                m_dependency_degree[i] -= m_degree[i];

                /* handle non-root nodes */
                if (m_parent_id[i] != i)
                    m_dependency_degree[i] -= 1;
            }
        });

    /* scan to determine offsets */
    PlusScan<luint_t, luint_t> d_scan(&m_dependency_degree[0],
        &m_dependency_offset[0]);
    tbb::parallel_scan(node_range, d_scan);

    /* save dependencies in list */
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            /**
             * for each node, neither the parent nor the child is
             * a dependency
             */
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                /* skip inactive nodes */
                if(m_parent_id[i] == invalid_luint_t)
                    continue;

                luint_t d_offset = 0;
                const luint_t i_parent = m_parent_id[i];

                const luint_t i_degree = graph->inc_edges(i).size();
                for(luint_t j = 0; j < i_degree; ++j)
                {
                    const luint_t j_edge = graph->inc_edges(i)[j];
                    const COSTTYPE j_weight = graph->edges()[j_edge].weight;
                    const luint_t j_node =
                        (graph->edges()[j_edge].node_a == i) ?
                        graph->edges()[j_edge].node_b :
                        graph->edges()[j_edge].node_a;
                    const luint_t j_parent = m_parent_id[j_node];

                    /**
                     * a node j is a dependency if it is not i's parent and
                     * j is not i's parent
                     */
                    if(j_parent != i && i_parent != j_node)
                    {
                        m_dependency_list[m_dependency_offset[i] + d_offset] =
                            j_node;
                        m_dependency_edge_id_list[m_dependency_offset[i] +
                            d_offset] = j_edge;
                        m_dependency_weight[m_dependency_offset[i] + d_offset] =
                            j_weight;

                        ++d_offset;
                    }
                }
            }
        });

    m_enable_modify = false;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_parent_ids()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_parent_id;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_to_parent_edge_ids()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_parent_edge_id;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_degrees()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_degree;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<COSTTYPE>&
Tree<COSTTYPE>::
raw_weights()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_weight;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_children_offsets()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_children_offset;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_children_list()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_children_list;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_dependency_offsets()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_dependency_offset;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_dependency_list()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_dependency_list;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_dependency_edge_id_list()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_dependency_edge_id_list;
}

/* ************************************************************************** */

template<typename COSTTYPE>
std::vector<luint_t>&
Tree<COSTTYPE>::
raw_dependency_degrees()
{
    if(!m_enable_modify)
        throw std::invalid_argument("Tree modification disabled.");

    return m_dependency_degree;
}

NS_MAPMAP_END
