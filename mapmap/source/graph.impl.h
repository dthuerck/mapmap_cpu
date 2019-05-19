/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <queue>
#include <utility>
#include <map>
#include <set>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_unordered_set.h"
#include "tbb/atomic.h"
#include "tbb/task.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "header/graph.h"

#include "dset.h"

NS_MAPMAP_BEGIN

/**
 * ****************************************************************************
 * ************ TBB task for lazy BFS-based component discovery ***************
 * ****************************************************************************
 */

/**
 * A complet contains a partial component: a list of node, its temporary
 * component ID and the ID of seen other complets
 */
using complet = std::tuple<std::vector<luint_t>, luint_t, std::set<luint_t>>;

template<typename COSTTYPE>
class LazyBFSTask : public tbb::task
{
public:
    LazyBFSTask(
        const luint_t start_node,
        Graph<COSTTYPE> * graph,
        std::vector<tbb::atomic<uint_t>> * visited,
        std::vector<luint_t> * components,
        tbb::atomic<luint_t> * nodes_left,
        tbb::concurrent_vector<complet> * complet_out)
    : tbb::task(),
      m_graph(graph),
      m_start_node(start_node),
      my_queue(),
      m_visited(visited),
      m_components(components),
      m_nodes_left(nodes_left),
      m_complet_out(complet_out)
    {
        my_queue.push(start_node);
    }

    ~LazyBFSTask()
    {
    }

    tbb::task* execute()
    {
        std::vector<luint_t> my_path;
        std::set<luint_t> my_neighbors;

        /**
         * do a standard BFS, stop once all neighbours have been visited
         * by any task in the pool
         */
        while(!my_queue.empty())
        {
            const luint_t cur_node = my_queue.front();
            my_queue.pop();

            if((*m_visited)[cur_node].compare_and_swap(
                (luint_t) 1, (luint_t) 0) == (luint_t) 0)
            {
                /**
                 * first thread to visit that node, hence iterate over
                 * neighbours
                 */
                --(*m_nodes_left);
                my_path.push_back(cur_node);
                (*m_components)[cur_node] = m_start_node;

                /**
                 * Exploit m_visited as spinlock - 1 means currently
                 * visited, 2 means finished processing.
                 */
                (*m_visited)[cur_node] = (luint_t) 2;

                for(const luint_t e_id : m_graph->inc_edges(cur_node))
                {
                    const GraphEdge<scalar_t<COSTTYPE>>& e =
                        m_graph->edges()[e_id];

                    const luint_t other_node = (e.node_a == cur_node) ?
                        e.node_b : e.node_a;
                    my_queue.push(other_node);
                }
            }
            else
            {
                while((*m_visited)[cur_node] < (luint_t) 2);

                const luint_t other_component = (*m_components)[cur_node];

                my_neighbors.insert(other_component);
            }
        }

        /**
         * Only save complet if it contains a node (would cause empty components
         * otherwise).
         */
        if (my_path.size() > 0)
            m_complet_out->push_back(complet(my_path, m_start_node,
                my_neighbors));

        return NULL;
    }

protected:
    Graph<COSTTYPE> * m_graph;
    luint_t m_start_node;
    std::queue<luint_t> my_queue;
    std::vector<tbb::atomic<uint_t>> * m_visited;
    std::vector<luint_t> * m_components;
    tbb::atomic<luint_t> * m_nodes_left;
    tbb::concurrent_vector<complet> * m_complet_out;
};

/**
 * *****************************************************************************
 * ************************** Graph - public functions *************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
FORCEINLINE
Graph<COSTTYPE>::
Graph(
    const luint_t num_nodes)
: m_num_nodes(num_nodes),
  m_nodes(num_nodes),
  m_edges(),
  m_components(num_nodes, 0),
  m_num_components(1),
  m_coloring(),
  m_was_colored(false)
{
    /* initial coloring: all 0 */
    m_coloring.resize(num_nodes);
    std::fill(m_coloring.begin(), m_coloring.end(), 0);
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
Graph<COSTTYPE>::
~Graph()
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
add_edge(
    const luint_t node_a,
    const luint_t node_b,
    const COSTTYPE weight)
throw()
{
    if(std::max(node_a, node_b) >= m_num_nodes)
        throw std::runtime_error("Graph::add_edge: "
                "At least one ID larger than number of nodes in the graph.");

    m_edges.emplace_back(GraphEdge<COSTTYPE>{node_a, node_b, weight});

    const lint_t new_edge_id = m_edges.size() - 1;

    m_nodes[node_a].incident_edges.push_back(new_edge_id);
    m_nodes[node_b].incident_edges.push_back(new_edge_id);
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<GraphNode>&
Graph<COSTTYPE>::
nodes()
const
{
    return m_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_nodes()
const
{
    return m_num_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
inc_edges(
    const luint_t node)
const
{
    return m_nodes[node].incident_edges;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<GraphEdge<COSTTYPE>>&
Graph<COSTTYPE>::
edges()
const
{
    return m_edges;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_edges()
const
{
    return m_edges.size();
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
update_components()
{

    m_components.resize(m_nodes.size());

    if(m_nodes.size() < UINT32_MAX) 
    {
        /* use disjoint-set datastructure if node ids fit into 32bit */

        DisjointSets comp_set(m_nodes.size());
        tbb::concurrent_unordered_set<uint32_t> comp_id_set(0);

        tbb::blocked_range<uint32_t> edge_range(0, m_edges.size(), 1024);
        tbb::blocked_range<uint32_t> node_range(0, m_nodes.size(), 1024);

        /* Create assigment of nodes to the representative node
         * of the connected component */
        tbb::parallel_for(edge_range, 
            [&] (const tbb::blocked_range<uint32_t>& range)
            {
                for (auto i = range.begin(); i != range.end(); ++i)
                {
                    auto edge = m_edges[i];
                    comp_set.unite(edge.node_a, edge.node_b);
                }
            });

        /* Collect all ids represantating the components */
        tbb::parallel_for(node_range, 
            [&] (const tbb::blocked_range<uint32_t>& range)
            {
                for (auto i = range.begin(); i != range.end(); ++i)
                {
                    comp_id_set.insert(comp_set.find(i));
                }
            });

        /* Sorting the representative ids into an array to map them to a contiguous
         * index via binary search */
        std::vector<uint32_t> comp_id_list(comp_id_set.begin(), comp_id_set.end());

        std::sort(comp_id_list.begin(), comp_id_list.end());

        /* Store the contiguous components ids for each node into m_components */
        tbb::parallel_for(node_range, 
            [&] (const tbb::blocked_range<uint32_t>& range)
            {
                for (auto i = range.begin(); i != range.end(); ++i)
                {
                    auto old_component_id = comp_set.find(i);
                    uint32_t new_component_id = std::lower_bound(comp_id_list.begin(),
                            comp_id_list.end(), old_component_id) - comp_id_list.begin();
                    /* If this fails, something went very wrong in the previous
                       parallel code.

                    assert(comp_id_list[new_component_id] == old_component_id); */
                    m_components[i] = (luint_t) new_component_id;
                }
            });

        m_num_components = comp_id_list.size();
    }
    else 
    {
        /* Use BFS as usual if we can't use the disjoint-set datastructure */

        m_num_components = 0;

        /* Start by assigning each node its ID as component */
        std::iota(m_components.begin(), m_components.end(), 0);

        /*
         * BFS until all nodes are marked by lazy algorithm: each thread discovers
         * part of the graph, defers building of the components until later
         */

        tbb::concurrent_vector<complet> accrued_complets;
        std::vector<tbb::atomic<uint_t>> visited(m_nodes.size());
        for (uint_t i = 0; i < m_nodes.size(); ++i)
            visited[i] = 0;
        tbb::atomic<luint_t> nodes_left;
        nodes_left = (luint_t) m_nodes.size();

        /* find random oder for start nodes */
        std::vector<luint_t> nodes(m_nodes.size());
        std::iota(nodes.begin(), nodes.end(), 0);
        std::random_shuffle(nodes.begin(), nodes.end());

        tbb::blocked_range<luint_t> node_range(0, m_nodes.size(), 32);
        while(nodes_left > 0)
        {
            /* select start nodes for this round of BFS */
            std::vector<luint_t> start_nodes(BFS_ROOTS);
            tbb::atomic<luint_t> start_nodes_selected;
            start_nodes_selected = (luint_t) 0;

            tbb::parallel_for(node_range,
                [&] (const tbb::blocked_range<luint_t>& range)
                {
                    if(start_nodes_selected >= BFS_ROOTS)
                        return;

                    for(luint_t i = range.begin(); i < range.end(); ++i)
                    {
                        const luint_t node_id = nodes[i];
                        if(visited[node_id] == (uint_t) 0)
                        {
                            const luint_t pos = start_nodes_selected++;
                            if(pos < BFS_ROOTS)
                                start_nodes[pos] = node_id;
                            else
                                return;
                        }

                    }
                });

            /* create tasks */
            const luint_t s_nodes = start_nodes_selected;
            const luint_t real_tasks = (std::min)((luint_t) BFS_ROOTS, s_nodes);
            tbb::task_list round_tasks;
            for(uint_t i = 0; i < real_tasks; ++i)
            {
                round_tasks.push_back(*new(tbb::task::allocate_root())
                    LazyBFSTask<COSTTYPE>(
                        start_nodes[i],
                        this,
                        &visited,
                        &m_components,
                        &nodes_left,
                        &accrued_complets));
            }

            /* spawn tasks and wait for completion */
            tbb::task::spawn_root_and_wait(round_tasks);
        }

        /* find unique ID mapping for components */
        std::map<luint_t, luint_t> comp_map;
        for (const complet& clet : accrued_complets)
        {
            const luint_t clet_id = std::get<1>(clet);
            const std::set<luint_t>& clet_join = std::get<2>(clet);

            if(clet_join.empty())
                comp_map[clet_id] = clet_id;
            else
                comp_map[clet_id] = invalid_luint_t;

            for(const luint_t& c : clet_join)
                comp_map[clet_id] = std::min(c, comp_map[clet_id]);
        }

        /* join complets */
        bool changed = true;
        while(changed)
        {
            changed = false;

            for (const complet& clet : accrued_complets)
            {
                const luint_t clet_id = std::get<1>(clet);
                const std::set<luint_t>& clet_join = std::get<2>(clet);

                for(const luint_t& c : clet_join)
                {
                    const luint_t cur_component = comp_map[clet_id];
                    const luint_t tr_component = comp_map[c];

                    if(tr_component < cur_component)
                    {
                        changed = true;
                        comp_map[clet_id] = tr_component;
                    }
                }
            }
        }

        /* assign contiguous IDs to components */
        luint_t id_counter = 0;
        std::map<luint_t, luint_t> new_ids;
        for(auto const& tuple : comp_map)
        {
            const luint_t val = tuple.second;
            if (new_ids.count(val) == 0)
            {
                new_ids[val] = id_counter++;
            }
        }

        /* mark all nodes with the new component ID */
        for (const complet& clet : accrued_complets)
        {
            const std::vector<luint_t> node_lst = std::get<0>(clet);
            const luint_t cmp_id = new_ids[comp_map[std::get<1>(clet)]];

            for (const luint_t& n : node_lst)
                m_components[n] = cmp_id;
        }

        m_num_components = id_counter;
    }
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
components()
const
{
    return m_components;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_components()
const
{
    return m_num_components;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
set_coloring(
    const std::vector<luint_t>& coloring)
{
    std::copy(coloring.begin(),
        coloring.begin() + std::min((luint_t) coloring.size(), m_num_nodes),
        m_coloring.begin());
    m_was_colored = true;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
get_coloring()
{
    return m_coloring;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
bool
Graph<COSTTYPE>::
was_colored()
{
    return m_was_colored;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
sort_incidence_lists()
{
    /* sort incidence list after connected nodes */
    struct inc_sorter {
        inc_sorter(luint_t node, const GraphEdge<COSTTYPE> * edges) :
            m_node(node), m_edges(edges) {};
        ~inc_sorter() {};

        bool operator()(const luint_t e_id1, const luint_t e_id2)
        {
            const GraphEdge<COSTTYPE>& e1 = m_edges[e_id1];
            const luint_t o_node1 = (e1.node_a == m_node) ?
                e1.node_b : e1.node_a;

            const GraphEdge<COSTTYPE>& e2 = m_edges[e_id2];
            const luint_t o_node2 = (e2.node_a == m_node) ?
                e2.node_b : e2.node_a;

            return (o_node1 < o_node2);
        };

        const luint_t m_node;
        const GraphEdge<COSTTYPE> * m_edges;
    };

    tbb::blocked_range<luint_t> node_range(0, m_num_nodes);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                std::sort(m_nodes[n].incident_edges.begin(),
                    m_nodes[n].incident_edges.end(),
                    inc_sorter(n, m_edges.data()));
            }
        });
}

NS_MAPMAP_END
