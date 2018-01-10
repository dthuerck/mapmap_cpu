/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/color.h"

#include <algorithm>

#include "tbb/atomic.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_for.h"
#include "tbb/mutex.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE>
Color<COSTTYPE>::
Color(
    Graph<COSTTYPE>& graph)
: Color<COSTTYPE>(graph, false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
Color<COSTTYPE>::
Color(
    Graph<COSTTYPE>& graph,
    const bool deterministic)
: m_graph(graph),
  m_deterministic(deterministic),
  m_conf_a(),
  m_conf_b()
{

}

/* ************************************************************************** */
template<typename COSTTYPE>
Color<COSTTYPE>::
~Color()
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
void
Color<COSTTYPE>::
color_graph(
    std::vector<luint_t>& coloring)
{
    const luint_t n = m_graph.num_nodes();

    /* pointers for in/out queue in every iteration */
    tbb::concurrent_vector<luint_t> * conf_in = &m_conf_a;
    tbb::concurrent_vector<luint_t> * conf_out = &m_conf_b;

    /* record all nodes as conflicts */
    conf_in->resize(n);
    std::iota(conf_in->begin(), conf_in->end(), 0);

    /* acc: use max_d as initial number of colors */
    tbb::atomic<luint_t> k = 1;

    /* use atomics for colors */
    std::vector<tbb::atomic<luint_t>> atom_colors(n, 0);

    std::vector<char> conf_arrays((k + 1) * conf_in->size());
    while(!conf_in->empty())
    {
        const luint_t old_k = k;

        /* clear output arrays */
        conf_out->clear();

        /* compute necessary memory for forbidden arrays */
        const luint_t mem_size = conf_in->size() * (k + 1);
        if(conf_arrays.size() < mem_size)
            conf_arrays.resize(mem_size);

        /* resolve detected conflicts */
        tbb::blocked_range<luint_t> conf_range(0, conf_in->size(), 64u);
        const auto fun_resolve = [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t ix = r.begin(); ix != r.end(); ++ix)
            {
                const luint_t conf_node = (*conf_in)[ix];

                /* clear forbidden array */
                char * forbidden = conf_arrays.data() + ix * old_k;
                std::memset(forbidden, 0, old_k);

                /* collect neighboring colors */
                const luint_t degree = m_graph.inc_edges(conf_node).size();
                for(luint_t j = 0; j < degree; ++j)
                {
                    const luint_t e_id =
                        m_graph.inc_edges(conf_node)[j];
                    const GraphEdge<COSTTYPE>& e = m_graph.edges()[e_id];
                    const luint_t o_node = (e.node_a == conf_node) ?
                        e.node_b : e.node_a;

                    forbidden[atom_colors[o_node]] = 1u;
                }

                /* find minimum non-conflicting color */
                luint_t new_color = atom_colors[conf_node];
                for(luint_t i = 0; i < old_k &&
                    forbidden[new_color]; ++i)
                    new_color = i;

                /* still not conflict-free? need to add one color */
                if(forbidden[new_color])
                {
                    k.compare_and_swap(old_k + 1, old_k);
                    new_color = old_k + 1;
                }

                /* set color atomically */
                atom_colors[conf_node] = new_color;
            }
        };

        /* for deterministic coloring, run function serially */
        if(m_deterministic)
            fun_resolve(conf_range);
        else
            tbb::parallel_for(conf_range, fun_resolve);

        /* detect remaining conflicts */
        const auto fun_detect = [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t ix = r.begin(); ix != r.end(); ++ix)
            {
                const luint_t conf_node = (*conf_in)[ix];

                /* detect conflicts */
                bool recorded_node = false;
                const luint_t degree = m_graph.inc_edges(conf_node).size();
                for(luint_t j = 0; j < degree && !recorded_node; ++j)
                {
                    const luint_t e_id =
                        m_graph.inc_edges(conf_node)[j];
                    const GraphEdge<COSTTYPE>& e = m_graph.edges()[e_id];
                    const luint_t o_node = (e.node_a == conf_node) ?
                        e.node_b : e.node_a;

                    if(atom_colors[conf_node] == atom_colors[o_node] &&
                        conf_node < o_node)
                    {
                        conf_out->push_back(conf_node);
                        recorded_node = true;
                    }
                }
            }
        };

        /* for deterministic coloring, run function serially */
        if(m_deterministic)
            fun_detect(conf_range);
        else
            tbb::parallel_for(conf_range, fun_detect);

        /* exchange input/output queues */
        std::swap(conf_in, conf_out);
    }

    /* copy coloring from atomics */
    coloring.resize(n);
    std::copy(atom_colors.begin(), atom_colors.end(), coloring.begin());
}

NS_MAPMAP_END