/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/multilevel_instances/group_same_label.h"

#include <algorithm>
#include <queue>
#include <iostream>

#include "tbb/atomic.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_do.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
GroupSameLabel<COSTTYPE, SIMDWIDTH>::
GroupSameLabel()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
GroupSameLabel<COSTTYPE, SIMDWIDTH>::
~GroupSameLabel()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
GroupSameLabel<COSTTYPE, SIMDWIDTH>::
group_nodes(
    std::vector<luint_t>& node_in_group,
    const LevelSet<COSTTYPE, SIMDWIDTH> * current_level,
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& current_solution,
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& projected_solution)
{
    const Graph<COSTTYPE> * graph = current_level->level_graph;
    const luint_t num_nodes = graph->num_nodes();

    tbb::blocked_range<luint_t> node_range(0, num_nodes);

    /* use atomic locks for synchronizing access to nodes */
    std::vector<tbb::atomic<char>> node_locks(num_nodes, 0u);
    node_in_group.resize(num_nodes);
    std::fill(node_in_group.begin(), node_in_group.end(), invalid_luint_t);

    /* create seed "queue" */
    std::vector<luint_t> qu(num_nodes);
    std::iota(std::begin(qu), std::end(qu), 0);
    std::random_shuffle(qu.begin(), qu.end());


    /* do BFSes from randomly selected seeds */
    tbb::parallel_do(qu.begin(), qu.end(),
        [&](const luint_t& seed)
        {
            /* retrieve label common to this group */
            const _iv_st<COSTTYPE, SIMDWIDTH> seed_lbl =
                current_level->level_label_set->label_from_offset(seed,
                current_solution[seed]);

            std::queue<luint_t> bfs;
            bfs.push(seed);

            bool skip_neighbors;
            while(!bfs.empty())
            {
                skip_neighbors = false;

                const luint_t cur = bfs.front();
                bfs.pop();

                /* lock node */
                while(node_locks[cur].compare_and_swap(1u, 0u) != 0u);

                /**
                 * smaller marker: another thread is already here or this
                 * thread already saw this node, so stop
                 */
                if(node_in_group[cur] <= seed)
                    skip_neighbors = true;
                else
                    node_in_group[cur] = seed;

                /* free node */
                node_locks[cur] = 0u;

                /* traverse neighbors if they have the same label */
                if(!skip_neighbors)
                {
                    for(const luint_t& e_id :
                        current_level->level_graph->inc_edges(cur))
                    {
                        const GraphEdge<COSTTYPE>& e =
                            current_level->level_graph->edges()[e_id];
                        const luint_t neighbor = (e.node_a == cur) ?
                            e.node_b : e.node_a;

                        /* retrieve other node's label */
                        const _iv_st<COSTTYPE, SIMDWIDTH> other_label =
                            current_level->level_label_set->label_from_offset(
                            neighbor, current_solution[neighbor]);

                        if(other_label == seed_lbl)
                        {
                            bfs.push(neighbor);
                        }
                    }
                }
            }
        });

    /* just copy old solution, all locations are correct, then */
    projected_solution.assign(current_solution.begin(), current_solution.end());
}

NS_MAPMAP_END