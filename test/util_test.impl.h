/*
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <queue>

#include "test/util_test.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE>
std::unique_ptr<Graph<COSTTYPE>>
createComponentGrid(
    const uint_t num_components,
    const uint_t component_dim)
{
    using cost_t = scalar_t<COSTTYPE>;

    std::unique_ptr<Graph<COSTTYPE>> graph(new Graph<COSTTYPE>);
    const cost_t w = (cost_t) 1.0;

    uint_t offset = 0;
    for (uint_t c_id = 0; c_id < num_components; ++c_id)
    {
        for (uint_t y = 0; y < component_dim; ++y)
        {
            for (uint_t x = 0; x < component_dim; ++x)
            {
                if (x < component_dim - 1)
                    graph->add_edge(
                        offset + y * component_dim + x,
                        offset + y * component_dim + x + 1,
                        w);

                if (y < component_dim - 1)
                    graph->add_edge(
                        offset + y * component_dim + x,
                        offset + (y + 1) * component_dim + x,
                        w);
            }
        }

        offset += component_dim * component_dim;
    }

    return graph;
}

template<typename COSTTYPE>
void
BFSWithCustomFunc(
    Tree<COSTTYPE> * tree,
    const luint_t& root_id,
    std::function<void(const Tree<COSTTYPE>*, const luint_t)> 
        per_node_func)
{
    std::queue<luint_t> qu;
    qu.push(root_id);

    while(!qu.empty())
    {
        const luint_t cur_node_id = qu.front();
        qu.pop();

        /* if available, execute function on each visited node */
        if (per_node_func)
            per_node_func(tree, cur_node_id);

        const TreeNode<COSTTYPE> cur_node = tree->node(cur_node_id);
        for(luint_t i = 0; i < cur_node.degree; ++i)
            qu.push(cur_node.children_ids[i]);
    }
}

template<typename T>
bool
cmp_vector(
    const std::vector<T>& a,
    const std::vector<T>& b,
    const size_t len)
{
    bool result = true;

    /* no length specified: compare full vectors, size must be the same */
    if(len == 0)
        result &= (a.size() == b.size());

    const luint_t size = (len == 0 ? a.size() : len); 

    for(luint_t i = 0; result && i < size; ++i)
        result &= (a[i] == b[i]);

    return result;
}

NS_MAPMAP_END