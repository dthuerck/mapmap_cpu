/*
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_UTIL_TEST_H_
#define __MAPMAP_UTIL_TEST_H_

#include <functional>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/graph.h"
#include "header/tree.h"

NS_MAPMAP_BEGIN

/*
 * Create a grid-graph with a number of connected components (i.e.
 * create that number of component_dim x component_dim 4-connected
 * grids in one graph).
 * Save directly in m_graph.
 */
template<typename COSTTYPE>
std::unique_ptr<Graph<COSTTYPE>>
createComponentGrid(
    const uint_t num_components,
    const uint_t component_dim);

template<typename COSTTYPE>
void
BFSWithCustomFunc(
    Tree<COSTTYPE> * tree,
    const luint_t& root_id,
    std::function<void(const Tree<COSTTYPE>*, const luint_t)> 
        per_node_func);

/* comparison of vectors */
template<typename T>
bool
cmp_vector(
    const std::vector<T>& a,
    const std::vector<T>& b,
    const size_t len = 0);

NS_MAPMAP_END

/* include templated implementation */
#include "test/util_test.impl.h"

#endif /* __MAPMAP_UTIL_TEST_H_ */