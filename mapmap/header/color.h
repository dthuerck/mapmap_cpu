/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_COLOR_H_
#define __MAPMAP_COLOR_H_

#include "tbb/concurrent_vector.h"
#include "header/defines.h"

#include "header/graph.h"

NS_MAPMAP_BEGIN

/**
 * Implements an optimistic graph coloring based on
 *
 * A. H. Gebremedhin and F. Manne, “Scalable parallel graph coloring
 * algorithms,” Concurrency: Practice & Experience, vol. 12, no. 12, pp.
 * 1131–1146, 2000
 */

template<typename COSTTYPE>
class Color
{
public:
    Color(Graph<COSTTYPE>& graph);
    Color(Graph<COSTTYPE>& graph, const bool deterministic);
    ~Color();

    void color_graph(std::vector<luint_t>& coloring);

protected:
    Graph<COSTTYPE>& m_graph;

    /* deterministic (serial) handling */
    const bool m_deterministic;

    tbb::concurrent_vector<luint_t> m_conf_a;
    tbb::concurrent_vector<luint_t> m_conf_b;
};

NS_MAPMAP_END

#include "source/color.impl.h"

#endif /* __MAPMAP_COLOR_H_ */