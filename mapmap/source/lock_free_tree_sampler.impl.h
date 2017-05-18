/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/lock_free_tree_sampler.h"

#include <random>

#include "tbb/parallel_reduce.h"

/**
 * *****************************************************************************
 * **************************** ColoredQueueBunch ******************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
ColoredQueueBunch<COSTTYPE>::
ColoredQueueBunch(
    const Graph<COSTTYPE> * graph)
: m_graph(graph),
  m_num_queues(1),
  m_data(graph->num_nodes()),
  m_qu_start(),
  m_qu_pos()
{
    init();
}

/* ************************************************************************** */

template<typename COSTTYPE>
ColoredQueueBunch<COSTTYPE>::
~ColoredQueueBunch()
{
}

/* ************************************************************************** */

template<typename COSTTYPE>
void
ColoredQueueBunch<COSTTYPE>::
push_to(
    const luint_t qu,
    const luint_t elem)
{
    if(qu < m_num_qu)
    {
        const luint_t pos = m_qu_pos[qu]++;
        m_qu_start[qu][pos] = elem;
    }
}

/* ************************************************************************** */

template<typename COSTTYPE>
luint_t
ColoredQueueBunch<COSTTYPE>::
queue_size(
    const luint_t qu)
{
    if(qu >= m_num_qu)
        return 0;

    return m_qu_pos[qu];
}

/* ************************************************************************** */

template<typename COSTTYPE>
luint_t
ColoredQueueBunch<COSTTYPE>::
queue_capacity(
    const luint_t qu)
{
    if(qu >= m_num_qu)
        return 0;

    return (m_qu_start[qu + 1] - m_qu_start[qu]);
}

/* ************************************************************************** */

template<typename COSTTYPE>
luint_t *
ColoredQueueBunch<COSTTYPE>::
queue(
    const luint_t qu)
{
    if(qu >= m_num_qu)
        return nullptr;

    return m_qu_start[qu];
}

/* ************************************************************************** */

template<typename COSTTYPE>
luint_t
ColoredQueueBunch<COSTTYPE>::
num_queues()
{
    return m_num_qu;
}

/* ************************************************************************** */

template<typename COSTTYPE>
void
ColoredQueueBunch<COSTTYPE>::
reset()
{
    /* reset pos counter per queue */
    for(luint_t i = 0; i < m_num_qu; ++i)
        m_qu_pos[i] = 0;
}

/* ************************************************************************** */

template<typename COSTTYPE>
void
ColoredQueueBunch<COSTTYPE>::
init()
{
    /* retrieve graph's coloring (assume colored input) */
    const std::vector<luint_t>& coloring = m_graph->get_coloring();

    /* compute maximum color (= num_colors - 1) */
    const luint_t k = tbb::parallel_reduce(coloring.begin(), coloring.end(),
        std::max<luint_t>());
    m_num_qu = k + 1;

    /* count capacity per queue */
    std::vector<luint_t> qu_size(m_num_qu, 0);
    for(const luint_t& c : coloring)
        ++qu_size[c];

    /* compute queue offsets */
    m_qu_start.resize(m_num_qu + 1);

    luint_t offset = 0;
    for(luint_t i = 0; i < m_qu_num; ++i)
    {
        m_qu_start[i] = m_data.data() + offset;
        offset += qu_size[i];
    }
    m_qu_start[m_qu_num] = m_data.data() + offset;

    /* initailize queue pos counters to zero */
    m_qu_pos.resize(m_num_qu);
    reset();
}

/**
 * *****************************************************************************
 * *************************** LockFreeTreeSampler *****************************
 * *****************************************************************************
 */

template<typename COSTTYPE, bool ACYCLIC>
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
LockFreeTreeSampler(
    const Graph<COSTTYPE> * graph)
: m_graph(graph),
  m_qu_a(graph),
  m_qu_b(graph)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
~LockFreeTreeSampler()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
select_random_roots(
    const luint_t k,
    std::vector<luint_t>& roots)
{
    /**
     * Can exploit information about the color distribution in the colored queue
     * bunch
     */
    std::mt19937 rnd(m_rnd_dev());

    /* select a color at random with >= k nodes */
    std::vector<luint_t> feasible_colors;

    const luint_t num_colors = m_qu_a.num_queues();
    for(luint_t c = 0; c < num_colors; ++c)
        if(m_qu_a.queue_capacity(c) >= k)
            feasible_colors.push_back(c);

    /* select a feasible color at random */
    luint_t root_c = 0;
    if(feasible_colors.size() > 0)
    {
        std::uniform_int_distribution<luint_t> cd(0,
            feasible_colors.size() - 1);
        root_c = cd(rnd);
    }

    /* gather all nodes of the chosen color */
    std::vector<luint_t> feasible_roots;
    feasible_roots.reserve(m_qu_a.queue_capacity(root_c));

    const std::vector<luint_t>& coloring = m_graph->get_coloring();
    for(luint_t i = 0; i < coloring.size(); ++i)
        if(coloring[i] == root_c)
            feasible_roots.push_back(i);

    const luint_t new_k = std::min(k, m_qu_a.queue_capacity(root_c));

    /* select nodes until happy */
    roots.resize(new_k);
    for(luint_t i = 0; i < new_k; ++i)
    {
        /* select next root */
        std::uniform_int_distribution<luint_t> rd(0, feasible_roots.size() - 1);

        const luint_t chosen_ix = rd(rnd);
        roots[i] = feasible_roots[chosen_ix];

        /* remove root from candidate list */
        feasible_roots[chosen_ix] = feasible_roots.back();
        feasible_roots.pop_back();
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
std::unique_ptr<Tree<COSTTYPE>>
LockFreeTreeSampler<COSTTYPE, ACYLIC>::
sample(
    std::vector<luint_t>& roots,
    bool record_dependencies,
    bool relax)
{
    /* TODO */
}