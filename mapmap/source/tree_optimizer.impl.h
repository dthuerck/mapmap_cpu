/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
#include "header/tree_optimizer.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

#include <iostream>

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
TreeOptimizer()
: m_has_tree(false),
  m_has_label_set(false),
  m_has_unaries(false),
  m_has_pairwise(false),
  m_uses_dependencies(false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~TreeOptimizer()
{
    
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_graph(
    const Graph<COSTTYPE> * graph)
{
    m_graph = graph;
    m_has_graph = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_tree(
    const Tree<COSTTYPE> * tree)
{
    m_tree = tree;
    m_has_tree = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_label_set(
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
{
    m_label_set = label_set;
    m_has_label_set = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_costs(
    const UNARY * unary,
    const PAIRWISE * pairwise)
{
    m_unaries = unary;
    m_pairwise = pairwise;

    m_has_unaries = true;
    m_has_pairwise = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
use_dependencies(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& current_solution)
{
    m_current_assignment = current_solution;
    m_uses_dependencies = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
_s_t<COSTTYPE, SIMDWIDTH>
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
objective(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
{
    _s_t<COSTTYPE, SIMDWIDTH> objective = (COSTTYPE) 0;
    tbb::blocked_range<luint_t> node_range(0, m_graph->num_nodes());
    tbb::blocked_range<luint_t> edge_range(0, m_graph->edges().size());

    /* unary costs */
    objective += tbb::parallel_reduce(node_range, (COSTTYPE) 0,
        [&](const tbb::blocked_range<luint_t>& r, COSTTYPE reduced)
        {
            _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
            _s_t<COSTTYPE, SIMDWIDTH> my_chunk = reduced;

            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                /* determine label index for node n */
                const uint_t n_l = solution[n];

                _v_t<COSTTYPE, SIMDWIDTH> u_costs;
                if(m_unaries->supports_enumerable_costs())
                {
                    u_costs = m_unaries->get_unary_costs_enum_offset(n, n_l);
                }
                else
                {
                    /* translate to label */
                    const _iv_st<COSTTYPE, SIMDWIDTH> l = 
                        m_label_set->label_from_offset(n, n_l); 

                    /* get cost vector */
                    u_costs = m_unaries->get_unary_costs(n, iv_init<COSTTYPE,
                        SIMDWIDTH>(l));
                }

                /* extract first element of vector */
                v_store<COSTTYPE, SIMDWIDTH>(u_costs, tmp);

                my_chunk += tmp[0];
            }

            return my_chunk;
        },
        std::plus<COSTTYPE>());

    /* pairwise costs */ 
    objective += tbb::parallel_reduce(edge_range, (COSTTYPE) 0,
        [&](const tbb::blocked_range<luint_t>& r, COSTTYPE reduced)
        {
            _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
            _s_t<COSTTYPE, SIMDWIDTH> my_chunk = reduced;

            for(luint_t e = r.begin(); e != r.end(); ++e)
            {
                /* determine label (indices) for both nodes */
                const luint_t n_a = m_graph->edges()[e].node_a;
                const luint_t n_b = m_graph->edges()[e].node_b;

                const _iv_st<COSTTYPE, SIMDWIDTH> n_a_l_i = solution[n_a];
                const _iv_st<COSTTYPE, SIMDWIDTH> n_b_l_i = solution[n_b];

                /* translate to labels */
                const _iv_st<COSTTYPE, SIMDWIDTH> n_a_l = 
                    m_label_set->label_from_offset(n_a, n_a_l_i);
                const _iv_st<COSTTYPE, SIMDWIDTH> n_b_l = 
                    m_label_set->label_from_offset(n_b, n_b_l_i);

                /* get cost vector */
                _v_t<COSTTYPE, SIMDWIDTH> p_costs = 
                    v_init<COSTTYPE, SIMDWIDTH>();

                if(m_pairwise->node_dependent())
                    p_costs = m_pairwise->get_binary_costs(n_a, 
                        iv_init<COSTTYPE, SIMDWIDTH>(n_a_l),
                        n_b, iv_init<COSTTYPE, SIMDWIDTH>(n_b_l));
                else
                    p_costs = m_pairwise->get_binary_costs(
                        iv_init<COSTTYPE, SIMDWIDTH>(n_a_l),
                        iv_init<COSTTYPE, SIMDWIDTH>(n_b_l));

                /* extract first element of vector */
                v_store<COSTTYPE, SIMDWIDTH>(p_costs, tmp);

                my_chunk += tmp[0];
            }

            return my_chunk;
        },
        std::plus<COSTTYPE>());

    return objective; 
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
bool
TreeOptimizer<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
data_complete()
{
    return (m_has_graph && m_has_tree && m_has_label_set && m_has_unaries && 
        m_has_pairwise);
}

NS_MAPMAP_END
