/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_bundle.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
CostBundle<COSTTYPE, SIMDWIDTH>::
CostBundle(
    const Graph<COSTTYPE> * graph)
: m_graph(graph),
  m_unary(graph->num_nodes(), nullptr),
  m_pairwise(graph->num_edges(), nullptr)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
CostBundle<COSTTYPE, SIMDWIDTH>::    
~CostBundle()
{
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CostBundle<COSTTYPE, SIMDWIDTH>::    
set_unary_costs(
    const UnaryCosts<COSTTYPE, SIMDWIDTH> * costs)
{
    std::fill(m_unary.begin(), m_unary.end(), costs);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CostBundle<COSTTYPE, SIMDWIDTH>::    
set_pairwise_costs(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
{
    std::fill(m_pairwise.begin(), m_pairwise.end(), costs);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CostBundle<COSTTYPE, SIMDWIDTH>::    
set_unary_costs(
    const luint_t node_id, 
    const UnaryCosts<COSTTYPE, SIMDWIDTH> * costs)
{
    if(node_id < m_graph->num_nodes())
        m_unary[node_id] = costs;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CostBundle<COSTTYPE, SIMDWIDTH>::    
set_pairwise_costs(
    const luint_t edge_id, 
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
{
    if(edge_id < m_graph->num_edges())
        m_pairwise[edge_id] = costs;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const UnaryCosts<COSTTYPE, SIMDWIDTH> *
CostBundle<COSTTYPE, SIMDWIDTH>::        
get_unary_costs(
    const luint_t node_id) 
const
{
    return (node_id < m_graph->num_nodes() ? m_unary[node_id] : nullptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const PairwiseCosts<COSTTYPE, SIMDWIDTH> *
CostBundle<COSTTYPE, SIMDWIDTH>::    
get_pairwise_costs(
    const luint_t edge_id) 
const
{
    return (edge_id < m_graph->num_edges() ? m_pairwise[edge_id] : nullptr);
}

NS_MAPMAP_END