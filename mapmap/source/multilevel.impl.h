/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/multilevel.h"

#include <stdexcept>
#include <utility>
#include <map>
#include <set>
#include <iostream>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/concurrent_vector.h"
#include "tbb/atomic.h"

#include "header/parallel_templates.h"
#include "header/costs.h"
#include "header/cost_instances/unary_table.h"
#include "header/cost_instances/pairwise_table.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * ********************************** PUBLIC ***********************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
Multilevel(
    const Graph<COSTTYPE> * original_graph,
    const LabelSet<COSTTYPE, SIMDWIDTH> * original_label_set,
    const UNARY * original_unaries,
    const PAIRWISE * original_pairwise,
    MultilevelCriterion<COSTTYPE, SIMDWIDTH> * criterion)
: m_criterion(criterion),
  m_level(0)
{
    /* save original problem data as level 0 */
    m_levels.emplace_back();
    m_levels.back().level_graph = original_graph;
    m_levels.back().level_label_set = original_label_set;
    m_levels.back().level_unaries = original_unaries;
    m_levels.back().level_pairwise = original_pairwise;
    m_levels.back().prev_node_in_group = std::vector<luint_t>(original_graph->
        nodes().size());

    tbb::blocked_range<luint_t> node_range(0, original_graph->nodes().size());
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                m_levels.back().prev_node_in_group[i] = i;
        });

    m_previous = NULL;
    m_current = &m_levels[0];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~Multilevel()
{

} 

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const UnaryTable<COSTTYPE, SIMDWIDTH> *
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
get_level_unaries()
const
throw()
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return (const UnaryTable<COSTTYPE, SIMDWIDTH>*) m_current->level_unaries;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const PairwiseTable<COSTTYPE, SIMDWIDTH> *
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
get_level_pairwise()
const
throw()
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return (const PairwiseTable<COSTTYPE, SIMDWIDTH>*) 
        m_current->level_pairwise;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const Graph<COSTTYPE> *
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
get_level_graph()
const
throw()
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return m_current->level_graph;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const LabelSet<COSTTYPE, SIMDWIDTH> *
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
get_level_label_set()
const
throw()
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return m_current->level_label_set;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
bool
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
prev_level()
{
    if(m_level == 0)
        return false;

    --m_level;
    m_levels.pop_back();

    m_current = &m_levels[m_level];
    m_previous = &m_levels[m_level - 1];

    return true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
bool
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
next_level(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& prev_solution,
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& projected_solution)
{
    m_solution = prev_solution;

    /* clean temporary data structures */
    m_num_supernodes = 0;

    m_supernode_sizes.clear();
    m_supernode_offsets.clear();
    m_supernode_list.clear();

    m_superedge_sizes.clear();
    m_superedge_offsets.clear();
    m_superedge_list.clear();

    /* create data for next level */
    m_levels.emplace_back();
    ++m_level;

    /* can't copy m_current to m_previous, since vector may be reallocated */
    m_previous = &m_levels[m_level - 1];
    m_current = &m_levels[m_level];

    /* group nodes to create supernodes (and project old solution) */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> old_solution;
    m_criterion->group_nodes(m_levels.back().prev_node_in_group,
        &m_levels[m_level - 1], prev_solution, old_solution);

    /* assign contiguous IDs to supernodes */
    compute_contiguous_ids(old_solution);

    /* construct graph and record superedge lists */
    compute_level_graph_from_node_groups();

    /* reconstruct label sets for upper level */
    compute_level_label_set();

    /* sum up unary and pairwise costs */
    compute_level_unaries();
    compute_level_pairwise();

    /* copy label indices for supernodes, projected from old solution */
    projected_solution.resize(m_num_supernodes);
    std::copy(m_labels.begin(), m_labels.end(), projected_solution.begin());

    return (m_levels.back().level_graph->nodes().size() > 1);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
reproject_solution(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& level_solution,
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& original_solution)
{
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> sol_a;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> sol_b;

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> * upper_solution = &sol_a;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> * current_solution = &sol_b;

    /* copy upper level assignment */
    upper_solution->assign(level_solution.begin(), level_solution.end());

    for(luint_t lvl = m_level; lvl > 0; --lvl)
    {
        /* determine number of nodes in lower level */
        const luint_t num_nodes_lower = 
            m_levels[lvl - 1].level_graph->nodes().size();

        /* resize solution vector */ 
        current_solution->resize(num_nodes_lower);

        /* reproject solution to previous level */
        tbb::blocked_range<luint_t> lower_level_range(0, num_nodes_lower);

        tbb::parallel_for(lower_level_range,
           [&](const tbb::blocked_range<luint_t>& r)
           {
                for(luint_t o_n = r.begin(); o_n != r.end(); ++o_n)
                {
                    const luint_t upper_node = 
                        m_levels[lvl].prev_node_in_group[o_n];

                    /* note: solutions are indices, so fetch real label first */
                    const luint_t level_label = m_levels[lvl].level_label_set->
                        label_from_offset(upper_node, (*upper_solution)
                        [upper_node]);

                    /* find label index on lower node */
                    const _iv_st<COSTTYPE, SIMDWIDTH> lower_offset = 
                        m_levels[lvl - 1].level_label_set->offset_for_label(o_n,
                        level_label);

                    /* save label index in original solution */
                    (*current_solution)[o_n] = lower_offset;
                }
           });

        /* swap vectors */
        std::swap(upper_solution, current_solution);
    }

    /* export solution on original graph */
    original_solution.clear();
    original_solution.assign(upper_solution->begin(), upper_solution->end());
}

/**
 * *****************************************************************************
 * ******************************** PROTECTED **********************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_contiguous_ids(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& projected_solution)
{
    const luint_t prev_num_nodes = m_previous->level_graph->nodes().size();
    tbb::blocked_range<luint_t> node_range(0, prev_num_nodes);

    /* assign ID to representative node's ID per group - in order */
    std::vector<luint_t> representatives(prev_num_nodes, 0);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                const luint_t new_id = m_current->prev_node_in_group[n];

                if(new_id == n)
                    representatives[n] = 1;
            }
        });

    /* perform scan to determine contiguous, order-respecting ID */
    std::vector<luint_t> new_ids(prev_num_nodes);
    PlusScan<luint_t, luint_t> id_scan(&representatives[0], &new_ids[0]);
    tbb::parallel_scan(node_range, id_scan);

    /* next ID is also number of supernodes */
    m_num_supernodes = new_ids.back() + representatives.back();
    m_labels.resize(m_num_supernodes);
    tbb::blocked_range<luint_t> supernode_range(0, m_num_supernodes);

    /* assign new IDs to all nodes and count supernode sizes */
    std::vector<tbb::atomic<luint_t>> supernode_sizes(m_num_supernodes, 0);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                /* with ID change, save label for supernode */
                if(representatives[n] > 0u)
                {
                    const _iv_st<COSTTYPE, SIMDWIDTH> lbl = m_previous->
                        level_label_set->label_from_offset(n,
                        projected_solution[n]);
                    m_labels[new_ids[m_current->prev_node_in_group[n]]] = lbl;
                }
                m_current->prev_node_in_group[n] = 
                    new_ids[m_current->prev_node_in_group[n]];
                supernode_sizes[m_current->prev_node_in_group[n]]++;
            }
        });

    /* compute offsets for supernode node lists */
    m_supernode_offsets.resize(m_num_supernodes, 0);
    m_supernode_sizes.resize(m_num_supernodes, 0);
    tbb::parallel_for(supernode_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                m_supernode_sizes[n] = supernode_sizes[n];
                supernode_sizes[n] = 0;
                m_supernode_offsets[n] = 0;
            }
        });
    PlusScan<luint_t, luint_t> p_scan(&m_supernode_sizes[0], 
        &m_supernode_offsets[0]);
    tbb::parallel_scan(supernode_range, p_scan);

    /* collect supernodes' original nodes in common list */
    m_supernode_list.resize(prev_num_nodes);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                const luint_t supernode = m_current->prev_node_in_group[n];
                const luint_t loc_offset = supernode_sizes[supernode].
                    fetch_and_increment();
                m_supernode_list[m_supernode_offsets[supernode] + loc_offset] = 
                    n;
            } 
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_level_label_set()
throw()
{
    /* collect label staticstics for original graph */
    const _iv_st<COSTTYPE, SIMDWIDTH> max_label = 
        m_previous->level_label_set->max_label();

    /* create label set structure for coarse graph (no compression) */
    m_storage_label_set.push_back(std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>>
        (new LabelSet<COSTTYPE, SIMDWIDTH>(m_num_supernodes, false)));
    
    /* determine a set of feasible labels per supernode */
    tbb::blocked_range<luint_t> supernode_range(0, m_num_supernodes);
    luint_t num_empty_label_sets = 0;
    tbb::parallel_for(supernode_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t s_n = r.begin(); s_n != r.end(); ++s_n)
            {
                /* collect intersection of incidence vectors */
                std::vector<bool> iset_labels(max_label + 1, true);
                std::vector<bool> inc_o_n(max_label + 1, false);
                std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lset;

                const luint_t s_num_nodes = m_supernode_sizes[s_n];
                for(luint_t i = 0; i < s_num_nodes; ++i)
                {
                    std::fill(inc_o_n.begin(), inc_o_n.end(), false);

                    const luint_t o_n = m_supernode_list[m_supernode_offsets[
                        s_n] + i];

                    /* compute incidence vector for o_n */
                    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < 
                        m_previous->level_label_set->label_set_size(o_n); ++l_i)
                    {
                        const _iv_st<COSTTYPE, SIMDWIDTH> l = 
                            m_previous->level_label_set->label_from_offset(o_n,
                            l_i);
                        inc_o_n[l] = true;
                    }

                    /* intersect with common vector */
                    for(_iv_st<COSTTYPE, SIMDWIDTH> l = 0; l <= max_label; ++l)
                        iset_labels[l] = (iset_labels[l] && inc_o_n[l]);
                }

                /* create label set from incidence vector */
                for(_iv_st<COSTTYPE, SIMDWIDTH> l = 0; l <= max_label; ++l)
                    if(iset_labels[l])
                        lset.push_back(l);

                /* replace projected label for supernode by its index */
                const _iv_st<COSTTYPE, SIMDWIDTH> maxl = lset.size();
                for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < maxl; ++l_i)
                {
                    if(m_labels[s_n] == lset[l_i])
                    {
                        m_labels[s_n] = l_i;
                        break;
                    }
                }

                if(lset.empty())
                    num_empty_label_sets++;

                /* create label set from incidence vector - serial access */
                {
                    tbb::mutex::scoped_lock lock(m_graph_write_mutex);
                    
                    m_storage_label_set.back()->set_label_set_for_node(
                        s_n, lset);
                }
            }
        });

    if(num_empty_label_sets > 0)
        throw std::domain_error("Empty label set for at least one supernode!");

    /* update pointers */
    m_current->level_label_set = m_storage_label_set.back().get();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_level_unaries()
{
    /* compute unary costs for level graph (as table) */
    tbb::blocked_range<luint_t> supernode_range(0, m_num_supernodes);

    /* create data structure for unary costs */
    m_storage_unaries.push_back(std::unique_ptr<UnaryTable<COSTTYPE,
        SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(
        m_current->level_graph, m_current->level_label_set)));
    UnaryTable<COSTTYPE, SIMDWIDTH> * un_tab = m_storage_unaries.back().get(); 

    tbb::parallel_for(supernode_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t s_n = r.begin(); s_n < r.end(); ++s_n)
            {
                const _iv_st<COSTTYPE, SIMDWIDTH> max_label = 
                    m_current->level_label_set->max_label();
                const _iv_st<COSTTYPE, SIMDWIDTH> lset_size = 
                    m_current->level_label_set->label_set_size(s_n);

                /* add up costs for labels 0 to max_label */
                std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(max_label + 1, 
                    (_s_t<COSTTYPE, SIMDWIDTH>) 0);

                _iv_st<COSTTYPE, SIMDWIDTH> i_tmp[SIMDWIDTH];
                _s_t<COSTTYPE, SIMDWIDTH> v_tmp[SIMDWIDTH];

                /* add up costs for all child nodes */
                for(luint_t i = 0; i < m_supernode_sizes[s_n]; ++i)
                {
                    const luint_t o_n = m_supernode_list[m_supernode_offsets
                        [s_n] + i];
                    const _iv_st<COSTTYPE, SIMDWIDTH> lset_size = 
                        m_previous->level_label_set->label_set_size(o_n);
                    const _iv_st<COSTTYPE, SIMDWIDTH> lset_chunk = 
                        SIMDWIDTH * DIV_UP(lset_size, SIMDWIDTH);

                    /* iterate over labels and costs */
                    for(_iv_st<COSTTYPE, SIMDWIDTH> o = 0; o < lset_chunk; 
                        o += SIMDWIDTH)
                    {
                        _iv_t<COSTTYPE, SIMDWIDTH> iv_l = 
                            m_previous->level_label_set->labels_from_offset(
                            o_n, o);
                        _v_t<COSTTYPE, SIMDWIDTH> v_c;

                        if(m_previous->level_unaries->
                            supports_enumerable_costs())
                        { 
                            v_c = m_previous->level_unaries->
                                get_unary_costs_enum_offset(o_n, o);
                        }
                        else
                        {
                            v_c = m_previous->level_unaries->get_unary_costs(
                                o_n, iv_l);
                        }

                        iv_store<COSTTYPE, SIMDWIDTH>(iv_l, i_tmp);
                        v_store<COSTTYPE, SIMDWIDTH>(v_c, v_tmp);

                        for(uint_t j = 0; j < SIMDWIDTH; ++j)
                            if((_iv_st<COSTTYPE, SIMDWIDTH>) (o + j) < lset_size)
                                costs[i_tmp[j]] += v_tmp[j];
                    }
                }

                /**
                 * apart from the (straightforward) addition of unary costs,
                 * one has to include pairwise costs between nodes of the same
                 * label, thus count edges and sum up these costs for all 
                 * feasible labels multiplied by their weight
                 */
                if(m_previous->level_pairwise->node_dependent())
                {
                    /* must view each edge separately */
                    const _iv_st<COSTTYPE, SIMDWIDTH> supernode_labels = 
                        m_current->level_label_set->label_set_size(s_n);
                    _iv_st<COSTTYPE, SIMDWIDTH> l_i, l_t;
                    _iv_t<COSTTYPE, SIMDWIDTH> l;
                    _v_t<COSTTYPE, SIMDWIDTH> c;
                    _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];

                    /* iterate over all original edges */
                    const luint_t node_size = m_supernode_sizes[s_n];
                    const luint_t node_offset = m_supernode_offsets[s_n];
                    for(luint_t i = 0; i < node_size; ++i)
                    {
                        const luint_t o_n = m_supernode_list[node_offset + i];

                        for(const luint_t& e_id : 
                            m_previous->level_graph->inc_edges(o_n))
                        {
                            const GraphEdge<COSTTYPE>& edge = 
                                m_previous->level_graph->edges()[e_id];
                            const luint_t other_node = (edge.node_a == o_n) ? 
                                edge.node_b : edge.node_a;
                            const luint_t ep_sn = 
                                m_current->prev_node_in_group[o_n];
                            const luint_t ep_so = 
                                m_current->prev_node_in_group[other_node];

                            if(ep_sn == ep_so && o_n < other_node)
                            {
                                /** 
                                 * add all costs for the edge - labels 
                                 * restricted to that label set of the supernode
                                 * in question which was already computed
                                 */
                                for(l_i = 0; l_i < supernode_labels; ++l_i)
                                {
                                    l_t = m_current->level_label_set->
                                        label_from_offset(s_n, l_i); 
                                    l = iv_init<COSTTYPE, SIMDWIDTH>(l_t);
                                    c = m_previous->level_pairwise->
                                        get_binary_costs(o_n, l, other_node, l);
                                    v_store<COSTTYPE, SIMDWIDTH>(c, tmp);
                                    costs[l_t] += edge.weight * tmp[0];
                                }
                            }
                        }
                    }
                }
                else
                {
                    /* can use common costs and multiply */
                    _s_t<COSTTYPE, SIMDWIDTH> sum_weight = 0;

                    /* add up weight of all intra-supernode edges */
                    const luint_t node_size = m_supernode_sizes[s_n];
                    const luint_t node_offset = m_supernode_offsets[s_n];
                    for(luint_t i = 0; i < node_size; ++i)
                    {
                        const luint_t o_n = m_supernode_list[node_offset + i];

                        for(const luint_t& e_id : 
                            m_previous->level_graph->inc_edges(o_n))
                        {
                            const GraphEdge<COSTTYPE>& edge = 
                                m_previous->level_graph->edges()[e_id];
                            const luint_t other_node = (edge.node_a == o_n) ? 
                                edge.node_b : edge.node_a;
                            const luint_t ep_sn = 
                                m_current->prev_node_in_group[o_n];
                            const luint_t ep_so = 
                                m_current->prev_node_in_group[other_node];

                            if(ep_sn == ep_so && o_n < other_node)
                                sum_weight += edge.weight;
                        }
                    }

                    /* now add up costs */
                    _iv_st<COSTTYPE, SIMDWIDTH> l_s;
                    _iv_t<COSTTYPE, SIMDWIDTH> l;
                    _v_t<COSTTYPE, SIMDWIDTH> c;
                    _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
                    for(l_s = 0; l_s <= max_label; ++l_s)
                    {
                        l = iv_init<COSTTYPE, SIMDWIDTH>(l_s);
                        c = m_previous->level_pairwise->get_binary_costs(l, l);
                        v_store<COSTTYPE, SIMDWIDTH>(c, tmp);
                        costs[l_s] += sum_weight * tmp[0];
                    } 
                }

                /* compact costs according to common labels */
                std::vector<_s_t<COSTTYPE, SIMDWIDTH>> sp_costs(lset_size, 0);
                for(_iv_st<COSTTYPE, SIMDWIDTH> i = 0; i < lset_size; ++i)
                {
                    const _iv_st<COSTTYPE, SIMDWIDTH> l = 
                        m_current->level_label_set->label_from_offset(s_n, i);
                    sp_costs[i] = costs[l];
                }

                /* save costs for node - serial access due to writing */
                {
                    tbb::mutex::scoped_lock lock(m_graph_write_mutex);
                    un_tab->set_costs_for_node(s_n, sp_costs);
                }
            }
        });

    m_current->level_unaries = m_storage_unaries.back().get();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_level_pairwise()
{
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs;

    if(m_previous->level_pairwise->node_dependent())
    {
        tbb::blocked_range<luint_t> superedge_range(0, m_current->level_graph->
            edges().size());

        /* determine the size of the packed cost list */
        std::vector<luint_t> superedge_cost_sizes(m_current->level_graph->
            edges().size(), 0);
        std::vector<luint_t> superedge_cost_offsets(m_current->level_graph->
            edges().size(), 0);
        tbb::parallel_for(superedge_range,
            [&](const tbb::blocked_range<luint_t>& r)
            {
                for(luint_t s_e = r.begin(); s_e != r.end(); ++s_e)
                {
                    const GraphEdge<COSTTYPE> e = 
                        m_current->level_graph->edges()[s_e];

                    /* determine number of feasible label pairs */
                    const luint_t num_label_pairs = 
                        m_current->level_label_set->label_set_size(e.node_a) *
                        m_current->level_label_set->label_set_size(e.node_b);

                    superedge_cost_sizes[s_e] = num_label_pairs;
                }
            });

        /* determine offsets into label table */
        PlusScan<luint_t, luint_t> p_scan(&superedge_cost_sizes[0],
            &superedge_cost_offsets[0]);
        tbb::parallel_scan(superedge_range, p_scan);

        /* allocate packed label table */
        const luint_t cost_size = superedge_cost_offsets.back() + 
            superedge_cost_sizes.back();
        costs.resize(cost_size);
        std::fill(costs.begin(), costs.end(), 0);
        
        /** 
         * Having prerecorded the original edges making up superedges,
         * summing up the costs is straightforward now.
         */
        tbb::parallel_for(superedge_range,
            [&](const tbb::blocked_range<luint_t>& r)
            {
                _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];

                for(luint_t se_id = r.begin(); se_id != r.end(); ++se_id)
                {
                    /* sum up costs for all original edges for se_id */
                    const luint_t num_edges = m_superedge_sizes[se_id];
                    const luint_t off_edges = m_superedge_offsets[se_id];

                    /* retrieve superedge */
                    GraphEdge<COSTTYPE> s_e = m_current->level_graph->
                        edges()[se_id];

                    for(luint_t i = 0; i < num_edges; ++i)
                    {
                        const luint_t o_edge_id = m_superedge_list[
                            off_edges + i];
                        const GraphEdge<COSTTYPE> e = 
                            m_previous->level_graph->edges()[o_edge_id];

                        const _iv_st<COSTTYPE, SIMDWIDTH> num_l1 = 
                            m_current->level_label_set->label_set_size(
                            s_e.node_a);
                        const _iv_st<COSTTYPE, SIMDWIDTH> num_l2 = 
                            m_current->level_label_set->label_set_size(
                            s_e.node_b);

                        /** 
                         * determine if supernode ordering and node ordering
                         * matches 
                         */
                        const bool is_in_order = 
                            (m_current->prev_node_in_group[e.node_a] == 
                            s_e.node_a) && 
                            (m_current->prev_node_in_group[e.node_b] ==
                            s_e.node_b);  

                        _iv_st<COSTTYPE, SIMDWIDTH> l1_i, l2_i;
                        _iv_t<COSTTYPE, SIMDWIDTH> l1, l2;
                        _v_t<COSTTYPE, SIMDWIDTH> c, c_o;
                        for(l1_i = 0; l1_i < num_l1; ++l1_i)
                        {
                            l1 = iv_init<COSTTYPE, SIMDWIDTH>(
                                m_current->level_label_set->label_from_offset(
                                s_e.node_a, l1_i));

                            for(l2_i = 0; l2_i < num_l2; ++l2_i)
                            {
                                l2 = iv_init<COSTTYPE, SIMDWIDTH>(
                                    m_current->level_label_set->
                                    label_from_offset(s_e.node_b, l2_i));

                                /* retrieve costs for original edge */
                                c = m_previous->level_pairwise->
                                    get_binary_costs(
                                        e.node_a, 
                                        is_in_order ? l1 : l2,
                                        e.node_b, 
                                        is_in_order ? l2 : l1);

                                /* add to costs already in vector */
                                c_o = v_load<COSTTYPE, SIMDWIDTH>(
                                    &costs[superedge_cost_offsets[se_id] + 
                                    l1_i * num_l2 + l2_i]);
                                c = v_add<COSTTYPE, SIMDWIDTH>(c, c_o);

                                /* store back to vector */
                                v_store<COSTTYPE, SIMDWIDTH>(c, tmp);
                                costs[superedge_cost_offsets[se_id] + 
                                    l1_i * num_l2 + l2_i] = tmp[0];
                            }
                        }
                    }
                }
            });

        /* construct node-dependent pairwise table */
        m_storage_pairwise.push_back(
            std::unique_ptr<PairwiseTable<COSTTYPE, SIMDWIDTH>>(new 
            PairwiseTable<COSTTYPE, SIMDWIDTH>(m_current->level_label_set,
            m_current->level_graph, costs)));
        m_current->level_pairwise = m_storage_pairwise.back().get();
    }
    else
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> max_label = 
            m_previous->level_label_set->max_label();
        const _iv_st<COSTTYPE, SIMDWIDTH> num_labels = max_label + 1;
        const _iv_st<COSTTYPE, SIMDWIDTH> num_labels_chunk = 
            SIMDWIDTH * DIV_UP(num_labels, SIMDWIDTH);
        costs.resize(num_labels * num_labels_chunk, 0);

        /* location-independent pairwise costs - copy old dense table */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> labels(num_labels_chunk);
        std::iota(labels.begin(), labels.end(), 0);

        _iv_st<COSTTYPE, SIMDWIDTH> i, j;
        _iv_t<COSTTYPE, SIMDWIDTH> l1, l2;
        _v_t<COSTTYPE, SIMDWIDTH> c;
        for(i = 0; i < num_labels_chunk; ++i)
        {
            l1 = iv_init<COSTTYPE, SIMDWIDTH>(labels[i]);
            for(j = 0; j < num_labels_chunk; j += SIMDWIDTH)
            {
                l2 = iv_load<COSTTYPE, SIMDWIDTH>(&labels[j]);
                c = m_previous->level_pairwise->get_binary_costs(l1, l2);

                v_store<COSTTYPE, SIMDWIDTH>(c, &costs[i * num_labels + j]);
            }   
        }

        /* construct node-independent pairwise table */
        m_storage_pairwise.push_back(
            std::unique_ptr<PairwiseTable<COSTTYPE, SIMDWIDTH>>(new 
            PairwiseTable<COSTTYPE, SIMDWIDTH>(num_labels, costs)));
        m_current->level_pairwise = m_storage_pairwise.back().get();
    }

    
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_level_graph_from_node_groups()
{
    /* create data structure for coarse graph */
    m_storage_graph.push_back(std::unique_ptr<Graph<COSTTYPE>>(
        new Graph<COSTTYPE>));

    /**
     * find superedges and add them to the current graph - similarly, record
     * all original edges (for node-dependent pairwise costs) associated with
     * these superedges
     */

    tbb::blocked_range<luint_t> supernode_range(0, m_num_supernodes);
    tbb::parallel_for(supernode_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t s_n = r.begin(); s_n != r.end(); ++s_n)
            {
                std::map<luint_t, luint_t> sedge_sizes;
                std::map<luint_t, COSTTYPE> sedge_weights;
                std::map<luint_t, luint_t> sedge_ids;
                luint_t total = 0;

                /* iterate over all contained nodes to find edges */
                const luint_t num_o_nodes = m_supernode_sizes[s_n];
                for(luint_t i = 0; i < num_o_nodes; ++i)
                {
                    const luint_t o_n = m_supernode_list[
                        m_supernode_offsets[s_n] + i];

                    for(const luint_t& e_id : 
                        m_previous->level_graph->inc_edges(o_n))
                    {
                        const GraphEdge<COSTTYPE>& e = m_previous->level_graph->
                            edges()[e_id];
                        const luint_t other_n = (e.node_a == o_n) ? 
                            e.node_b : e.node_a;
                        const luint_t o_s_n = 
                            m_current->prev_node_in_group[other_n];

                        /* avoid loops and only add superedge once */
                        if(o_s_n < s_n)
                            continue;

                        sedge_sizes[o_s_n] += 1;
                        sedge_weights[o_s_n] += e.weight;
                        total += 1;
                    }
                }

                /* compute local offsets for superedges */
                const luint_t num_superedges = sedge_sizes.size();
                std::vector<luint_t> loc_offsets(num_superedges + 1, 0);

                luint_t counter = 0;
                for(const auto& se : sedge_sizes)
                {
                    sedge_ids[se.first] = counter;
                    loc_offsets[counter + 1] = loc_offsets[counter] + 
                        se.second;
                    ++counter;
                }

                /* reset sizes */
                for(const auto& se : sedge_ids)
                    sedge_sizes[se.first] = 0;

                /* collect edge IDs associated with superedges */
                std::vector<luint_t> loc_list(total);
                for(luint_t i = 0; i < num_o_nodes; ++i)
                {
                    const luint_t o_n = m_supernode_list[
                        m_supernode_offsets[s_n] + i];

                    for(const luint_t& e_id : 
                        m_previous->level_graph->inc_edges(o_n))
                    {
                        const GraphEdge<COSTTYPE>& e = m_previous->level_graph->
                            edges()[e_id];
                        const luint_t other_n = (e.node_a == o_n) ? 
                            e.node_b : e.node_a;
                        const luint_t o_s_n = 
                            m_current->prev_node_in_group[other_n];

                        if(o_s_n < s_n)
                            continue;
                        
                        const luint_t off = sedge_sizes[o_s_n];

                        /* save original edge ID */
                        loc_list[loc_offsets[sedge_ids[o_s_n]] + off] = e_id;

                        ++sedge_sizes[o_s_n];
                    }
                }

                /**
                 * Having computed an indexed list for this supernode,
                 * add these superedges and the list to the current graph 
                 * (serially).
                 */
                {
                    tbb::mutex::scoped_lock lock(m_graph_write_mutex);

                    /* enlarge storage for edge table */
                    m_superedge_sizes.reserve(m_superedge_sizes.size() + 
                        sedge_ids.size());
                    m_superedge_offsets.reserve(m_superedge_offsets.size() + 
                        sedge_ids.size());
                    m_superedge_list.reserve(m_superedge_list.size() + 
                        total);

                    for(const auto& e : sedge_ids)
                    {
                        if(s_n >= e.first)
                            continue;

                        /**
                         * to preserve validity of pairwise costs in the 
                         * optimization, use different weights:
                         * - node-independent: reuse costs, so just add weights
                         *                     of original edges covered
                         * - node-dependent: costs are summed up, hence adding
                         *                   up weights too would return wrong
                         *                   results, just set weight to 1
                         */
                        const bool is_node_dep = m_previous->level_pairwise->
                            node_dependent();

                        /* add edge to graph */
                        m_storage_graph.back()->add_edge(s_n, e.first,
                            is_node_dep ? 1 : sedge_weights[e.first]);

                        /* record number of covered original edges */
                        m_superedge_sizes.push_back(sedge_sizes[e.first]);
                        
                        /* append edges to list of original edges */
                        for(luint_t i = 0; i < sedge_sizes[e.first]; ++i)
                            m_superedge_list.push_back(
                                loc_list[loc_offsets[sedge_ids[e.first]] + i]);
                    }
                }
            }
        });

    /* compute global offsets for edge table */
    tbb::blocked_range<luint_t> superedge_range(0, 
        m_storage_graph.back()->edges().size());
    m_superedge_offsets.clear();
    m_superedge_offsets.resize(m_storage_graph.back()->edges().size());
    PlusScan<luint_t, luint_t> p_scan(&m_superedge_sizes[0],
        &m_superedge_offsets[0]);
    tbb::parallel_scan(superedge_range, p_scan);

    /* update components */
    m_storage_graph.back()->update_components();

    /* update pointer */
    m_current->level_graph = m_storage_graph.back().get();
}

NS_MAPMAP_END

