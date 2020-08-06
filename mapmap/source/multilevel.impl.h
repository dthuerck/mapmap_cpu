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

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
Multilevel<COSTTYPE, SIMDWIDTH>::
Multilevel(
    Graph<COSTTYPE> * original_graph,
    const LabelSet<COSTTYPE, SIMDWIDTH> * original_label_set,
    const CostBundle<COSTTYPE, SIMDWIDTH> * original_cost_bundle,
    MultilevelCriterion<COSTTYPE, SIMDWIDTH> * criterion)
: Multilevel<COSTTYPE, SIMDWIDTH>(original_graph, original_label_set,
    original_cost_bundle, criterion, false)
{
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
Multilevel<COSTTYPE, SIMDWIDTH>::
Multilevel(
    Graph<COSTTYPE> * original_graph,
    const LabelSet<COSTTYPE, SIMDWIDTH> * original_label_set,
    const CostBundle<COSTTYPE, SIMDWIDTH> * original_cost_bundle,
    MultilevelCriterion<COSTTYPE, SIMDWIDTH> * criterion,
    const bool deterministic)
: m_deterministic(deterministic),
  m_criterion(criterion),
  m_level(0)
{
    /* save original problem data as level 0 */
    m_levels.emplace_back();
    m_levels.back().level_graph = original_graph;
    m_levels.back().level_label_set = original_label_set;
    m_levels.back().level_cost_bundle = (CostBundle<COSTTYPE, SIMDWIDTH> *)
        original_cost_bundle;
    m_levels.back().prev_node_in_group = std::vector<luint_t>(original_graph->
        nodes().size());

    tbb::blocked_range<luint_t> node_range(0, original_graph->num_nodes());
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                m_levels.back().prev_node_in_group[i] = i;
        });

    m_previous = NULL;
    m_current = &m_levels[0];

    /* create allocator */
    m_value_allocator = std::unique_ptr<tbb::tbb_allocator<_s_t<COSTTYPE,
        SIMDWIDTH>>>(
        (tbb::tbb_allocator<_s_t<COSTTYPE, SIMDWIDTH>> *)
        new tbb::cache_aligned_allocator<_s_t<COSTTYPE, SIMDWIDTH>>);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
Multilevel<COSTTYPE, SIMDWIDTH>::
~Multilevel()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const CostBundle<COSTTYPE, SIMDWIDTH> *
Multilevel<COSTTYPE, SIMDWIDTH>::
get_level_cost_bundle()
const
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return m_current->level_cost_bundle;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
Graph<COSTTYPE> *
Multilevel<COSTTYPE, SIMDWIDTH>::
get_level_graph()
const
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return m_current->level_graph;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const LabelSet<COSTTYPE, SIMDWIDTH> *
Multilevel<COSTTYPE, SIMDWIDTH>::
get_level_label_set()
const
{
    if(m_level == 0)
        throw std::logic_error("No coarser level graph computed yet.");

    return m_current->level_label_set;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
Multilevel<COSTTYPE, SIMDWIDTH>::
prev_level()
{
    if(m_level == 0)
        return false;

    --m_level;
    m_levels.pop_back();

    m_current = &m_levels[m_level];
    m_previous = (m_level ? &m_levels[m_level - 1] : nullptr);

    return true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
Multilevel<COSTTYPE, SIMDWIDTH>::
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

    /* compute level costs */
    compute_level_cost_bundle();
    compute_level_unaries();
    compute_level_pairwise();

    /* copy label indices for supernodes, projected from old solution */
    projected_solution.resize(m_num_supernodes);
    std::copy(m_labels.begin(), m_labels.end(), projected_solution.begin());

    return (m_levels.back().level_graph->num_nodes() > 1);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
reproject_solution(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& level_solution,
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& original_solution)
{
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> upper_solution;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> current_solution;

    /* copy upper level assignment */
    upper_solution.assign(level_solution.begin(), level_solution.end());

    for(luint_t lvl = m_level; lvl > 0; --lvl)
    {
        /* determine number of nodes in lower level */
        const luint_t num_nodes_lower =
            m_levels[lvl - 1].level_graph->num_nodes();

        /* resize solution vector */
        current_solution.resize(num_nodes_lower);

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
                        label_from_offset(upper_node, upper_solution
                        [upper_node]);

                    /* find label index on lower node */
                    const _iv_st<COSTTYPE, SIMDWIDTH> lower_offset =
                        m_levels[lvl - 1].level_label_set->offset_for_label(o_n,
                        level_label);

                    /* save label index in original solution */
                    current_solution[o_n] = lower_offset;
                }
           });

        /* swap vectors */
        std::swap(upper_solution, current_solution);
    }

    /* export solution on original graph */
    std::swap(original_solution, upper_solution);
}

/**
 * *****************************************************************************
 * ******************************** PROTECTED **********************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_contiguous_ids(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& projected_solution)
{
    const luint_t prev_num_nodes = m_previous->level_graph->num_nodes();
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

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_level_graph_from_node_groups()
{
    /* create data structure for coarse graph */
    m_storage_graph.push_back(std::unique_ptr<Graph<COSTTYPE>>(
        new Graph<COSTTYPE>(m_num_supernodes)));

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

                /* iterate over all previously contained nodes to find edges */
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
                        if(o_s_n <= s_n)
                            continue;

                        sedge_sizes[o_s_n] += 1;
                        sedge_weights[o_s_n] += e.weight;
                        ++total;
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
                bool c_costs = true;
                const PairwiseCosts<COSTTYPE, SIMDWIDTH> * first_costs =
                    nullptr;
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

                        if(o_s_n <= s_n)
                            continue;

                        /* check if cluster has a common cost function */
                        if(first_costs == nullptr)
                        {
                            first_costs = m_previous->level_cost_bundle->
                                get_pairwise_costs(e_id);
                            c_costs = !first_costs->supports_enumerable_costs();
                        }
                        else
                        {
                            c_costs &= m_previous->level_cost_bundle->
                                get_pairwise_costs(e_id)->eq(first_costs);
                        }

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
                    m_superedge_list.reserve(m_superedge_list.size() +
                        total);

                    for(const auto& e : sedge_ids)
                    {
                        /**
                         * to preserve validity of pairwise costs in the
                         * optimization, use different weights:
                         * - different types: costs are summed up, hence adding
                         *                    up weights too would return wrong
                         *                    results, just set weight to 1
                         * - common type: sum up weights and copy costs
                         */

                        /* add edge to graph */
                        m_storage_graph.back()->add_edge(s_n, e.first, c_costs ?
                            sedge_weights[e.first] : 1.0);

                        /* record number of covered original edges */
                        m_superedge_sizes.push_back(sedge_sizes[e.first]);

                        /* append edges to list of original edges */
                        for(luint_t i = 0; i < sedge_sizes[e.first]; ++i)
                        {
                            m_superedge_list.push_back(
                                loc_list[loc_offsets[sedge_ids[e.first]] + i]);
                        }
                    }
                }
            }
        });

    /* compute global offsets for edge table */
    tbb::blocked_range<luint_t> superedge_range(0,
        m_storage_graph.back()->edges().size());
    m_superedge_offsets.resize(m_storage_graph.back()->edges().size());
    PlusScan<luint_t, luint_t> p_scan(&m_superedge_sizes[0],
        &m_superedge_offsets[0]);
    tbb::parallel_scan(superedge_range, p_scan);

    /* update components */
    m_storage_graph.back()->update_components();

    /* update pointer */
    m_current->level_graph = m_storage_graph.back().get();

    /* for deterministic execution, sort graph's incidence lists */
    if(this->m_deterministic)
        m_current->level_graph->sort_incidence_lists();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_level_label_set()
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
                        iset_labels[l] = (iset_labels[l] & inc_o_n[l]);
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

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_level_cost_bundle()
{
    m_storage_cbundle.emplace_back(std::unique_ptr<CostBundle<COSTTYPE,
        SIMDWIDTH>>(new CostBundle<COSTTYPE, SIMDWIDTH>(
        m_current->level_graph)));
    m_current->level_cost_bundle = m_storage_cbundle.back().get();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_level_unaries()
{
    /* compute unary costs for level graph (as table) */
    tbb::blocked_range<luint_t> supernode_range(0, m_num_supernodes);

    /* allocate one table per supernode */
    const luint_t store_offset = m_storage_unaries.size();
    m_storage_unaries.reserve(m_storage_unaries.size() + m_num_supernodes);

    for(luint_t i = 0; i < m_num_supernodes; ++i)
    {
        m_storage_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
            SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(
            i, m_current->level_label_set)));
        m_current->level_cost_bundle->set_unary_costs(i,
            m_storage_unaries.back().get());
    }

    tbb::parallel_for(supernode_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t s_n = r.begin(); s_n < r.end(); ++s_n)
            {
                UnaryTable<COSTTYPE, SIMDWIDTH> * un_tab =
                    m_storage_unaries[store_offset + s_n].get();

                const _iv_st<COSTTYPE, SIMDWIDTH> max_label =
                    m_current->level_label_set->max_label();
                const _iv_st<COSTTYPE, SIMDWIDTH> lset_size =
                    m_current->level_label_set->label_set_size(s_n);

                /* add up costs for labels 0 to max_label */
                const luint_t costs_size = DIV_UP(max_label + 1, SIMDWIDTH) *
                    SIMDWIDTH * sizeof(_s_t<COSTTYPE, SIMDWIDTH>);
                _s_t<COSTTYPE, SIMDWIDTH> * costs = m_value_allocator->
                    allocate(costs_size);
                std::fill(costs, costs + DIV_UP(max_label + 1, SIMDWIDTH) *
                    SIMDWIDTH, 0);

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
                    _iv_t<COSTTYPE, SIMDWIDTH> iv_l;
                    _v_t<COSTTYPE, SIMDWIDTH> v_c;
                    for(_iv_st<COSTTYPE, SIMDWIDTH> o = 0; o < lset_chunk;
                        o += SIMDWIDTH)
                    {
                        iv_l = m_previous->level_label_set->labels_from_offset(
                            o_n, o);

                        if(m_previous->level_cost_bundle->get_unary_costs(o_n)->
                            supports_enumerable_costs())
                        {
                            v_c = m_previous->level_cost_bundle->
                                get_unary_costs(o_n)->
                                get_unary_costs_enum_offset(o);
                        }
                        else
                        {
                            v_c = m_previous->level_cost_bundle->
                                get_unary_costs(o_n)->
                                get_unary_costs(iv_l);
                        }

                        iv_store<COSTTYPE, SIMDWIDTH>(iv_l, i_tmp);
                        v_store<COSTTYPE, SIMDWIDTH>(v_c, v_tmp);

                        for(uint_t j = 0; j < SIMDWIDTH; ++j)
                            if((_iv_st<COSTTYPE, SIMDWIDTH>) (o + j) <
                                lset_size)
                                costs[i_tmp[j]] += v_tmp[j];
                    }
                }

                /* compact (unary) costs according to common labels */
                _s_t<COSTTYPE, SIMDWIDTH> * sp_costs = un_tab->get_raw_costs();
                for(_iv_st<COSTTYPE, SIMDWIDTH> i = 0; i < lset_size; ++i)
                {
                    const _iv_st<COSTTYPE, SIMDWIDTH> l =
                        m_current->level_label_set->label_from_offset(s_n, i);
                    sp_costs[i] = costs[l];
                }

                /**
                 * apart from the (straightforward) addition of unary costs,
                 * one has to include pairwise costs between nodes of the same
                 * label, thus count edges and sum up these costs for all
                 * feasible labels multiplied by their weight
                 */

                /* must view each edge separately */
                const _iv_st<COSTTYPE, SIMDWIDTH> supernode_labels =
                m_current->level_label_set->label_set_size(s_n);
                _iv_st<COSTTYPE, SIMDWIDTH> l_i;
                _iv_t<COSTTYPE, SIMDWIDTH> l;
                _v_t<COSTTYPE, SIMDWIDTH> c;

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
                            const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l =
                                m_current->level_label_set->
                                    label_from_offset(s_n, 0);

                            /**
                             * add all costs for the edge - labels
                             * restricted to that label set of the supernode
                             * in question which was already computed
                             */
                            _iv_t<COSTTYPE, SIMDWIDTH> l_ix, valid_labels;
                            for(l_i = 0; l_i < supernode_labels; l_i
                                += SIMDWIDTH)
                            {
                                l = m_current->level_label_set->
                                    labels_from_offset(s_n, l_i);
                                l_ix = iv_sequence<COSTTYPE, SIMDWIDTH>(l_i);

                                /* mask out invalid labels */
                                valid_labels = iv_le<COSTTYPE, SIMDWIDTH>
                                    (l_ix, iv_init<COSTTYPE, SIMDWIDTH>(
                                    supernode_labels - 1));
                                l = iv_blend<COSTTYPE, SIMDWIDTH>(
                                    iv_init<COSTTYPE, SIMDWIDTH>(exceed_l),
                                    l, valid_labels);

                                /* compute costs for valid labels */
                                c = v_mult<COSTTYPE, SIMDWIDTH>(
                                    m_previous->level_cost_bundle->
                                    get_pairwise_costs(e_id)->
                                    get_pairwise_costs(l, l),
                                    v_init<COSTTYPE, SIMDWIDTH>(edge.weight));

                                /* store data in indexed table */
                                v_store<COSTTYPE, SIMDWIDTH>(
                                    v_add<COSTTYPE, SIMDWIDTH>(c,
                                    v_load<COSTTYPE, SIMDWIDTH>(
                                    &sp_costs[l_i])), &sp_costs[l_i]);
                            }
                        }
                    }
                }

                /* clean up */
                m_value_allocator->deallocate(costs, costs_size);
            }
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
Multilevel<COSTTYPE, SIMDWIDTH>::
compute_level_pairwise()
{
    tbb::blocked_range<luint_t> superedge_range(0, m_current->level_graph->
        edges().size());

    /* preallocate cost functions for individual edges */
    const luint_t store_offset = m_storage_pairwise.size();
    m_storage_pairwise.resize(m_storage_pairwise.size() +
        m_current->level_graph->edges().size());

    /**
     * Having prerecorded the original edges making up superedges,
     * summing up the costs is straightforward now.
     */
    tbb::parallel_for(superedge_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t se_id = r.begin(); se_id != r.end(); ++se_id)
            {
                /* sum up costs for all original edges for se_id */
                const luint_t num_edges = m_superedge_sizes[se_id];
                const luint_t off_edges = m_superedge_offsets[se_id];

                /* check whether all covered edges have the same costs */
                const PairwiseCosts<COSTTYPE, SIMDWIDTH> * first_cost =
                    m_previous->level_cost_bundle->get_pairwise_costs(
                        m_superedge_list[off_edges]);
                bool common_cost = !first_cost->supports_enumerable_costs();
                for(luint_t i = 1; i < num_edges && common_cost; ++i)
                {
                    const luint_t o_edge_id = m_superedge_list[
                        off_edges + i];

                    common_cost &= (first_cost->eq(m_previous->
                        level_cost_bundle->get_pairwise_costs(o_edge_id)));
                }

                /* if so: clone costs - weights already set */
                if(common_cost)
                {
                    /* clone costs from original graph and use ptr */
                    m_storage_pairwise[store_offset + se_id] =
                        first_cost->copy();
                    m_current->level_cost_bundle->set_pairwise_costs(se_id,
                        m_storage_pairwise[store_offset + se_id].get());

                    continue;
                }

                /* otherwise: retrieve superedge */
                GraphEdge<COSTTYPE> s_e = m_current->level_graph->
                    edges()[se_id];
                m_storage_pairwise[store_offset + se_id] = std::unique_ptr<
                    PairwiseCosts<COSTTYPE, SIMDWIDTH>>(new PairwiseTable<
                    COSTTYPE, SIMDWIDTH>(s_e.node_a, s_e.node_b,
                    m_current->level_label_set));

                /* create superedge cost table */
                _s_t<COSTTYPE, SIMDWIDTH> * costs =
                    dynamic_cast<PairwiseTable<COSTTYPE, SIMDWIDTH>*>(
                    m_storage_pairwise[
                    store_offset + se_id].get())->get_raw_costs();

                /* retrieve level label indices */
                const _iv_st<COSTTYPE, SIMDWIDTH> pad_size =
                    DIV_UP(m_current->level_label_set->max_label_set_size(),
                    SIMDWIDTH) * SIMDWIDTH;
                _s_t<COSTTYPE, SIMDWIDTH> * buf = m_value_allocator->allocate(
                    2 * pad_size);
                _iv_st<COSTTYPE, SIMDWIDTH> * lbl_assoc =
                    (_iv_st<COSTTYPE, SIMDWIDTH> *) buf;
                _iv_st<COSTTYPE, SIMDWIDTH> * lbl_assoc_mask =
                    (_iv_st<COSTTYPE, SIMDWIDTH> *) (buf + pad_size);

                /* information regarding the superedge */
                const _iv_st<COSTTYPE, SIMDWIDTH> s_num_l1 =
                    m_current->level_label_set->label_set_size(
                    s_e.node_a);
                const _iv_st<COSTTYPE, SIMDWIDTH> s_num_l2 =
                    m_current->level_label_set->label_set_size(
                    s_e.node_b);
                const _iv_st<COSTTYPE, SIMDWIDTH> pad_l2 =
                    DIV_UP(s_num_l2, SIMDWIDTH) * SIMDWIDTH;

                for(luint_t i = 0; i < num_edges; ++i)
                {
                    const luint_t o_edge_id = m_superedge_list[
                        off_edges + i];
                    const GraphEdge<COSTTYPE> e =
                        m_previous->level_graph->edges()[o_edge_id];

                    /* retrieve old edge's costs */
                    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * c_pairwise =
                        m_previous->level_cost_bundle->get_pairwise_costs(
                        o_edge_id);

                    /**
                     * determine if supernode ordering and node ordering
                     * matches
                     */
                    const bool is_in_order =
                        (m_current->prev_node_in_group[e.node_a] ==
                        s_e.node_a) &&
                        (m_current->prev_node_in_group[e.node_b] ==
                        s_e.node_b);
                    const luint_t o_node_a = is_in_order ? e.node_a : e.node_b;
                    const luint_t o_node_b = is_in_order ? e.node_b : e.node_a;

                    const _iv_st<COSTTYPE, SIMDWIDTH> o_num_la =
                        m_previous->level_label_set->label_set_size(o_node_a);
                    const _iv_st<COSTTYPE, SIMDWIDTH> o_num_lb =
                        m_previous->level_label_set->label_set_size(o_node_b);

                    if(c_pairwise->supports_enumerable_costs())
                    {
                        /* enumerable costs: find label correspondence once */
                        _iv_st<COSTTYPE, SIMDWIDTH> o_e_ptr = 0;

                        /* match labels from original node and supernode */
                        _iv_st<COSTTYPE, SIMDWIDTH> l1_i, l2_i;
                        _iv_t<COSTTYPE, SIMDWIDTH> l1, l2, l2_ix, mask,
                            valid_labels;
                        _v_t<COSTTYPE, SIMDWIDTH> c;

                        /* for each supernode B label, fix corr. ix in node b */
                        for(l2_i = 0; l2_i < s_num_l2; ++l2_i)
                        {
                            const _iv_st<COSTTYPE, SIMDWIDTH> l_s =
                                m_current->level_label_set->label_from_offset(
                                s_e.node_b, l2_i);

                            _iv_st<COSTTYPE, SIMDWIDTH> l_o =
                                m_previous->level_label_set->label_from_offset(
                                o_node_b, o_e_ptr);
                            while(l_o < l_s && o_e_ptr < o_num_lb)
                                l_o = m_previous->level_label_set->
                                    label_from_offset(o_node_b, ++o_e_ptr);

                            lbl_assoc[l2_i] = (l_o == l_s) ?
                                o_e_ptr : 0;
                            lbl_assoc_mask[l2_i] = (l_o == l_s) ?
                                ~0x0 : 0x0;
                        }

                        /* overflow to next SIMD multiple */
                        for(l2_i = s_num_l2; l2_i < pad_l2; ++l2_i)
                        {
                            lbl_assoc[l2_i] = 0;
                            lbl_assoc_mask[l2_i] = 0x0;
                        }

                        /* now use indexed pairwise costs for l2 */
                        o_e_ptr = 0;

                        for(l1_i = 0; l1_i < s_num_l1; ++l1_i)
                        {
                            const _iv_st<COSTTYPE, SIMDWIDTH> ls =
                                m_current->level_label_set->label_from_offset(
                                s_e.node_a, l1_i);

                            /* find corresponding label index in o_node_a */
                            while(m_previous->level_label_set->
                                label_from_offset(o_node_a, o_e_ptr) < ls &&
                                o_num_la)
                                ++o_e_ptr;

                            /* ignore label if not in superedge */
                            if(m_previous->level_label_set->
                                label_from_offset(o_node_a, o_e_ptr) != ls)
                                continue;

                            l1 = iv_init<COSTTYPE, SIMDWIDTH>(o_e_ptr);
                            for(l2_i = 0; l2_i < s_num_l2; l2_i +=
                                SIMDWIDTH)
                            {
                                l2 = iv_load<COSTTYPE, SIMDWIDTH>(
                                    &lbl_assoc[l2_i]);
                                l2_ix = iv_sequence<COSTTYPE, SIMDWIDTH>(l2_i);

                                /* mask out exceeding label indices */
                                valid_labels = iv_le<COSTTYPE, SIMDWIDTH>(l2_ix,
                                    iv_init<COSTTYPE, SIMDWIDTH>(s_num_l2 - 1));
                                l2 = iv_blend<COSTTYPE, SIMDWIDTH>(
                                    iv_init<COSTTYPE, SIMDWIDTH>(0), l2,
                                    valid_labels);

                                /* retrieve original costs */
                                c = v_mult<COSTTYPE, SIMDWIDTH>(
                                    c_pairwise->get_pairwise_costs_enum_offsets(
                                        is_in_order ? l1 : l2,
                                        is_in_order ? l2 : l1),
                                    v_init<COSTTYPE, SIMDWIDTH>(e.weight));

                                /* mask out invalid labels */
                                mask = iv_load<COSTTYPE, SIMDWIDTH>(
                                    &lbl_assoc_mask[l2_i]);
                                c = v_and<COSTTYPE, SIMDWIDTH>(
                                    c, iv_reinterpret_v<COSTTYPE,
                                    SIMDWIDTH>(mask));

                                /* masked store for matching labels */
                                c = v_add<COSTTYPE, SIMDWIDTH>(c,
                                    v_load<COSTTYPE, SIMDWIDTH>(
                                    &costs[l1_i * pad_l2 + l2_i]));
                                v_store<COSTTYPE, SIMDWIDTH>(
                                    c, &costs[l1_i * pad_l2 + l2_i]);
                            }
                        }
                    }
                    else
                    {
                        /* non - enumerable: use actual labels for query */
                        _iv_st<COSTTYPE, SIMDWIDTH> l1_i, l2_i;
                        _iv_t<COSTTYPE, SIMDWIDTH> l1, l2, l2_ix, valid_labels;
                        _v_t<COSTTYPE, SIMDWIDTH> c, c_o;

                        /* label for masking out invalid indices */
                        const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l =
                            m_current->level_label_set->label_from_offset(
                            s_e.node_b, 0);

                        /* iterate over supernode labels directly */
                        for(l1_i = 0; l1_i < s_num_l1; ++l1_i)
                        {
                            l1 = iv_init<COSTTYPE, SIMDWIDTH>(
                                m_current->level_label_set->label_from_offset(
                                s_e.node_a, l1_i));

                            for(l2_i = 0; l2_i < s_num_l2; l2_i += SIMDWIDTH)
                            {
                                l2 = m_current->level_label_set->
                                    labels_from_offset(s_e.node_b, l2_i);
                                l2_ix = iv_sequence<COSTTYPE, SIMDWIDTH>(
                                    l2_i);

                                /* mask out exceeding labels */
                                valid_labels = iv_le<COSTTYPE, SIMDWIDTH>
                                    (l2_ix, iv_init<COSTTYPE, SIMDWIDTH>(
                                    s_num_l2 - 1));
                                l2 = iv_blend<COSTTYPE, SIMDWIDTH>(
                                    iv_init<COSTTYPE, SIMDWIDTH>(exceed_l), l2,
                                    valid_labels);

                                /* retrieve costs for original edge */
                                c = v_mult<COSTTYPE, SIMDWIDTH>(
                                    c_pairwise->get_pairwise_costs(
                                        is_in_order ? l1 : l2,
                                        is_in_order ? l2 : l1),
                                    v_init<COSTTYPE, SIMDWIDTH>(e.weight));

                                /* add to costs already in vector */
                                c_o = v_load<COSTTYPE, SIMDWIDTH>(
                                    &costs[l1_i * pad_l2 + l2_i]);
                                c = v_add<COSTTYPE, SIMDWIDTH>(c, c_o);

                                /* store back to vector */
                                v_store<COSTTYPE, SIMDWIDTH>(c,
                                    &costs[l1_i * pad_l2 + l2_i]);
                            }
                        }
                    }
                }

                /* deallocate level assoc */
                m_value_allocator->deallocate(buf,
                    2 * m_current->level_label_set->max_label_set_size());

                /* use pointer to costs */
                m_current->level_cost_bundle->set_pairwise_costs(se_id,
                    m_storage_pairwise[store_offset + se_id].get());
            }
        });
}

NS_MAPMAP_END

