/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <exception>
#include <functional>
#include <iostream>

#include "tbb/tbb.h"


#include "header/dynamic_programming.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * *********************** DynamicProgrammingTableEntry ************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
DynamicProgrammingTableEntry(
    const TreeNode<COSTTYPE>& node,
    _s_t<COSTTYPE, SIMDWIDTH> * tbl_opt_values,
    _iv_st<COSTTYPE, SIMDWIDTH> * tbl_opt_labels)
: m_node(node),
  m_opt_values(tbl_opt_values),
  m_opt_labels(tbl_opt_labels)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~DynamicProgrammingTableEntry()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const luint_t&
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
node_id()
{
    return m_node.node_id;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const luint_t&
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
parent_id()
{
    return m_node.parent_id;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize_entry(
    const DPBundle<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>& costs)
{
    /* determine if we are optimizing a root node */
    const bool is_root = (node_id() == parent_id());

    /* optimize a single node - if it's a root, assume an artificial parent */
    const _iv_st<COSTTYPE, SIMDWIDTH> num_parent_labels = is_root ? 1 :
        costs.c_labels->label_set_size(parent_id());
    const _iv_st<COSTTYPE, SIMDWIDTH> num_node_labels =
        costs.c_labels->label_set_size(node_id());

    /* initialize running minimum */
    _v_t<COSTTYPE, SIMDWIDTH> min_costs = v_init<COSTTYPE, SIMDWIDTH>(
        std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    _iv_t<COSTTYPE, SIMDWIDTH> min_labels = iv_init<COSTTYPE, SIMDWIDTH>(
        std::numeric_limits<_iv_st<COSTTYPE, SIMDWIDTH>>::max());

    /* space for final reduction */
    _s_t<COSTTYPE, SIMDWIDTH> r_cost[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> r_label[SIMDWIDTH];

    /* mask for excluding invalid labels */
    _iv_t<COSTTYPE, SIMDWIDTH> valid_labels;

    /* label to substitute for exceeding label indices */
    const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l = costs.c_labels->
        label_from_offset(node_id(), 0);

    /* one DP table entry per iteration of this loop is created */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_p_i = 0; l_p_i < num_parent_labels;
        ++l_p_i)
    {
        min_costs = v_init<COSTTYPE, SIMDWIDTH>(
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
        min_labels = iv_init<COSTTYPE, SIMDWIDTH>(
            std::numeric_limits<_iv_st<COSTTYPE, SIMDWIDTH>>::max());

        /* retrieve label for parent's index */
        _iv_t<COSTTYPE, SIMDWIDTH> l_p = iv_init<COSTTYPE, SIMDWIDTH>(
            costs.c_labels->label_from_offset(m_node.parent_id, l_p_i));

        /* handle all this node's labels */
        for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < num_node_labels;
            l_i += SIMDWIDTH)
        {
            /* vector holding costs */
            _v_t<COSTTYPE, SIMDWIDTH> cost = v_init<COSTTYPE, SIMDWIDTH>();

            /* retrieve label index vector for this offset */
            const _iv_t<COSTTYPE, SIMDWIDTH> v_l_i =
                iv_sequence<COSTTYPE, SIMDWIDTH>(l_i);

            /* retrieve label vector for this offset */
            _iv_t<COSTTYPE, SIMDWIDTH> l = costs.c_labels->
                labels_from_offset(node_id(), l_i);

            /* mask out label indices exceeding num_node_labels */
            valid_labels = iv_le<COSTTYPE, SIMDWIDTH>
                (v_l_i, iv_init<COSTTYPE, SIMDWIDTH>(num_node_labels - 1));
            l = iv_blend<COSTTYPE, SIMDWIDTH>(
                iv_init<COSTTYPE, SIMDWIDTH>(exceed_l), l, valid_labels);

            /* determine weight to parent */
            _v_t<COSTTYPE, SIMDWIDTH> w_p = v_init<COSTTYPE, SIMDWIDTH>(
                m_node.to_parent_weight);

            /* add pairwise costs to parent (multiplied by weight) */
            if(!is_root)
            {
                if(costs.c_binary->node_dependent())
                    cost = v_add<COSTTYPE, SIMDWIDTH>(
                        v_mult<COSTTYPE, SIMDWIDTH>(w_p,
                        costs.c_binary->get_binary_costs(parent_id(),
                        l_p, node_id(), l)), cost);
                else
                    cost = v_add<COSTTYPE, SIMDWIDTH>(
                        v_mult<COSTTYPE, SIMDWIDTH>(w_p,
                        costs.c_binary->get_binary_costs(l_p, l)), cost);
            }

            /* add unary cost for node */
            if(costs.c_unary->supports_enumerable_costs())
                cost = v_add<COSTTYPE, SIMDWIDTH>(cost, costs.c_unary->
                    get_unary_costs_enum_offset(node_id(), l_i));
            else
                cost = v_add<COSTTYPE, SIMDWIDTH>(cost, costs.c_unary->
                    get_unary_costs(node_id(), l));

            /* add pairwise costs for dependencies (if applicable) */
            for(luint_t d_i = 0; costs.respect_dependencies &&
                (d_i < m_node.dependency_degree); ++d_i)
            {
                const luint_t d = m_node.dependency_ids[d_i];
                const _s_t<COSTTYPE, SIMDWIDTH> w =
                    m_node.dependency_weights[d_i];

                /* fetch dependency label index */
                _iv_st<COSTTYPE, SIMDWIDTH> l_d_i = (*costs.c_assignment)[d];
                _iv_t<COSTTYPE, SIMDWIDTH> l_d = iv_init<COSTTYPE, SIMDWIDTH>(
                    costs.c_labels->label_from_offset(d, l_d_i));

                /* add pairwise cost to dependency (multiplied by weight) */
                if(costs.c_binary->node_dependent())
                {
                    const _v_t<COSTTYPE, SIMDWIDTH> d_c =
                        costs.c_binary->get_binary_costs(d, l_d, node_id(), l);

                    cost = (w != 1.0) ?
                        v_add<COSTTYPE, SIMDWIDTH>(cost,
                        v_mult<COSTTYPE, SIMDWIDTH>(d_c,
                        v_init<COSTTYPE, SIMDWIDTH>(w))) :
                        v_add<COSTTYPE, SIMDWIDTH>(cost, d_c);
                }
                else
                {
                    const _v_t<COSTTYPE, SIMDWIDTH> d_c =
                        costs.c_binary->get_binary_costs(l_d, l);

                    cost = (w != 1.0) ?
                        v_add<COSTTYPE, SIMDWIDTH>(cost,
                        v_mult<COSTTYPE, SIMDWIDTH>(d_c,
                        v_init<COSTTYPE, SIMDWIDTH>(w))) :
                        v_add<COSTTYPE, SIMDWIDTH>(cost, d_c);
                }
            }

            /* add children's table entries */
            for(luint_t i = 0; i < m_node.degree; ++i)
            {
                const _s_t<COSTTYPE, SIMDWIDTH> * c_vals =
                    (*costs.c_child_values)[m_node.children_ids[i]];

                /* load corresponding optima from c's table */
                _v_t<COSTTYPE, SIMDWIDTH> c_dp = v_load<COSTTYPE, SIMDWIDTH>(
                    &c_vals[l_i]);

                cost = v_add<COSTTYPE, SIMDWIDTH>(cost, c_dp);
            }

            /**
             * SSE quirk: doing cmple (v_le) results in 0xff... if the first
             * operand is smaller, whereas blendv (v_blend) would copy
             * the second operand given 0xff....
             * Hence, in v_blend, we flip the arguments.
             *
             * Here, mask out costs for labels not in this node's table.
             */
            cost = v_blend<COSTTYPE, SIMDWIDTH>(
                v_init<COSTTYPE, SIMDWIDTH>(std::numeric_limits<
                _s_t<COSTTYPE, SIMDWIDTH>>::max()), cost,
                iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(valid_labels));

            /* for a root, just save costs */
            if(is_root)
            {
                v_store<COSTTYPE, SIMDWIDTH>(cost, &m_opt_values[l_i]);
                iv_store<COSTTYPE, SIMDWIDTH>(v_l_i, &m_opt_labels[l_i]);
            }
            else
            {
                /* determine componentwise minimum */
                _v_t<COSTTYPE, SIMDWIDTH> min_mask =
                    v_le<COSTTYPE, SIMDWIDTH>(cost, min_costs);

                /* update componentwise minimas (values + label indices) */
                min_costs = v_blend<COSTTYPE, SIMDWIDTH>(min_costs, cost,
                    min_mask);
                min_labels = iv_blend<COSTTYPE, SIMDWIDTH>(min_labels, v_l_i,
                    v_reinterpret_iv<COSTTYPE, SIMDWIDTH>(min_mask));
            }
        }

        /**
         * From the loop above, we have a vector of SIMDWIDTH minimum
         * candidates, now need to reduce it to find the minimum and
         * save it.
         */
        if(!is_root)
        {
            /* reduce the componentwise minimum */
            v_store<COSTTYPE, SIMDWIDTH>(min_costs, r_cost);
            iv_store<COSTTYPE, SIMDWIDTH>(min_labels, r_label);

            uint_t min_ix = 0;

            for(uint_t ix = 0; ix < SIMDWIDTH; ++ix)
                if(r_cost[ix] < r_cost[min_ix])
                    min_ix = ix;

            /* save DP table entry */
            m_opt_values[l_p_i] = r_cost[min_ix];

            /* save minimum label (index) */
            m_opt_labels[l_p_i] = r_label[min_ix];
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const _s_t<COSTTYPE, SIMDWIDTH> *
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimal_value()
{
    return m_opt_values;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH> *
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimal_labels()
{
    return m_opt_labels;
}

/**
 * *****************************************************************************
 * *********************** CombinatorialDynamicProgramming *********************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
CombinatorialDynamicProgramming()
: m_level(0),
  m_level_size(0),
  m_value_allocator((tbb::tbb_allocator<_s_t<COSTTYPE, SIMDWIDTH>>*)
    (new tbb::cache_aligned_allocator<_s_t<COSTTYPE, SIMDWIDTH>>))
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~CombinatorialDynamicProgramming()
{
    /* free all node optimization spaces */
#if defined(BUILD_MEMORY_SAVE)
    for(const luint_t r : m_root_ids)
        m_value_allocator->deallocate(m_opt_value_nodes[r],
            m_opt_value_sizes[r]);
#endif
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize(
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
throw()
{
    if(!this->data_complete())
        throw std::domain_error("Data for optimization problem "
            "incomplete!");

    /* search for leafs as initial nodes to optimize */
    this->discover_leaves();

    /* allocate required memory */
    this->allocate_memory();

    /* bottom-up pass: create table */
    bottom_up_opt();

    /* top-down pass: assign labels for minimum solution */
    top_down_opt(solution);

    return this->objective(solution);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
discover_leaves()
{
    tbb::blocked_range<luint_t> tree_range(0, this->m_tree->num_graph_nodes());
    std::vector<luint_t> leaf_list(this->m_tree->num_graph_nodes(),
        (luint_t) 0);
    std::vector<luint_t> leaf_offsets(this->m_tree->num_graph_nodes(),
        (luint_t) 0);

    /* Leaf = node without children, count offsets */
    tbb::parallel_for(tree_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                leaf_list[i] = (this->m_tree->node(i).is_in_tree &&
                    this->m_tree->node(i).degree == 0);
        });

    PlusScan<luint_t, luint_t> p_scan(&leaf_list[0], &leaf_offsets[0]);
    tbb::parallel_scan(tree_range, p_scan);
    const luint_t num_leaves = leaf_offsets.back() + leaf_list.back();

    /* save leaf IDs in vector */
    m_leaf_ids = std::vector<luint_t>(num_leaves, invalid_luint_t);

    tbb::parallel_for(tree_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                if(leaf_list[i] > 0)
                    m_leaf_ids[leaf_offsets[i]] = i;
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
allocate_memory()
{
    tbb::blocked_range<luint_t> leaf_range(0, m_leaf_ids.size(), 32u);
    tbb::blocked_range<luint_t> node_range(0, this->m_tree->num_graph_nodes(),
        32u);

    /* create pointer table for nodes' value tables (and their sizes) */
    m_opt_value_nodes = std::vector<_s_t<COSTTYPE, SIMDWIDTH>*>(
        this->m_tree->num_graph_nodes());
    m_opt_value_sizes = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>(
        this->m_tree->num_graph_nodes());

    /* allocate memory to hold the DP table for indices for all nodes */
    std::vector<luint_t> lbl_set_sizes(this->m_tree->num_graph_nodes(),
        (luint_t) 0);
    std::vector<luint_t> lbl_set_offsets(this->m_tree->num_graph_nodes(),
        (luint_t) 0);

    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                /* root nodes (= self-parented) are included here */
                if(this->m_tree->node(i).parent_id != invalid_luint_t)
                    lbl_set_sizes[i] = SIMDWIDTH * DIV_UP(
                        this->m_label_set->label_set_size(
                        this->m_tree->node(i).parent_id), SIMDWIDTH);
            }
        });

    PlusScan<luint_t, luint_t> p_scan(&lbl_set_sizes[0], &lbl_set_offsets[0]);
    tbb::parallel_scan(node_range, p_scan);
    const luint_t total_mem_req = lbl_set_offsets.back() + lbl_set_sizes.back();

    /* index table for all nodes */
    m_opt_labels = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>(total_mem_req,
        (uint_t) invalid_luint_t);

#ifndef BUILD_MEMORY_SAVE
    /* value table for all nodes */
    m_opt_values = std::vector<_s_t<COSTTYPE, SIMDWIDTH>>(total_mem_req);
#endif

    /* create pointer tables for indices for all nodes */
    m_opt_label_nodes = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>*>(
        this->m_tree->num_graph_nodes());

    /* save offsets into index table for all nodes */
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                if(this->m_tree->node(i).parent_id != invalid_luint_t)
                {
                    m_opt_label_nodes[i] = &m_opt_labels[0] +
                        lbl_set_offsets[i];
#ifndef BUILD_MEMORY_SAVE
                    m_opt_value_nodes[i] = &m_opt_values[0] +
                        lbl_set_offsets[i];
#endif
                }
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
node_memory_allocate(
    const luint_t node_id)
{
    /* determine number of needed storage spaces */
    const _iv_st<COSTTYPE, SIMDWIDTH> entry_size = SIMDWIDTH *
        DIV_UP(this->m_label_set->label_set_size(
        this->m_tree->node(node_id).parent_id), SIMDWIDTH);

    /* allocate memory */
    m_opt_value_nodes[node_id] = m_value_allocator->allocate(entry_size);
    m_opt_value_sizes[node_id] = entry_size;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
node_memory_clean_children(
    const luint_t node_id)
{
    /* deallocate children's value tables */
    const TreeNode<COSTTYPE> tree_n = this->m_tree->node(node_id);
    const luint_t degree = tree_n.degree;
    const luint_t * children = tree_n.children_ids;

    for(luint_t d = 0; d < degree; ++d)
    {
        const luint_t c = children[d];
        m_value_allocator->deallocate(m_opt_value_nodes[c],
            m_opt_value_sizes[c]);
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
bottom_up_opt()
{
    /* mark the number of unprocessed children atomically per node */
    std::vector<tbb::atomic<luint_t>> unproc_children(
        this->m_tree->num_graph_nodes(), 0);

    /* fill child counter for all node's parents */
    tbb::blocked_range<luint_t> node_range(0, this->m_graph->num_nodes());
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
                unproc_children[i] = this->m_tree->node(i).degree;
        });

    /* create a bundle with all necessary information for DP */
    DPBundle<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE> dpb;
    dpb.c_labels = this->m_label_set;
    dpb.c_unary = this->m_unaries;
    dpb.c_binary = this->m_pairwise;
    dpb.c_child_values = &m_opt_value_nodes;
    dpb.c_child_labels = &m_opt_label_nodes;
    dpb.c_assignment = &this->m_current_assignment;
    dpb.respect_dependencies = this->m_uses_dependencies;

    /* use feeder instead of level-wise queue */
    tbb::concurrent_vector<luint_t> queue;
    queue.assign(m_leaf_ids.begin(), m_leaf_ids.end());

    int processed = 0;
    tbb::parallel_do(queue.begin(), queue.end(),
        [&](const luint_t n, tbb::parallel_do_feeder<luint_t>& feeder)
        {
                /* allocate memory */
#if defined(BUILD_MEMORY_SAVE)
                node_memory_allocate(n);
#endif

                /* create one table entry per node */
                DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH, UNARY,
                    PAIRWISE> dpe(this->m_tree->node(n), m_opt_value_nodes[n],
                        m_opt_label_nodes[n]);

                /* start DP for this node */
                dpe.optimize_entry(dpb);

                /* decrement parent's unprocessed children counter */
                const luint_t parent_id = this->m_tree->node(n).parent_id;
                if(parent_id != n && unproc_children[parent_id].
                    fetch_and_decrement() == (luint_t) 1)
                {
                    /* last child processed: push parent into next level */
                    feeder.add(parent_id);
                }

                /* free children's memory */
#if defined(BUILD_MEMORY_SAVE)
                node_memory_clean_children(n);
#endif

                /* for roots - record for top-down pass */
                if(parent_id == n)
                    m_root_ids.push_back(n);

                ++processed;
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
top_down_opt(
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
{
    /* find minimum label (index) for each root */
    tbb::concurrent_vector<luint_t> queue;
    queue.reserve(this->m_graph->num_nodes());
    tbb::blocked_range<luint_t> root_range(0, m_root_ids.size());

    tbb::parallel_for(root_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t i = r.begin(); i != r.end(); ++i)
            {
                const luint_t root = m_root_ids[i];
                const uint_t r_label_set_size = this->m_label_set->
                    label_set_size(root);

                const _s_t<COSTTYPE, SIMDWIDTH> * my_costs =
                    m_opt_value_nodes[root];
                const _iv_st<COSTTYPE, SIMDWIDTH> * my_labels =
                    m_opt_label_nodes[root];

                _s_t<COSTTYPE, SIMDWIDTH> min_costs = my_costs[0];
                _iv_st<COSTTYPE, SIMDWIDTH> min_label = my_labels[0];
                for(uint_t l_i = 0; l_i < r_label_set_size; ++l_i)
                {
                    if(my_costs[l_i] < min_costs)
                    {
                        min_costs = my_costs[l_i];
                        min_label = my_labels[l_i];
                    }
                }

                solution[root] = min_label;

                /* add children to queue */
                for(luint_t i = 0; i < this->m_tree->node(root).degree; ++i)
                    queue.push_back(this->m_tree->node(root).children_ids[i]);
            }
        });

    /* continue traversal */
    tbb::parallel_do(queue.begin(), queue.end(),
        [&](const luint_t n, tbb::parallel_do_feeder<luint_t>& feeder)
        {
            /* retrieve current node */
            const TreeNode<COSTTYPE>& node =
                this->m_tree->node(n);

            /* retrieve parent's label (index) */
            const _iv_st<COSTTYPE, SIMDWIDTH> p_label =
                solution[node.parent_id];

            /* set n's label (index) */
            const _iv_st<COSTTYPE, SIMDWIDTH> * my_labels =
                m_opt_label_nodes[n];
            solution[n] = my_labels[p_label];

            /* add children to queue */
            for(luint_t i = 0; i < node.degree; ++i)
                feeder.add(node.children_ids[i]);
        });
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_iv_st<COSTTYPE, SIMDWIDTH>*
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
index_ptr()
{
    return &m_opt_labels[0];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
std::vector<luint_t> *
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
current_level_queue_ptr()
{
    /* _a for even levels, _b else */
    return ((m_level & 0x1) ? &m_queue_b : &m_queue_a);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
std::vector<luint_t> *
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
next_level_queue_ptr()
{
    /* _b for even levels, _a else */
    return ((m_level & 0x1) ? &m_queue_a : &m_queue_b);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
next_level()
{
    ++m_level;
}

NS_MAPMAP_END
