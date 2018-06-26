/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/dp_node_solver.h"

#include <limits>

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * *************************** GeneralDPNodeSolver *****************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
GeneralDPNodeSolver(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
: m_node(node)
{
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~GeneralDPNodeSolver()
{
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize_node()
{
    /* save pointer to node for member functions */
    const luint_t parent_id = m_node->c_node.parent_id;
    const luint_t node_id = m_node->c_node.node_id;

    /* typed costs for performance optimization */
    const PAIRWISE * c_pairwise = (const PAIRWISE *) m_node->c_pairwise;

    /* save pointers to allocated scratch space */
    m_icost_table = m_node->c_scratch;

    /* determine if we are optimizing a root node */
    const bool is_root = (node_id == parent_id);

    /* optimize a single node - if it's a root, assume an artificial parent */
    const _iv_st<COSTTYPE, SIMDWIDTH> num_parent_labels = is_root ? 1 :
        m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> num_node_labels =
        m_node->c_labels->label_set_size(node_id);

    /* precompute parent-independent costs */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < num_node_labels;
        l_i += SIMDWIDTH)
        v_store<COSTTYPE, SIMDWIDTH>(get_independent_of_parent_costs(l_i),
        &m_icost_table[l_i]);

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
    const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l = m_node->c_labels->
        label_from_offset(node_id, 0);

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
            m_node->c_labels->label_from_offset(parent_id, l_p_i));

        /* handle all this node's labels */
        for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < num_node_labels;
            l_i += SIMDWIDTH)
        {
            /* vector holding costs */
            _v_t<COSTTYPE, SIMDWIDTH> cost =
                v_load<COSTTYPE, SIMDWIDTH>(&m_icost_table[l_i]);

            /* retrieve label index vector for this offset */
            _iv_t<COSTTYPE, SIMDWIDTH> v_l_i =
                iv_sequence<COSTTYPE, SIMDWIDTH>(l_i);

            /* retrieve label vector for this offset */
            _iv_t<COSTTYPE, SIMDWIDTH> l = m_node->c_labels->
                labels_from_offset(node_id, l_i);

            /* mask out label indices exceeding num_node_labels */
            valid_labels = iv_le<COSTTYPE, SIMDWIDTH>
                (v_l_i, iv_init<COSTTYPE, SIMDWIDTH>(num_node_labels - 1));
            v_l_i = iv_blend<COSTTYPE, SIMDWIDTH>(
                iv_init<COSTTYPE, SIMDWIDTH>(0), v_l_i, valid_labels);
            l = iv_blend<COSTTYPE, SIMDWIDTH>(
                iv_init<COSTTYPE, SIMDWIDTH>(exceed_l), l, valid_labels);

            /* determine weight to parent */
            _v_t<COSTTYPE, SIMDWIDTH> w_p = v_init<COSTTYPE, SIMDWIDTH>(
                m_node->c_node.to_parent_weight);

            /* add pairwise costs to parent (multiplied by weight) */
            if(!is_root)
            {
                /* determine the order of labelling in the edge */
                const luint_t edge_id = m_node->c_node.to_parent_edge_id;
                const luint_t edge_a = m_node->c_graph->edges()[edge_id].node_a;
                const bool parent_is_node_a = (edge_a == parent_id);

                if(c_pairwise->supports_enumerable_costs())
                    cost = v_add<COSTTYPE, SIMDWIDTH>(
                        v_mult<COSTTYPE, SIMDWIDTH>(w_p,
                        c_pairwise->get_pairwise_costs_enum_offsets(
                        parent_is_node_a ?
                        iv_init<COSTTYPE, SIMDWIDTH>(l_p_i) : v_l_i,
                        parent_is_node_a ?
                        v_l_i : iv_init<COSTTYPE, SIMDWIDTH>(l_p_i))), cost);
                else
                    cost = v_add<COSTTYPE, SIMDWIDTH>(
                        v_mult<COSTTYPE, SIMDWIDTH>(w_p,
                        c_pairwise->get_pairwise_costs(
                        parent_is_node_a ? l_p : l,
                        parent_is_node_a ? l : l_p)), cost);
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
                v_store<COSTTYPE, SIMDWIDTH>(cost, &m_node->c_opt_values[l_i]);
                iv_store<COSTTYPE, SIMDWIDTH>(v_l_i,
                    &m_node->c_opt_labels[l_i]);
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
            m_node->c_opt_values[l_p_i] = r_cost[min_ix];

            /* save minimum label (index) */
            m_node->c_opt_labels[l_p_i] = r_label[min_ix];
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
luint_t
GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed()
{
    const luint_t node_id = m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels_padded =
        DIV_UP(node_num_labels, SIMDWIDTH) * SIMDWIDTH;

    return (node_num_labels_padded * sizeof(_s_t<COSTTYPE, SIMDWIDTH>));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
_v_t<COSTTYPE, SIMDWIDTH>
GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
get_independent_of_parent_costs(
    const _iv_st<COSTTYPE, SIMDWIDTH> l_i)
{
    const luint_t node_id = m_node->c_node.node_id;

    /* typed costs for performance optimization */
    const UNARY * c_unary = (const UNARY *) m_node->c_unary;

    const _iv_st<COSTTYPE, SIMDWIDTH> num_node_labels =
        m_node->c_labels->label_set_size(node_id);

    /* label to substitute for exceeding label indices */
    const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l = m_node->c_labels->
        label_from_offset(node_id, 0);

    /* vector holding costs */
    _v_t<COSTTYPE, SIMDWIDTH> cost = v_init<COSTTYPE, SIMDWIDTH>();

    /* retrieve label vector for this offset */
    _iv_t<COSTTYPE, SIMDWIDTH> l = m_node->c_labels->
        labels_from_offset(node_id, l_i);

    /* retrieve label index vector for this offset */
    _iv_t<COSTTYPE, SIMDWIDTH> v_l_i = iv_sequence<COSTTYPE, SIMDWIDTH>(l_i);

    /* mask for excluding invalid labels */
    _iv_t<COSTTYPE, SIMDWIDTH> valid_labels = iv_le<COSTTYPE, SIMDWIDTH>(
        v_l_i, iv_init<COSTTYPE, SIMDWIDTH>(num_node_labels - 1));
    v_l_i = iv_blend<COSTTYPE, SIMDWIDTH>(iv_init<COSTTYPE, SIMDWIDTH>(0),
        v_l_i, valid_labels);
    l = iv_blend<COSTTYPE, SIMDWIDTH>(iv_init<COSTTYPE, SIMDWIDTH>(exceed_l), l,
        valid_labels);

    /* add unary cost for node */
    cost = c_unary->supports_enumerable_costs() ?
        v_add<COSTTYPE, SIMDWIDTH>(cost,
        c_unary->get_unary_costs_enum_offset(l_i)) :
        v_add<COSTTYPE, SIMDWIDTH>(cost, c_unary->get_unary_costs(l));

    /* add pairwise costs for dependencies (if applicable) */
    for(luint_t d_i = 0; m_node->respect_dependencies &&
        (d_i < m_node->c_node.dependency_degree); ++d_i)
    {
        const luint_t d = m_node->c_node.dependency_ids[d_i];
        const _s_t<COSTTYPE, SIMDWIDTH> w =
            m_node->c_node.dependency_weights[d_i];

        /* fetch dependency label index */
        const _iv_st<COSTTYPE, SIMDWIDTH> l_d_i = (*m_node->c_assignment)[d];
        const _iv_t<COSTTYPE, SIMDWIDTH> l_d = iv_init<COSTTYPE, SIMDWIDTH>(
            m_node->c_labels->label_from_offset(d, l_d_i));

        /* add pairwise cost to dependency (multiplied by weight) */
        const luint_t edge_id = m_node->c_node.dependency_edge_ids[d_i];
        const luint_t edge_a = m_node->c_graph->edges()[edge_id].node_a;
        const bool node_is_node_a = (edge_a == node_id);

        const _v_t<COSTTYPE, SIMDWIDTH> d_c =
            m_node->c_dep_costs[d_i]->supports_enumerable_costs() ?
            m_node->c_dep_costs[d_i]->get_pairwise_costs_enum_offsets(
                node_is_node_a ?
                v_l_i : iv_init<COSTTYPE, SIMDWIDTH>(l_d_i),
                node_is_node_a ?
                iv_init<COSTTYPE, SIMDWIDTH>(l_d_i) : v_l_i) :
            m_node->c_dep_costs[d_i]->get_pairwise_costs(
                node_is_node_a ? l : l_d,
                node_is_node_a ? l_d : l);

        cost = (w != 1.0) ?
            v_add<COSTTYPE, SIMDWIDTH>(cost,
            v_mult<COSTTYPE, SIMDWIDTH>(d_c,
            v_init<COSTTYPE, SIMDWIDTH>(w))) :
            v_add<COSTTYPE, SIMDWIDTH>(cost, d_c);
    }

    /* add children's table entries */
    for(luint_t i = 0; i < m_node->c_node.degree; ++i)
    {
        const _s_t<COSTTYPE, SIMDWIDTH> * c_vals =
            (*m_node->c_child_values)[m_node->c_node.children_ids[i]];

        /* load corresponding optima from c's table */
        _v_t<COSTTYPE, SIMDWIDTH> c_dp = v_load<COSTTYPE, SIMDWIDTH>(
            &c_vals[l_i]);

        cost = v_add<COSTTYPE, SIMDWIDTH>(cost, c_dp);
    }

    cost = v_blend<COSTTYPE, SIMDWIDTH>(
        v_init<COSTTYPE, SIMDWIDTH>(std::numeric_limits<
        _s_t<COSTTYPE, SIMDWIDTH>>::max()), cost,
        iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(valid_labels));

    return cost;
}

/**
 * *****************************************************************************
 * ************************* SubmodularDPNodeSolver ****************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
SubmodularDPNodeSolver(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>& env)
: GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>(node),
  m_env(env)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~SubmodularDPNodeSolver()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize_node()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const luint_t parent_id = this->m_node->c_node.parent_id;

    /* save pointers to allocated scratch space */
    const _iv_st<COSTTYPE, SIMDWIDTH> p_len =
        this->m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> n_len =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels_padded =
        DIV_UP(node_num_labels, SIMDWIDTH) * SIMDWIDTH;

    m_mprime = this->m_node->c_scratch;
    m_icost_cache = m_mprime + p_len + n_len;
    m_label_union = (_iv_st<COSTTYPE, SIMDWIDTH> *) (m_icost_cache +
        node_num_labels_padded);
    m_mprime_ix = (_iv_st<COSTTYPE, SIMDWIDTH> *) (m_label_union +
        p_len + n_len);

    /* compute label union */
    label_union();

    /* fill independent-of-parent cost cache */
    fill_icost_cache();

    /* root: only copy inode cache */
    if(node_id == parent_id)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels = this->m_node->
            c_labels->label_set_size(parent_id);
        std::copy(m_icost_cache, m_icost_cache + parent_num_labels,
            this->m_node->c_opt_values);

        _iv_t<COSTTYPE, SIMDWIDTH> lbl;
        for(_iv_st<COSTTYPE, SIMDWIDTH> i = 0; i < parent_num_labels;
            i += SIMDWIDTH)
        {
            lbl = iv_sequence<COSTTYPE, SIMDWIDTH>(i);
            iv_store<COSTTYPE, SIMDWIDTH>(lbl, &this->m_node->c_opt_labels[i]);
        }

        return;
    }

    /* use cache to compute m' */
    compute_mp();

    /* retrieve cost bound d */
    const _s_t<COSTTYPE, SIMDWIDTH> d = m_env->cost_bound_d();

    /* execute pointwise minimization similar to Potts model */
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels = this->m_node->
        c_labels->label_set_size(parent_id);
    _s_t<COSTTYPE, SIMDWIDTH> v_min_fp_cost = m_min_fp_cost +
        this->m_node->c_node.to_parent_weight * d;
    _iv_st<COSTTYPE, SIMDWIDTH> iv_min_fp = m_min_fp;

    _iv_st<COSTTYPE, SIMDWIDTH> u_ptr = 0;
    for(_iv_st<COSTTYPE, SIMDWIDTH> p_l_i = 0; p_l_i < parent_num_labels;
        ++p_l_i)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> p_l =
            this->m_node->c_labels->label_from_offset(parent_id, p_l_i);

        while(u_ptr < m_label_union_size && m_label_union[u_ptr] < p_l)
            ++u_ptr;

        if(m_label_union[u_ptr] == p_l)
        {
            this->m_node->c_opt_values[p_l_i] = std::min(m_mprime[u_ptr],
                v_min_fp_cost);
            this->m_node->c_opt_labels[p_l_i] =
                (m_mprime[u_ptr] < v_min_fp_cost) ?
                m_mprime_ix[u_ptr] : iv_min_fp;
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
luint_t
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels_padded =
        DIV_UP(node_num_labels + 1, SIMDWIDTH) * SIMDWIDTH;

    const luint_t parent_id = this->m_node->c_node.parent_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels =
        this->m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels_padded =
        DIV_UP(parent_num_labels + 1, SIMDWIDTH) * SIMDWIDTH;

    return ((3 * node_num_labels_padded + 4 * parent_num_labels_padded) *
        sizeof(_s_t<COSTTYPE, SIMDWIDTH>)
        + this->m_env->scratch_bytes_needed(this->m_node));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
fill_icost_cache()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);

    const _iv_st<COSTTYPE, SIMDWIDTH> exceed_l = this->m_node->c_labels->
        label_from_offset(node_id, 0);

    _v_t<COSTTYPE, SIMDWIDTH> min_costs = v_init<COSTTYPE, SIMDWIDTH>(
        std::numeric_limits<COSTTYPE>::max());
    _iv_t<COSTTYPE, SIMDWIDTH> min_labels =
        iv_init<COSTTYPE, SIMDWIDTH>(exceed_l);
    _iv_t<COSTTYPE, SIMDWIDTH> valid_labels;
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < node_num_labels;
        l_i += SIMDWIDTH)
    {
        /* compute vector */
        _v_t<COSTTYPE, SIMDWIDTH> icost =
            this->get_independent_of_parent_costs(l_i);
        v_store<COSTTYPE, SIMDWIDTH>(icost, &m_icost_cache[l_i]);

        /* retrieve label vector for this offset */
        _iv_t<COSTTYPE, SIMDWIDTH> v_l_i =
            iv_sequence<COSTTYPE, SIMDWIDTH>(l_i);

        /* mask out label indices exceeding node_num_labels */
        valid_labels = iv_le<COSTTYPE, SIMDWIDTH>
            (v_l_i, iv_init<COSTTYPE, SIMDWIDTH>(node_num_labels - 1));
        v_l_i = iv_blend<COSTTYPE, SIMDWIDTH>(
            iv_init<COSTTYPE, SIMDWIDTH>(exceed_l), v_l_i, valid_labels);

        /* similarly, mask out costs */
        icost = v_blend<COSTTYPE, SIMDWIDTH>(
            v_init<COSTTYPE, SIMDWIDTH>(std::numeric_limits<
            _s_t<COSTTYPE, SIMDWIDTH>>::max()), icost,
            iv_reinterpret_v<COSTTYPE, SIMDWIDTH>(valid_labels));

        /* determine componentwise minimum */
        _v_t<COSTTYPE, SIMDWIDTH> min_mask =
            v_le<COSTTYPE, SIMDWIDTH>(icost, min_costs);

        /* update componentwise minimas (values + label indices) */
        min_costs = v_blend<COSTTYPE, SIMDWIDTH>(min_costs, icost,
            min_mask);
        min_labels = iv_blend<COSTTYPE, SIMDWIDTH>(min_labels, v_l_i,
            v_reinterpret_iv<COSTTYPE, SIMDWIDTH>(min_mask));
    }

    /* compute minimum i-cost and its label from SIMDWIDTH-vector */
    _s_t<COSTTYPE, SIMDWIDTH> cost_store[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> lbl_store[SIMDWIDTH];

    v_store<COSTTYPE, SIMDWIDTH>(min_costs, cost_store);
    iv_store<COSTTYPE, SIMDWIDTH>(min_labels, lbl_store);

    luint_t min_ix = 0;
    for(luint_t i = 1; i < SIMDWIDTH; ++i)
        if(cost_store[i] < cost_store[min_ix])
            min_ix = i;

    /* save minimum i-cost and label */
    m_min_fp_cost = cost_store[min_ix];
    m_min_fp = lbl_store[min_ix];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_mp()
{
    /* splatter parent's i-costs into the mprime initialization */
    const luint_t parent_id = this->m_node->c_node.parent_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> p_len =
        this->m_node->c_labels->label_set_size(parent_id);

    const _iv_st<COSTTYPE, SIMDWIDTH> max_diff =
        this->m_node->c_labels->label_from_offset(parent_id, p_len - 1) -
        this->m_node->c_labels->label_from_offset(parent_id, 0);
    std::fill(m_mprime, m_mprime + m_label_union_size, std::numeric_limits<
        _s_t<COSTTYPE, SIMDWIDTH>>::max() - max_diff *
        this->m_env->cost_bound_d());

    const luint_t node_id = this->m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> n_len =
        this->m_node->c_labels->label_set_size(node_id);

    _iv_st<COSTTYPE, SIMDWIDTH> u_ptr = 0;
    for(_iv_st<COSTTYPE, SIMDWIDTH> n_i = 0; n_i < n_len; ++n_i)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> n_l =
            this->m_node->c_labels->label_from_offset(node_id, n_i);

        while(u_ptr < m_label_union_size && m_label_union[u_ptr] < n_l)
            ++u_ptr;

        if(m_label_union[u_ptr] == n_l)
            m_mprime[u_ptr] = m_icost_cache[n_i];
    }

    /* compute address of scratch space for envelope */
    const luint_t solver_bytes = scratch_bytes_needed() -
        m_env->scratch_bytes_needed(this->m_node);
    _s_t<COSTTYPE, SIMDWIDTH> * env_scratch = (_s_t<COSTTYPE, SIMDWIDTH> *)
        ((char *) this->m_node->c_scratch + (DIV_UP(solver_bytes,
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>)) *
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>)));

    /* delegate to envelope */
    this->m_env->compute_m_primes(this->m_node, m_icost_cache,
        m_label_union, m_label_union_size, m_mprime, m_mprime_ix,
        env_scratch);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
label_union()
{
    const luint_t parent_id = this->m_node->c_node.parent_id;
    const luint_t node_id = this->m_node->c_node.node_id;

    m_label_union_size = 0;

    _iv_st<COSTTYPE, SIMDWIDTH> p_ptr = 0;
    _iv_st<COSTTYPE, SIMDWIDTH> n_ptr = 0;

    const _iv_st<COSTTYPE, SIMDWIDTH> p_len =
        this->m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> n_len =
        this->m_node->c_labels->label_set_size(node_id);

    const _iv_st<COSTTYPE, SIMDWIDTH> max_label = std::max(
        this->m_node->c_labels->label_from_offset(parent_id, p_len - 1),
        this->m_node->c_labels->label_from_offset(node_id, n_len - 1)) + 1;
    while(p_ptr < p_len || n_ptr < n_len)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> p_lbl = (p_ptr < p_len) ?
            this->m_node->c_labels->label_from_offset(parent_id, p_ptr) :
            max_label;
        const _iv_st<COSTTYPE, SIMDWIDTH> n_lbl = (n_ptr < n_len) ?
            this->m_node->c_labels->label_from_offset(node_id, n_ptr) :
            max_label;

        m_label_union[m_label_union_size] = std::min(p_lbl, n_lbl);
        m_mprime_ix[m_label_union_size] = (n_lbl <= p_lbl) ? n_ptr : -1;

        if(p_lbl <= n_lbl)
            ++p_ptr;
        if(n_lbl <= p_lbl)
            ++n_ptr;

        ++m_label_union_size;
    }
}

/**
 * *****************************************************************************
 * ************************ SupermodularDPNodeSolver ***************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
SupermodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
SupermodularDPNodeSolver(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>& env)
: SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>(node, env)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
SupermodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~SupermodularDPNodeSolver()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SupermodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize_node()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const luint_t parent_id = this->m_node->c_node.parent_id;

    /* save pointers to allocated scratch space */
    const _iv_st<COSTTYPE, SIMDWIDTH> p_len =
        this->m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> n_len =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels_padded =
        DIV_UP(node_num_labels, SIMDWIDTH) * SIMDWIDTH;

    this->m_mprime = this->m_node->c_scratch;
    this->m_icost_cache = this->m_mprime + p_len + n_len;
    this->m_label_union = (_iv_st<COSTTYPE, SIMDWIDTH> *) (this->m_icost_cache +
        node_num_labels_padded);
    this->m_mprime_ix = (_iv_st<COSTTYPE, SIMDWIDTH> *) (
        this->m_label_union + p_len + n_len);

    /* compute label union */
    this->label_union();

    /* fill independent-of-parent cost cache */
    fill_icost_cache();

    /* root: only copy inode cache */
    if(node_id == parent_id)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels = this->m_node->
            c_labels->label_set_size(parent_id);
        std::copy(this->m_icost_cache, this->m_icost_cache + parent_num_labels,
            this->m_node->c_opt_values);

        _iv_t<COSTTYPE, SIMDWIDTH> lbl;
        for(_iv_st<COSTTYPE, SIMDWIDTH> i = 0; i < parent_num_labels;
            i += SIMDWIDTH)
        {
            lbl = iv_sequence<COSTTYPE, SIMDWIDTH>(i);
            iv_store<COSTTYPE, SIMDWIDTH>(lbl, &this->m_node->c_opt_labels[i]);
        }

        return;
    }

    /* use cache to compute m' */
    this->compute_mp();

    /* retrieve cost factor c */
    const PAIRWISE * pcosts = (const PAIRWISE *) this->m_node->c_pairwise;
    const _s_t<COSTTYPE, SIMDWIDTH> d = pcosts->get_label_diff_cap();

    /* execute pointwise minimization similar to (Anti-) Potts model */
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels = this->m_node->
        c_labels->label_set_size(parent_id);

    const _iv_st<COSTTYPE, SIMDWIDTH> l_min_fp = this->m_node->c_labels->
        label_from_offset(node_id, this->m_min_fp);

    _iv_st<COSTTYPE, SIMDWIDTH> u_ptr = 0;
    for(_iv_st<COSTTYPE, SIMDWIDTH> p_l_i = 0; p_l_i < parent_num_labels;
        ++p_l_i)
    {
        const _iv_st<COSTTYPE, SIMDWIDTH> p_l =
            this->m_node->c_labels->label_from_offset(parent_id, p_l_i);

        while(u_ptr < this->m_label_union_size &&
                this->m_label_union[u_ptr] < p_l)
            ++u_ptr;

        if(this->m_label_union[u_ptr] != p_l)
            continue;

        if(std::abs(p_l - l_min_fp) <= d)
        {
            /* in the hot zone - use m_prime */
            this->m_node->c_opt_values[p_l_i] = this->m_mprime[u_ptr];
            this->m_node->c_opt_labels[p_l_i] = this->m_mprime_ix[u_ptr];
        }
        else
        {
            /* outside of the hot zone - use min-label */
            this->m_node->c_opt_values[p_l_i] = this->m_min_fp_cost;
            this->m_node->c_opt_labels[p_l_i] = this->m_min_fp;
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
luint_t
SupermodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels_padded =
        DIV_UP(node_num_labels, SIMDWIDTH) * SIMDWIDTH;

    const luint_t parent_id = this->m_node->c_node.parent_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels =
        this->m_node->c_labels->label_set_size(parent_id);
    const _iv_st<COSTTYPE, SIMDWIDTH> parent_num_labels_padded =
        DIV_UP(parent_num_labels + 1, SIMDWIDTH) * SIMDWIDTH;

    return ((3 * node_num_labels_padded + 4 * parent_num_labels_padded) *
        sizeof(_s_t<COSTTYPE, SIMDWIDTH>)
        + this->m_env->scratch_bytes_needed(this->m_node));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
void
SupermodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
fill_icost_cache()
{
    const luint_t node_id = this->m_node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        this->m_node->c_labels->label_set_size(node_id);

    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < node_num_labels;
        l_i += SIMDWIDTH)
    {
        /* compute vector */
        _v_t<COSTTYPE, SIMDWIDTH> icost =
            this->get_independent_of_parent_costs(l_i);
        v_store<COSTTYPE, SIMDWIDTH>(icost, &this->m_icost_cache[l_i]);
    }

    /* compute minimum cost */
    this->m_min_fp_cost = std::numeric_limits<COSTTYPE>::max();

    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < node_num_labels; ++l_i)
    {
        const _s_t<COSTTYPE, SIMDWIDTH> cost = this->m_icost_cache[l_i];

        if(cost < this->m_min_fp_cost)
        {
            this->m_min_fp_cost = cost;
            this->m_min_fp = l_i;
        }
    }
}

NS_MAPMAP_END