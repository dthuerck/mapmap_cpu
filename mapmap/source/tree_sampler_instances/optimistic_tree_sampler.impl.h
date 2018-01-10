/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/tree_sampler_instances/optimistic_tree_sampler.h"

#include <iostream>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_do.h"

#include "header/parallel_templates.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, bool ACYCLIC>
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
OptimisticTreeSampler(
    Graph<COSTTYPE> * graph)
: OptimisticTreeSampler<COSTTYPE, ACYCLIC>(graph, false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
OptimisticTreeSampler(
    Graph<COSTTYPE> * graph,
    const bool deterministic,
    const uint_t initial_seed)
: TreeSampler<COSTTYPE, ACYCLIC>(graph, deterministic, initial_seed)
{
    this->build_component_lists();
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
~OptimisticTreeSampler()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
select_random_roots(
    const luint_t k,
    std::vector<luint_t>& roots)
{
    roots.clear();
    std::vector<unsigned char> root_marker(this->m_graph->num_nodes(), 0u);

    /* constant seed if expected by the user */
    const uint_t seed = this->m_deterministic ? this->m_initial_seed :
        this->m_rnd_dev();
    std::mt19937 rnd(seed);

    /* make sure that k <= number of nodes */
    const luint_t num_components = this->m_graph->num_components();
    luint_t corrected_k = std::max(std::min(k, this->m_graph->num_nodes()),
        num_components);

    std::vector<luint_t> component_sizes(num_components, 0);
    for(luint_t c = 0; c < num_components; ++c)
        component_sizes[c] = this->m_component_lists[c].size();

    luint_t next_component;
    m_it = 0;

    while(roots.size() < corrected_k)
    {
        if (m_it < num_components)
        {
            /* first pass: make sure each component is covered */
            next_component = m_it;
        }
        else
        {
            /* second pass: select component proportional to its size */
            std::discrete_distribution<luint_t> dd(component_sizes.begin(),
                component_sizes.end());
            next_component = dd(rnd);
        }

        std::uniform_int_distribution<luint_t> ud(0,
            component_sizes[next_component] - 1);
        luint_t root_index = ud(rnd);
        luint_t next_root = this->m_component_lists[next_component]
            [root_index];

        /* defer conflict handling to main procedure */
        roots.push_back(next_root);
        root_marker[next_root] = 1u;

        /**
         * ease further node selection by putting selected node to
         * the end of the list
         */
        if(component_sizes[next_component] > 1)
            std::swap(this->m_component_lists[next_component][root_index],
                this->m_component_lists[next_component]
                [this->m_component_lists[next_component].size() - 1]);

        --component_sizes[next_component];

        ++m_it;
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
std::unique_ptr<Tree<COSTTYPE>>
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
sample(
    std::vector<luint_t>& roots,
    bool record_dependencies,
    bool relax)
{
    const luint_t num_nodes = this->m_graph->num_nodes();
    const luint_t num_edges = this->m_graph->edges().size();
    m_tree = std::unique_ptr<Tree<COSTTYPE>>(new Tree<COSTTYPE>(num_nodes,
        2 * num_edges));

    /* create acceleration structure for edge sampling */
    this->create_adj_acc();

    /* for early termination, count number of nodes remaining */
    m_rem_nodes = this->m_graph->num_nodes();

    /* work queues */
    m_w_in = &m_w_a;
    m_w_out = &m_w_b;

    m_w_in->reserve(num_nodes);
    m_w_out->reserve(num_nodes);
    m_w_new.reserve(num_nodes);

    /* clear markers */
    m_markers.clear();
    m_markers.resize(num_nodes, 0u);

    /* put roots into queue */
    for(const luint_t r : roots)
        m_tree->raw_parent_ids()[r] = r;
    m_w_new.assign(roots.begin(), roots.end());
    m_rem_nodes -= roots.size();

    /* copy original degrees and initialize locks */
    m_node_locks.resize(num_nodes);
    m_rem_degrees.resize(num_nodes);
    m_in_queue.resize(num_nodes);

    for(luint_t i = 0; i < num_nodes; ++i)
        m_rem_degrees[i] = this->m_graph->nodes()[i].incident_edges.size();
    std::fill(m_node_locks.begin(), m_node_locks.end(), 0);

    /* start actual growing process */
    luint_t it = 0;

    /* exploit first iteration to solve root conflicts */
    bool skip_ph1 = true;
    while(m_rem_nodes > 0 || skip_ph1)
    {
        m_w_in = (it % 2 == 0 ? &m_w_a : &m_w_b);
        m_w_out = (it % 2 == 0 ? &m_w_b : &m_w_a);

        m_w_out->clear();
        m_w_out->reserve(num_nodes);

        m_w_conflict.clear();
        std::fill(m_in_queue.begin(), m_in_queue.end(), 0);

        /* phase I: try growing new branches (or just copy selected nodes) */
        if(!skip_ph1)
        {
            m_w_new.clear();
            m_w_new.reserve(num_nodes);

            sample_phase_I();
        }
        else
        {
            m_w_out->assign(m_w_in->begin(), m_w_in->end());
        }

        /* phase II: update markers and detect collisions */
        sample_phase_II();

        if (ACYCLIC)
        {
            /* phase III: resolve conflicts */
            sample_phase_III();

            /* record roots from conflict-resolved rescue nodes */
            if(skip_ph1)
            {
                for(const luint_t& cand : (*m_w_in))
                {
                    if(m_tree->raw_parent_ids()[cand] == cand)
                    {
                        roots.push_back(cand);
                        --m_rem_nodes;
                    }
                }
            }
        }
        skip_ph1 = false;

        /* append new (not removed nodes to queue) */
        for(const luint_t& n : m_w_new)
            m_w_out->push_back(n);

        /**
         * if procedure gets stuck - add nodes with marker 0
         * as new nodes (that respect acyclicity).
         */
        if(ACYCLIC && !relax && m_w_out->empty())
        {
            sample_rescue();

            /* defer conflict-solving and root recording to later */
            skip_ph1 = true;
        }
        else
        {
            /* after rescue, defer break until after conflict handling */
            if(m_w_out->empty())
                break;
        }

        ++it;
    }

    /* gather children and finalize tree */
    m_tree->finalize(record_dependencies, this->m_graph);

    return std::move(m_tree);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
sample_phase_I()
{
    const uint_t base_seed = this->m_deterministic ? this->m_initial_seed :
        this->m_rnd_dev();
    tbb::blocked_range<luint_t> in_range(0, m_w_in->size());

    const auto fun_phaseI = [&](const tbb::blocked_range<luint_t>& r)
    {
        std::mt19937 rnd_gen(base_seed + 1 + r.begin());

        for (luint_t i = r.begin(); i != r.end(); ++i)
        {
            const luint_t in_node = (*m_w_in)[i];
            /**
             * due to lazy queue handling, removed nodes might be in the
             * queue, need to avoid that here
             */
            const bool is_in_tree = (m_tree->raw_parent_ids()[in_node] !=
                invalid_luint_t);

            if(!is_in_tree)
            {
                continue;
            }

            /**
             * try to acquire lock, if not accessible - then other thread
             * already handles this node, so just skip it
             */
            if(m_node_locks[in_node].compare_and_swap(1u, 0u) == 0u)
            {
                if(m_rem_degrees[in_node] > 0u)
                {
                    /* select random free incident edge */
                    std::uniform_int_distribution<luint_t> d(
                        0, m_rem_degrees[in_node] - 1);
                    const luint_t inc_list_ix = d(rnd_gen);
                    const luint_t e_id = this->m_adj[
                        this->m_adj_offsets[in_node] + inc_list_ix];
                    const GraphEdge<COSTTYPE> e =
                        this->m_graph->edges()[e_id];

                    /* extract corresponding adjacent node */
                    const luint_t o_node = (e.node_a == in_node ?
                        e.node_b : e.node_a);

                    /* try to acquire lock on adjacent node */
                    if(m_node_locks[o_node].compare_and_swap(1u, 0u)
                        == 0u)
                    {
                        /* check conditions for growing a branch */
                        const bool is_in_tree =
                            (m_tree->raw_parent_ids()[o_node] !=
                            invalid_luint_t);
                        const bool markers_exceed = (ACYCLIC ?
                            (m_markers[o_node] >= 2u) : false);

                        if(!is_in_tree && !markers_exceed)
                        {
                            /* grow branch to adjacent node */
                            m_tree->raw_parent_ids()[o_node] = in_node;
                            m_tree->raw_to_parent_edge_ids()[o_node] = e_id;
                            m_w_new.push_back(o_node);

                            /* one node less to consider */
                            --m_rem_nodes;
                        }

                        /* release lock */
                        m_node_locks[o_node] = 0u;
                    }

                    /**
                     * swap out selected entry in in_node's
                     * adjacency table
                     */
                    if(m_rem_degrees[in_node] > 1u)
                    {
                        std::swap(this->m_adj[this->m_adj_offsets[in_node] +
                            inc_list_ix],
                            this->m_adj[this->m_adj_offsets[in_node] +
                            m_rem_degrees[in_node] - 1]);

                        if(!m_in_queue[in_node].fetch_and_store(1))
                            m_w_out->push_back(in_node);
                    }
                    --m_rem_degrees[in_node];
                }

                /* release lock */
                m_node_locks[in_node] = 0u;
            }
            else
            {
                if(!m_in_queue[in_node].fetch_and_store(1))
                    m_w_out->push_back(in_node);
            }
        }
    };

    /* execute serially for deterministic sampling */
    if(this->m_deterministic)
        fun_phaseI(in_range);
    else
        tbb::parallel_for(in_range, fun_phaseI);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
sample_phase_II()
{
    tbb::blocked_range<luint_t> new_range(0, m_w_new.size());
    const auto fun_phaseII = [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t i = r.begin(); i != r.end(); ++i)
        {
            const luint_t new_node = m_w_new[i];

            /* update markers of adjacent nodes */
            for(const luint_t& e_id : this->m_graph->inc_edges(new_node))
            {
                const GraphEdge<COSTTYPE> e = this->m_graph->edges()[e_id];
                const luint_t o_node = (e.node_a == new_node ?
                        e.node_b : e.node_a);
                const bool o_in_tree =
                    (m_tree->raw_parent_ids()[o_node] != invalid_luint_t);

                /* don't check collisions with the parent node... */
                const bool is_parent = (m_tree->raw_parent_ids()[new_node]
                    == o_node);

                if(ACYCLIC && o_in_tree && !is_parent &&
                    (new_node < o_node))
                {
                    /**
                     * record conflict pair - only once by exploiting
                     * the node id's total ordering
                     */
                    m_w_conflict.push_back(std::make_pair(new_node,
                        o_node));
                }

                /* update marker and remove node from consideration */
                if(m_markers[o_node].fetch_and_increment() == 1 &&
                    !o_in_tree)
                    --m_rem_nodes;
            }
        }
    };

    /* execute serially for deterministic sampling */
    if(this->m_deterministic)
        fun_phaseII(new_range);
    else
        tbb::parallel_for(new_range, fun_phaseII);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
sample_phase_III()
{
    const luint_t num_conflicts_found = m_w_conflict.size();
    if (num_conflicts_found == 0)
        return;

    const uint_t base_seed = this->m_deterministic ? this->m_initial_seed :
        this->m_rnd_dev();

    tbb::concurrent_vector<luint_t> del;
    tbb::blocked_range<luint_t> conflict_range(0, num_conflicts_found);
    const auto fun_conflict = [&](const tbb::blocked_range<luint_t>& r)
    {
        std::mt19937 rnd_gen(base_seed + 1 + r.begin());
        std::uniform_int_distribution<luint_t> d(0, 1u);

        for(luint_t i = r.begin(); i != r.end(); ++i)
        {
            const std::pair<luint_t, luint_t> c_pair = m_w_conflict[i];

            /* select one node at random to remove */
            const luint_t remove_ix = d(rnd_gen);
            const luint_t remove_node = (remove_ix == 0 ? c_pair.first :
                c_pair.second);

            /**
             * a node could have multiple conflicts at the same time
             * and thus multiple removals would corrupt the marker
             * decrement process.
             */
            if(m_node_locks[remove_node].compare_and_swap(1u, 0u) == 0u)
            {
                const bool is_in_tree = (m_tree->raw_parent_ids()
                    [remove_node] != invalid_luint_t);

                /**
                 * skip removal operation if already happened by
                 * another conflict pair.
                 */
                if(is_in_tree)
                {
                    const luint_t old_parent = m_tree->raw_parent_ids()
                        [remove_node];
                    m_tree->raw_parent_ids()[remove_node] =
                        invalid_luint_t;
                    m_tree->raw_to_parent_edge_ids()[remove_node] =
                        invalid_luint_t;

                    del.push_back(remove_node);

                    /* if removing 'new' root, don't restore table */
                    if(remove_node != old_parent)
                        ++m_rem_degrees[old_parent];

                    /* rollback parent and put it into queue */
                    if(!m_in_queue[old_parent].fetch_and_store(1))
                        m_w_out->push_back(old_parent);
                }

                /* release lock */
                m_node_locks[remove_node] = 0u;
            }
        }
    };

    /* run serially for deterministic execution */
    if(this->m_deterministic)
        fun_conflict(conflict_range);
    else
        tbb::parallel_for(conflict_range, fun_conflict);

    /* update markers of nodes adjacent to deleted candidates */
    for(const luint_t& remove_node : del)
    {
        /* decrement marker of adjacent nodes */
        for(const luint_t& e_id : this->m_graph->inc_edges(remove_node))
        {
            const GraphEdge<COSTTYPE> e = this->m_graph->edges()[e_id];
            const luint_t o_node = (e.node_a == remove_node ?
                    e.node_b : e.node_a);
            const bool o_in_tree = (m_tree->raw_parent_ids()[o_node] !=
                invalid_luint_t);

            /* if marker-threshold passed while decrementing, reconsider node */
            if(m_markers[o_node].fetch_and_decrement() == 2 && !o_in_tree)
            {
                /* abuse m_in_queue to correctly handle reconsideration */
                m_in_queue[o_node] = 255;
                ++m_rem_nodes;
            }
        }
    }

    /* root case: put back deleted nodes if possible */
    for(const luint_t& remove_node : del)
    {
        if(m_markers[remove_node] < 2 && m_in_queue[remove_node] < 255)
            ++m_rem_nodes;
    }

    /* for correctness, make sure there are only tree-nodes in the queue */
    if(!del.empty())
    {
        /* filter removed nodes from m_w_new */
        std::vector<luint_t> filter_new;
        for(const luint_t& newn : m_w_new)
            if(m_tree->raw_parent_ids()[newn] != invalid_luint_t)
                filter_new.push_back(newn);

        m_w_new.assign(filter_new.begin(), filter_new.end());
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
OptimisticTreeSampler<COSTTYPE, ACYCLIC>::
sample_rescue()
{
    const uint_t seed = this->m_deterministic ? this->m_initial_seed :
        this->m_rnd_dev();
    std::mt19937 rnd(seed);

    /* find potential nodes with marker 0 */
    tbb::blocked_range<luint_t> node_range(0, this->m_graph->num_nodes());
    const auto fun_rescue = [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t i = r.begin(); i < r.end(); ++i)
        {
            const bool is_in_tree = (m_tree->raw_parent_ids()
                [i] != invalid_luint_t);

            if(!is_in_tree && m_markers[i] == 0)
            {
                m_tree->raw_parent_ids()[i] = i;

                m_w_new.push_back(i);
                --m_rem_nodes;
            }
        }
    };

    /* run serially for deterministic execution */
    if(this->m_deterministic)
        fun_rescue(node_range);
    else
        tbb::parallel_for(node_range, fun_rescue);

    /* resolve conflicts later */
}

NS_MAPMAP_END
