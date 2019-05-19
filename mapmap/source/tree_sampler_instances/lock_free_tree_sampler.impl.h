/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/tree_sampler_instances/lock_free_tree_sampler.h"

#include "header/color.h"

#include <random>
#include <set>
#include <algorithm>

#include "tbb/parallel_reduce.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * **************************** ColoredQueueBunch ******************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
ColoredQueueBunch<COSTTYPE>::
ColoredQueueBunch(
    Graph<COSTTYPE> * graph)
: ColoredQueueBunch<COSTTYPE>(graph, false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
ColoredQueueBunch<COSTTYPE>::
ColoredQueueBunch(
    Graph<COSTTYPE> * graph,
    const bool deterministic)
: m_graph(graph),
  m_num_qu(1),
  m_data(graph->num_nodes()),
  m_qu_start(),
  m_qu_pos(),
  m_size(0)
{
    /* create coloring if none available */
    if(!graph->was_colored())
    {
        std::vector<luint_t> coloring;
        Color<COSTTYPE> pen(*graph, deterministic);
        pen.color_graph(coloring);

        graph->set_coloring(coloring);
    }

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

        ++m_size;
    }
}

/* ************************************************************************** */

template<typename COSTTYPE>
void
ColoredQueueBunch<COSTTYPE>::
replace_queue(
    const luint_t qu,
    const luint_t * new_qu,
    const luint_t new_size)
{
    if(qu >= m_num_qu)
        return;

    if(new_size > queue_capacity(qu))
        return;

    m_size -= queue_size(qu);

    /* copy all elements to queue, then change pos */
    std::copy(new_qu, new_qu + new_size, m_qu_start[qu]);
    m_qu_pos[qu] = new_size;

    m_size += new_size;
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

    m_size = 0;
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
    luint_t k = 0;
    for(const luint_t c : coloring)
        k = std::max(k, c);
    m_num_qu = k + 1;

    /* count capacity per queue */
    std::vector<luint_t> qu_size(m_num_qu, 0);
    for(const luint_t& c : coloring)
        ++qu_size[c];

    /* compute queue offsets */
    m_qu_start.resize(m_num_qu + 1);

    luint_t offset = 0;
    for(luint_t i = 0; i < m_num_qu; ++i)
    {
        m_qu_start[i] = m_data.data() + offset;
        offset += qu_size[i];
    }
    m_qu_start[m_num_qu] = m_data.data() + offset;

    /* initailize queue pos counters to zero */
    m_qu_pos.resize(m_num_qu);
    reset();
}

/* ************************************************************************** */

template<typename COSTTYPE>
luint_t
ColoredQueueBunch<COSTTYPE>::
size()
{
    return m_size;
}

/**
 * *****************************************************************************
 * *************************** LockFreeTreeSampler *****************************
 * *****************************************************************************
 */

template<typename COSTTYPE, bool ACYCLIC>
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
LockFreeTreeSampler(
    Graph<COSTTYPE> * graph)
: LockFreeTreeSampler<COSTTYPE, ACYCLIC>(graph, false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
LockFreeTreeSampler(
    Graph<COSTTYPE> * graph,
    const bool deterministic,
    const uint_t initial_seed)
: TreeSampler<COSTTYPE, ACYCLIC>(graph, deterministic, initial_seed),
  m_qu(graph, deterministic)
{
    this->build_component_lists();
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
    const uint_t seed = this->m_deterministic ?
        this->m_initial_seed : this->m_rnd_dev();
    std::mt19937 rnd(seed);
    roots.clear();

    /* execute selection procedure per component */
    const luint_t num_nodes = this->m_graph->num_nodes();
    const luint_t num_components = this->m_component_lists.size();
    const luint_t num_colors = m_qu.num_queues();
    const luint_t * coloring = this->m_graph->get_coloring().data();
    std::vector<luint_t> feasible_colors(num_colors, 0);

    for(luint_t c = 0; c < num_components; ++c)
    {
        /* determine k proportional to component size */
        const luint_t c_k =
            std::min(num_nodes, std::max((luint_t) 1, (luint_t) std::floor(k *
                this->m_component_lists[c].size() / num_nodes)));

        /* select a color at random with >= c_k nodes */
	feasible_colors.assign(num_colors, 0);

        /* count nodes per color */
        for(const luint_t n : this->m_component_lists[c])
            ++feasible_colors[coloring[n]];

        /* select a maximum color */
        luint_t root_c = 0;
        luint_t root_c_size = 0;
        for(luint_t i = 0; i < num_colors; ++i)
        {
            if(feasible_colors[i] > root_c_size)
            {
                root_c = i;
                root_c_size = feasible_colors[i];
            }
        }

        /* gather all nodes of the chosen color */
        std::vector<luint_t> feasible_roots;
        feasible_roots.reserve(m_qu.queue_capacity(root_c));

        for(const luint_t n : this->m_component_lists[c])
            if(coloring[n] == root_c)
                feasible_roots.push_back(n);

        /* select nodes until happy */
        for(luint_t i = 0; i < std::min(c_k, (luint_t) feasible_roots.size()); ++i)
        {
            /* select next root */
            std::uniform_int_distribution<luint_t> rd(0,
                feasible_roots.size() - 1);

            const luint_t chosen_ix = rd(rnd);
            roots.push_back(feasible_roots[chosen_ix]);

            /* remove root from candidate list */
            feasible_roots[chosen_ix] = feasible_roots.back();
            feasible_roots.pop_back();
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
std::unique_ptr<Tree<COSTTYPE>>
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
sample(
    std::vector<luint_t>& roots,
    bool record_dependencies,
    bool relax)
{
    const luint_t n = this->m_graph->num_nodes();
    const luint_t m = this->m_graph->edges().size();

    m_rem_nodes = n;
    m_tree = std::unique_ptr<Tree<COSTTYPE>>(new Tree<COSTTYPE>(n, 2 * m));

    /* initialize output queue (size: maximal color) */
    luint_t out_size = 0;
    for(luint_t i = 0; i < m_qu.num_queues(); ++i)
        out_size = std::max(out_size, m_qu.queue_capacity(i));
    m_qu_out.resize(out_size);
    m_qu_out_pos = 0;

    /* initialize queue for nodes */
    m_queued.resize(n);
    std::fill(m_queued.begin(), m_queued.end(), 0);

    /* copy original degrees and initialize locks */
    m_rem_degrees.resize(n);
    for(luint_t i = 0; i < n; ++i)
        m_rem_degrees[i] = this->m_graph->nodes()[i].incident_edges.size();

    /* initialize markers */
    m_markers.resize(n);
    std::fill(m_markers.begin(), m_markers.end(), 0);

    /* initialize new-queue */
    m_new.resize(n);
    m_new_size = 0;

    /* handle roots */
    std::copy(roots.begin(), roots.end(), m_new.begin());
    m_new_size = roots.size();
    for(const luint_t r : roots)
    {
        m_tree->raw_parent_ids()[r] = r;
        m_markers[r] = 2u;
    }
    m_rem_nodes -= roots.size();

    /* roots constitute the first iterations' phase I */
    bool skip_ph1 = true;
    m_cur_col = this->m_graph->get_coloring()[roots[0]];

    m_it = 0;
    bool first = true;
    while(m_rem_nodes > 0)
    {
        /* relax option: do not enforce maximality */
        if(relax && !first && m_qu.size() == 0)
            break;

        /* add new nodes if viable */
        if(!skip_ph1)
        {
            /* remove old 'new' nodes */
            m_new_size = 0;
            m_qu_out_pos = 0;

            sample_phase_I();
        }

        /* process neighboring nodes and increment markers */
        sample_phase_II();
        skip_ph1 = false;

        /* replace color in input queue */
        m_qu.replace_queue(m_cur_col, m_qu_out.data(), m_qu_out_pos);

        /* no nodes that could be included? Look for new roots */
        if(m_qu.size() == 0 && !relax)
        {
            sample_rescue();
            skip_ph1 = true;
        }
        first = false;

        ++m_it;
    }

    /* gather children and finalize tree */
    m_tree->finalize(record_dependencies, this->m_graph);

    return std::move(m_tree);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
sample_phase_I()
{
    const uint_t seed = this->m_deterministic ? (this->m_initial_seed + m_it) :
        this->m_rnd_dev();
    std::mt19937 rnd_qu(seed);
    std::vector<luint_t> feas_queue;
    std::vector<luint_t> feas_queue_size;

    /* select next color (rule: random nonempty queue) */
    luint_t max_qu_size = 0;
    luint_t * max_qu = nullptr;
    for(luint_t i = 0; i < m_qu.num_queues(); ++i)
    {
        if(m_qu.queue_size(i) > 0)
        {
            feas_queue.push_back(i);
            feas_queue_size.push_back(m_qu.queue_size(i));
        }
    }
    std::discrete_distribution<luint_t> d(feas_queue_size.begin(),
        feas_queue_size.end());
    m_cur_col = feas_queue[d(rnd_qu)];
    max_qu_size = m_qu.queue_size(m_cur_col);
    max_qu = m_qu.queue(m_cur_col);

    /* iterate over nodes in queue */
    tbb::blocked_range<luint_t> qu_range(0, max_qu_size);
    const auto fn_phaseI = [&](const tbb::blocked_range<luint_t>& r)
    {
        std::mt19937 rnd_gen(seed + 1 + r.begin());
        std::unique_ptr<luint_t[]> buf =
            std::unique_ptr<luint_t[]>(new luint_t[m_buf_edges]);
        luint_t num_buf = 0;

        for(luint_t ix = r.begin(); ix != r.end(); ++ix)
        {
            const luint_t q_node = max_qu[ix];
            num_buf = 0;

            /* ACYCLIC -> stop if marker >= 2 */
            if(!ACYCLIC || m_markers[q_node] < 2)
            {
                /* scan incident edges */
                const std::vector<luint_t>& q_inc =
                    this->m_graph->inc_edges(q_node);
                for(const luint_t e_ix : q_inc)
                {
                    const GraphEdge<COSTTYPE> e =
                        this->m_graph->edges()[e_ix];

                    /* extract corresponding adjacent node */
                    const luint_t o_node = (e.node_a == q_node ?
                        e.node_b : e.node_a);

                    /* save all nodes in tree (potential parents) */
                    if(m_tree->raw_parent_ids()[o_node] != invalid_luint_t
                        && num_buf < m_buf_edges)
                        buf.get()[num_buf++] = e_ix;
                }

                /* select edge at random */
                std::uniform_int_distribution<luint_t> d(0, num_buf - 1);
                const luint_t o_id = buf.get()[d(rnd_gen)];

                /* extract corresponding adjacent node */
                const GraphEdge<COSTTYPE> e = this->m_graph->edges()[o_id];
                const luint_t o_node = (e.node_a == q_node ?
                    e.node_b : e.node_a);

                /* select that node as parent */
                m_tree->raw_parent_ids()[q_node] = o_node;
                m_tree->raw_to_parent_edge_ids()[q_node] = o_id;

                /* allows for shorthand check for 'in tree' */
                m_markers[q_node] = 2u;

                /* save as 'new' node */
                m_new[m_new_size++] = q_node;

                --m_rem_nodes;
            }
        }
    };

    /* use function serially if determinism turned on */
    if(this->m_deterministic)
        fn_phaseI(qu_range);
    else
        tbb::parallel_for(qu_range, fn_phaseI);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
sample_phase_II()
{
    /* edit markers for neighbors of newly added nodes */
    tbb::blocked_range<luint_t> new_range(0, m_new_size);
    const auto fn_phaseII = [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t ix = r.begin(); ix != r.end(); ++ix)
        {
            const luint_t n_new = m_new[ix];

            /* increment marker for neighboring nodes */
            for(const luint_t e_id : this->m_graph->inc_edges(n_new))
            {
                const GraphEdge<COSTTYPE>& e = this->m_graph->edges()[e_id];
                const luint_t o_node = (e.node_a == n_new) ? e.node_b :
                    e.node_a;

                /* update marker, if now == 2, stop considering node */
                if(ACYCLIC)
                    if(m_markers[o_node].fetch_and_increment() == 1)
                        --m_rem_nodes;

                /* marker < 2 && not queued -> put in queue */
                if(m_markers[o_node] < 2 &&
                    m_queued[o_node].compare_and_swap(1u, 0u) == 0u)
                {
                    const luint_t o_color =
                        this->m_graph->get_coloring()[o_node];

                    if(o_color == m_cur_col)
                        m_qu_out[m_qu_out_pos.fetch_and_increment()] =
                            o_node;
                    else
                        m_qu.push_to(o_color, o_node);
                }
            }
        }
    };

    /* use function serially if determinism turned on */
    if(this->m_deterministic)
        fn_phaseII(new_range);
    else
        tbb::parallel_for(new_range, fn_phaseII);
}

/* ************************************************************************** */

template<typename COSTTYPE, bool ACYCLIC>
void
LockFreeTreeSampler<COSTTYPE, ACYCLIC>::
sample_rescue()
{
    /* find nodes with marker 0 of the same color */
    tbb::atomic<luint_t> rescue_color = invalid_luint_t;
    m_new_size = 0;

    tbb::blocked_range<luint_t> node_range(0, this->m_graph->num_nodes());
    const auto fn_rescue = [&](const tbb::blocked_range<luint_t>& r)
    {
        for(luint_t i = r.begin(); i < r.end(); ++i)
        {
            const bool is_in_tree = (m_tree->raw_parent_ids()
                [i] != invalid_luint_t);

            if(!is_in_tree && m_markers[i] == 0)
            {
                const luint_t my_color = this->m_graph->get_coloring()[i];
                rescue_color.compare_and_swap(my_color, invalid_luint_t);

                if(my_color == rescue_color && m_new_size < m_max_rescue)
                {
                    m_tree->raw_parent_ids()[i] = i;

                    m_new[m_new_size++] = i;
                    m_markers[i] = 2u;
                    --m_rem_nodes;
                }
            }
        }
    };

    /* use function serially if determinism turned on */
    if(this->m_deterministic)
        fn_rescue(node_range);
    else
        tbb::parallel_for(node_range, fn_rescue);
}

NS_MAPMAP_END
