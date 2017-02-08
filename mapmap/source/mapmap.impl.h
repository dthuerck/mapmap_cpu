/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/mapmap.h"

#include <iostream>

#include "header/multilevel_instances/group_same_label.h"
#include "header/termination_instances/stop_when_flat.h"
#include "header/dynamic_programming.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * ********************************* PUBLIC ************************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
mapMAP()
: m_construct_graph(false),
  m_set_graph(false),
  m_set_label_set(false),
  m_set_unaries(false),
  m_set_pairwise(false),
  m_set_multilevel_criterion(false),
  m_set_termination_criterion(false),
  m_hist_energy(),
  m_hist_acyclic_iterations(0),
  m_hist_spanningtree_iterations(0),
  m_hist_multilevel_iterations(0)
{
    
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
mapMAP(
    const luint_t num_nodes,
    const luint_t num_labels)
: m_construct_graph(true),
  m_set_graph(false),
  m_set_label_set(false),
  m_set_unaries(false),
  m_set_pairwise(false),
  m_set_multilevel_criterion(false),
  m_set_termination_criterion(false),
  m_num_nodes(num_nodes),
  m_num_labels(num_labels),
  m_graph(new Graph<COSTTYPE>),
  m_label_set(new LabelSet<COSTTYPE, SIMDWIDTH>(num_nodes, true)),
  m_hist_energy(),
  m_hist_acyclic_iterations(0),
  m_hist_spanningtree_iterations(0),
  m_hist_multilevel_iterations(0)
{
    
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~mapMAP()
{
    
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_graph(
    const Graph<COSTTYPE> * graph)
throw()
{
    if(m_construct_graph)
        throw std::runtime_error("Setting the graph is not allowed "
            "in construction mode.");

    m_graph = graph;
    m_set_graph = true;
    m_num_nodes = graph->num_nodes();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_label_set(
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
throw()
{
    if(m_construct_graph)
        throw std::runtime_error("Setting the label set is not allowed "
            "in construction mode.");

    m_label_set = label_set;
    m_set_label_set = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
add_edge(
    const luint_t node_a,
    const luint_t node_b,
    const _s_t<COSTTYPE, SIMDWIDTH> weight)
throw()
{   
    if(!m_construct_graph)
        throw std::runtime_error("Adding edges is only allowed "
            "in construction mode.");

    m_graph->add_edge(node_a, node_b, weight);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_node_label_set(
    const luint_t node_id,
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& label_set)
throw()
{
    if(!m_construct_graph)
        throw std::runtime_error("Setting a label set is only allowed "
            "in construction mode.");

    m_label_set->set_label_set_for_node(node_id, label_set);
    m_label_set_check.insert(node_id);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_unaries(
    const UNARY * unaries)
{
    m_unaries = unaries;
    m_set_unaries = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_pairwise(
    const PAIRWISE * pairwise)
{
    m_pairwise = pairwise;
    m_set_pairwise = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_multilevel_criterion(
    MultilevelCriterion<COSTTYPE, SIMDWIDTH> * criterion)
{
    m_multilevel_criterion = criterion;
    m_set_multilevel_criterion = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
set_termination_criterion(
    TerminationCriterion<COSTTYPE, SIMDWIDTH> * criterion)
{
    m_termination_criterion = criterion;

    m_set_termination_criterion = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
optimize(
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
throw()
{
    _s_t<COSTTYPE, SIMDWIDTH> obj;

    /* create std modules for uninitialized fields */
    create_std_modules();

    /* check inputs for completeness and various sanity checks */
    if(!check_data_complete())
        throw std::runtime_error("Input data for optimization "
            "incomplete or not sane.");

    /* report on starting the optimization process */
     std::cout << "[mapMAP] "
              << UNIX_COLOR_GREEN
              << "Starting optimization..."
              << UNIX_COLOR_RESET
              << std::endl;

    /* start timer */
    m_time_start = std::chrono::system_clock::now();

    /* initialize current solution */
    const luint_t num_nodes = m_graph->num_nodes();
    m_solution.resize(num_nodes);
    std::fill(m_solution.begin(), m_solution.end(), 0);

    /* find initial solution by tree-optimization w/o dependencies */
    m_objective = initial_labelling();
    record_time_from_start();
    print_status();

    /* check for termination */
    if(check_termination())
    {
        solution.clear();
        solution.assign(m_solution.begin(), m_solution.end());

        return m_objective;
    }

    /* rapid initial descent by multilevel */
    m_objective = opt_step_multilevel();
    record_time_from_start();
    print_status();

    /* take spanning tree steps until no more improvements occur */
    luint_t sp_it = 0;

    _s_t<COSTTYPE, SIMDWIDTH> old_objective = m_objective;
    while(true)
    {
        ++sp_it;

        /* check if algorithms needs to terminate */
        if(check_termination())
        {
            solution.clear();
            solution.assign(m_solution.begin(), m_solution.end());

            return m_objective;
        }

        /* execute spannign tree step */
        obj = opt_step_spanning_tree();
        record_time_from_start();

        if(obj < old_objective)
            old_objective = obj;
        else
            break;

        print_status();

        if(sp_it % 5 == 0)
        {
            m_objective = opt_step_multilevel();
            record_time_from_start();
            print_status();
        }
    }

    /* lastly, execute acyclic steps until termination */
    while(!check_termination())
    {
        m_objective = opt_step_acyclic();

        record_time_from_start();
        print_status();
    }

    /* output solution */
    solution.clear();
    solution.assign(m_solution.begin(), m_solution.end());

    return m_objective;
}

/**
 * *****************************************************************************
 * ******************************** PROTECTED **********************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
create_std_modules()
{
    /* std module - group nodes that have the same label */
    if(!m_set_multilevel_criterion)
    {
        m_storage_multilevel_criterion = std::unique_ptr<MultilevelCriterion<
            COSTTYPE, SIMDWIDTH>>(new GroupSameLabel<COSTTYPE, SIMDWIDTH>);
        m_multilevel_criterion = m_storage_multilevel_criterion.get();
        m_set_multilevel_criterion = true;
    }

    /* std module - terminate after 5 flat iterations in a row */
    if(!m_set_termination_criterion)
    {
        m_storage_termination_criterion = std::unique_ptr<TerminationCriterion<
            COSTTYPE, SIMDWIDTH>>(
            new StopWhenFlat<COSTTYPE, SIMDWIDTH>(5, true));
        m_termination_criterion = m_storage_termination_criterion.get();
        m_set_termination_criterion = true;
    }

    /* create a multilevel module for the current graph */
    m_multilevel = std::unique_ptr<Multilevel<COSTTYPE, SIMDWIDTH, 
        UNARY, PAIRWISE>>(new Multilevel<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>(
        m_graph, m_label_set, m_unaries, m_pairwise, m_multilevel_criterion));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
bool
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
check_data_complete()
{
    if(!m_construct_graph && !m_set_graph && !m_set_label_set)
        return false;

    if(!m_set_unaries)
        return false;

    if(!m_set_pairwise)
        return false;

    if(!m_set_multilevel_criterion)
        return false;

    if(!m_set_termination_criterion)
        return false;

    if(m_construct_graph && m_label_set_check.size() != m_num_nodes)
        return false;

    if(m_construct_graph && (m_label_set->max_label() > 
        (_iv_st<COSTTYPE, SIMDWIDTH>) m_num_labels))
        return false;

    return true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
record_time_from_start()
{
    m_hist_time.push_back(std::chrono::duration_cast<
        std::chrono::milliseconds>(
        std::chrono::system_clock::now() - m_time_start).count());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
print_status()
{
    std::cout << "[mapMAP] "
              << UNIX_COLOR_RED
              << m_hist_time.back() << " ms"
              << UNIX_COLOR_RESET
              << ", "
              << UNIX_COLOR_GREEN
              << "Objective "
              << m_objective
              << UNIX_COLOR_RESET
              << " (after "
              << m_hist_multilevel_iterations
              << " multilevel, "
              << m_hist_spanningtree_iterations
              << " spanning tree, "
              << m_hist_acyclic_iterations
              << " acyclic iterations)"
              << std::endl;
}
    

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH> 
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
initial_labelling()
{
    /* sample a tree (forest) without dependencies */
    TreeSampler<COSTTYPE, false> sampler(m_graph);

    /* sample roots of forest */
    std::vector<luint_t> roots;
    sampler.select_random_roots(m_num_roots, roots);

    /* grow trees in forest */
    std::unique_ptr<Tree<COSTTYPE>> tree = sampler.sample(roots, false);

    /* create tree optimizer (std: DP) and pass parameters and modules */
    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE> opt;
    opt.set_graph(m_graph);
    opt.set_tree(tree.get());
    opt.set_label_set(m_label_set);
    opt.set_costs(m_unaries, m_pairwise);

    /* optimize! */
    opt.optimize(m_solution);
    m_objective = opt.objective(m_solution);
    m_hist_energy.push_back(m_objective);

    return m_objective;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>:: 
opt_step_spanning_tree()
{
    /* sample a tree (forest) without dependencies */
    TreeSampler<COSTTYPE, false> sampler(m_graph);

    /* sample roots of forest */
    std::vector<luint_t> roots;
    sampler.select_random_roots(m_num_roots, roots);

    /* grow trees in forest */
    std::unique_ptr<Tree<COSTTYPE>> tree = sampler.sample(roots, true);

    /* create tree optimizer (std: DP) and pass parameters and modules */
    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE> opt;
    opt.set_graph(m_graph);
    opt.set_tree(tree.get());
    opt.set_label_set(m_label_set);
    opt.set_costs(m_unaries, m_pairwise);
    opt.use_dependencies(m_solution);

    /* optimize! */
    opt.optimize(m_solution);
    m_objective = opt.objective(m_solution);
    ++m_hist_spanningtree_iterations;
    m_hist_mode.push_back(SolverMode::SOLVER_SPANNINGTREE);
    m_hist_energy.push_back(m_objective);

    return m_objective;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>:: 
opt_step_multilevel()
{
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lvl_solution;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> upper_solution;
    lvl_solution.assign(m_solution.begin(), m_solution.end());

    /* tree optimizer needed for calculating the objective */
    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY,
        PAIRWISE> org_opt;
    org_opt.set_graph(m_graph);
    org_opt.set_label_set(m_label_set);
    org_opt.set_costs(m_unaries, m_pairwise);
    org_opt.use_dependencies(m_solution);

    /* optimize level-up until no improvement happens on the reprojection */
    std::vector<luint_t> roots;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> org_solution;
    org_solution.assign(m_solution.begin(), m_solution.end());

    /* record iteration */
    ++m_hist_multilevel_iterations;
    m_hist_mode.push_back(SolverMode::SOLVER_MULTILEVEL);

    while(true)
    {
        if(!m_multilevel->next_level(lvl_solution, upper_solution))
            break;

        /* retrieve this level's MRF problem */
        const Graph<COSTTYPE> * lvl_graph = m_multilevel->get_level_graph();
        const LabelSet<COSTTYPE, SIMDWIDTH> * lvl_label_set = m_multilevel->
            get_level_label_set();
        const UnaryTable<COSTTYPE, SIMDWIDTH> * lvl_unaries = 
            m_multilevel->get_level_unaries();
        const PairwiseTable<COSTTYPE, SIMDWIDTH> * lvl_pairwise = 
            m_multilevel->get_level_pairwise();

        /* create new optimizer for level graph */
        CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, 
            UnaryTable<COSTTYPE, SIMDWIDTH>, PairwiseTable<COSTTYPE, SIMDWIDTH>> 
            lvl_opt;
        lvl_opt.set_graph(lvl_graph);
        lvl_opt.set_label_set(lvl_label_set);
        lvl_opt.set_costs(lvl_unaries, lvl_pairwise);
        lvl_opt.use_dependencies(upper_solution);

        /* sample tree on graph */
        TreeSampler<COSTTYPE, true> sampler(m_multilevel->get_level_graph());

        roots.clear();
        sampler.select_random_roots(m_num_roots, roots);
        std::unique_ptr<Tree<COSTTYPE>> lvl_tree = 
            sampler.sample(roots, true);
        lvl_opt.set_tree(lvl_tree.get());
        
        /* optimize for level solution */
        lvl_opt.optimize(upper_solution);

        /* reproject solution to original and compute objective */
        m_multilevel->reproject_solution(upper_solution, org_solution);
        const _s_t<COSTTYPE, SIMDWIDTH> new_obj = org_opt.objective(
            org_solution);

        /* check if the solution was an improvement and save if necessary */
        if(new_obj < m_objective)
        {
            m_solution = org_solution;
            m_objective = new_obj;

            lvl_solution = upper_solution;
        }
        else
        {
            break;
        }
    }

    /* delete all levels */
    while(m_multilevel->prev_level());

    m_hist_energy.push_back(m_objective);

    return m_objective;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH> 
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
opt_step_acyclic()
{
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ac_solution = m_solution;

    /* sample a tree (forest) without dependencies */
    TreeSampler<COSTTYPE, true> sampler(m_graph);

    /* sample roots of forest */
    std::vector<luint_t> roots;
    sampler.select_random_roots(m_num_roots, roots);

    /* grow trees in forest */
    std::unique_ptr<Tree<COSTTYPE>> tree = sampler.sample(roots, true);

    /* create tree optimizer (std: DP) and pass parameters and modules */
    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH, UNARY,
        PAIRWISE> opt;
    opt.set_graph(m_graph);
    opt.set_tree(tree.get());
    opt.set_label_set(m_label_set);
    opt.set_costs(m_unaries, m_pairwise);
    opt.use_dependencies(ac_solution);

    /* optimize! */
    opt.optimize(ac_solution);
    ++m_hist_acyclic_iterations;
    m_hist_mode.push_back(SolverMode::SOLVER_ACYCLIC);

    const _s_t<COSTTYPE, SIMDWIDTH> ac_opt = opt.objective(ac_solution); 
    if(ac_opt < m_objective)
    {
        m_objective = ac_opt;
        m_solution = ac_solution;
    }

    m_hist_energy.push_back(m_objective);

    return m_objective;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
bool
mapMAP<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
check_termination()
{
    /* collect information for termination criterion */
    SolverHistory<COSTTYPE, SIMDWIDTH> term_info;

    term_info.energy_history = &m_hist_energy;
    term_info.time_history = &m_hist_time;
    term_info.mode_history = &m_hist_mode;
    term_info.acyclic_iterations = m_hist_acyclic_iterations;
    term_info.spanningtree_iterations = m_hist_spanningtree_iterations;
    term_info.multilevel_iterations = m_hist_multilevel_iterations;

    return m_termination_criterion->check_termination(&term_info);
}

NS_MAPMAP_END
