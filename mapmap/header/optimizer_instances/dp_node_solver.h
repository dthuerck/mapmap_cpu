/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_DP_NODE_SOLVER_H_
#define __MAPMAP_HEADER_DP_NODE_SOLVER_H_

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/optimizer_instances/dp_node.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class DPNodeSolver
{
public:
    virtual ~DPNodeSolver() {};

    virtual void optimize_node() = 0;
    virtual luint_t scratch_bytes_needed() = 0;
};

template<typename COSTTYPE, uint_t SIMDWIDTH>
using DPNodeSolver_ptr = std::unique_ptr<DPNodeSolver<COSTTYPE, SIMDWIDTH>>;

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
class GeneralDPNodeSolver :
    public DPNodeSolver<COSTTYPE, SIMDWIDTH>
{
public:
    GeneralDPNodeSolver(
        DPNode<COSTTYPE, SIMDWIDTH> * node);
    virtual ~GeneralDPNodeSolver();

    virtual void optimize_node();
    virtual luint_t scratch_bytes_needed();

protected:
    _v_t<COSTTYPE, SIMDWIDTH> get_independent_of_parent_costs(
        const _iv_st<COSTTYPE, SIMDWIDTH> l_i);

protected:
    DPNode<COSTTYPE, SIMDWIDTH> * m_node;

    _s_t<COSTTYPE, SIMDWIDTH> * m_icost_table;
};

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
class SubmodularDPNodeSolver :
    public GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>
{
public:
    SubmodularDPNodeSolver(
        DPNode<COSTTYPE, SIMDWIDTH> * node,
        PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>& env);
    virtual ~SubmodularDPNodeSolver();

    virtual void optimize_node();
    virtual luint_t scratch_bytes_needed();

protected:
    virtual void fill_icost_cache();
    void compute_mp();
    void label_union();

protected:
    PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE> m_env;

    _iv_st<COSTTYPE, SIMDWIDTH> * m_label_union;
    _iv_st<COSTTYPE, SIMDWIDTH> m_label_union_size;

    _s_t<COSTTYPE, SIMDWIDTH> * m_mprime;
    _iv_st<COSTTYPE, SIMDWIDTH> * m_mprime_ix;
    _s_t<COSTTYPE, SIMDWIDTH> * m_icost_cache;

    _s_t<COSTTYPE, SIMDWIDTH> m_min_fp_cost;
    _iv_st<COSTTYPE, SIMDWIDTH> m_min_fp;
};

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
class SupermodularDPNodeSolver :
    public SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>
{
public:
    SupermodularDPNodeSolver(
        DPNode<COSTTYPE, SIMDWIDTH> * node,
        PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>& env);
    virtual ~SupermodularDPNodeSolver();

    virtual void optimize_node();
    virtual luint_t scratch_bytes_needed();

protected:
    void fill_icost_cache();
};

NS_MAPMAP_END

/* include function implementations */
#include "source/optimizer_instances/dp_node_solver.impl.h"

#endif /* __MAPMAP_HEADER_DP_NODE_SOLVER_H_ */