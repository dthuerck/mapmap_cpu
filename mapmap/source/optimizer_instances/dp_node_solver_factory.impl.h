/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/dp_node_solver_factory.h"

#include "header/cost_instances/unary_table.h"

#include "header/cost_instances/pairwise_antipotts.h"
#include "header/optimizer_instances/envelope_instances/pairwise_antipotts_envelope.h"

#include "header/cost_instances/pairwise_linear_peak.h"
#include "header/optimizer_instances/envelope_instances/pairwise_linear_peak_envelope.h"

#include "header/cost_instances/pairwise_potts.h"
#include "header/optimizer_instances/envelope_instances/pairwise_potts_envelope.h"

#include "header/cost_instances/pairwise_truncated_linear.h"
#include "header/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.h"

#include "header/cost_instances/pairwise_truncated_quadratic.h"
#include "header/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>
DPNodeSolverFactory<COSTTYPE, SIMDWIDTH>::
get_enumerative_solver(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{

    /* so far, only UnaryTable costs have been implemented */
    if(typeid(*(node->c_unary)) == typeid(UnaryTable<COSTTYPE, SIMDWIDTH>))
    {
        /* now switch between pairwise cost types */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwisePotts<COSTTYPE, SIMDWIDTH>))
        {
            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
                new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwisePotts<COSTTYPE, SIMDWIDTH>>(node));
        }
        /* ****************************************************************** */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>))
        {
            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
                new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>>(node));
        }
        /* ****************************************************************** */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>))
        {
            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
                new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>>(node));
        }
    }

    /* default case - enumerative, handle costs at runtime */
    return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
        new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
            UnaryCosts<COSTTYPE, SIMDWIDTH>,
            PairwiseCosts<COSTTYPE, SIMDWIDTH>>(node));
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>
DPNodeSolverFactory<COSTTYPE, SIMDWIDTH>::
get_solver(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    /* so far, only UnaryTable costs have been implemented */
    if(typeid(*(node->c_unary)) == typeid(UnaryTable<COSTTYPE, SIMDWIDTH>))
    {
        if(node->c_pairwise == nullptr)
            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
                new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
                    UnaryTable<COSTTYPE, SIMDWIDTH>,
                    PairwiseCosts<COSTTYPE, SIMDWIDTH>>(node));

        /* now switch between pairwise cost types */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwisePotts<COSTTYPE, SIMDWIDTH>))
        {
            const PairwisePotts<COSTTYPE, SIMDWIDTH> * inst_costs =
                (const PairwisePotts<COSTTYPE, SIMDWIDTH> *) node->c_pairwise;

            PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwisePotts<COSTTYPE, SIMDWIDTH>> env_ptr(
                new PairwisePottsEnvelope<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwisePotts<COSTTYPE, SIMDWIDTH>>(inst_costs));

            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(new
                SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwisePotts<COSTTYPE, SIMDWIDTH>>(node, env_ptr));
        }
        /* ****************************************************************** */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>))
        {
            const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH> * inst_costs =
                (const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH> *)
                node->c_pairwise;

            PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>> env_ptr(
                new PairwiseTruncatedLinearEnvelope<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>>(inst_costs));

            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(new
                SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>>(node, env_ptr));
        }
        /* ****************************************************************** */
        if(typeid(*(node->c_pairwise)) ==
            typeid(PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>))
        {
            const PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH> * inst_costs =
                (const PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH> *)
                node->c_pairwise;

            PairwiseCostsEnvelope_ptr<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>> env_ptr(
                new PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>>(inst_costs));

            return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(new
                SubmodularDPNodeSolver<COSTTYPE, SIMDWIDTH,
                UnaryTable<COSTTYPE, SIMDWIDTH>,
                PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>>(node,
                env_ptr));
        }
    }

    /* default case - enumerative, handle costs at runtime */
    return DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH>(
        new GeneralDPNodeSolver<COSTTYPE, SIMDWIDTH,
            UnaryCosts<COSTTYPE, SIMDWIDTH>,
            PairwiseCosts<COSTTYPE, SIMDWIDTH>>(node));
}

NS_MAPMAP_END