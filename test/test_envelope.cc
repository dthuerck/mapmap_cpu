/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include "header/defines.h"
#include "header/costs.h"
#include "header/graph.h"

#include "header/optimizer_instances/dynamic_programming.h"
#include "header/optimizer_instances/dp_node_solver_factory.h"
#include "header/cost_instances/pairwise_potts.h"
#include "header/cost_instances/pairwise_antipotts.h"
#include "header/cost_instances/pairwise_linear_peak.h"
#include "header/cost_instances/pairwise_truncated_linear.h"
#include "header/cost_instances/pairwise_truncated_quadratic.h"
#include "header/cost_instances/pairwise_table.h"
#include "header/cost_instances/unary_table.h"

#include <random>

NS_MAPMAP_BEGIN

template<typename T, uint_t N, typename P>
class TestTuple {
public:
    typedef T Type;
    static const uint_t Value = N;
    typedef P Unit;
};

template<typename T>
class mapMAPTestEnvelope : public ::testing::Test
{
public:
    /* rename template parameters for convenience */
    typedef typename T::Type COSTTYPE;
    static const uint_t SIMDWIDTH = T::Value;
    typedef typename T::Unit PAIRWISE;

    const COSTTYPE lbl_prob = 0.35;
    const int num_labels = 101;
    const COSTTYPE max_val = 10;

    const COSTTYPE e_weight = 1.314;
    const COSTTYPE c = 2.7;
    const COSTTYPE d = 3.8;

public:
    mapMAPTestEnvelope()
    {

    }

    ~mapMAPTestEnvelope()
    {

    }

    void
    SetUp()
    {
        const luint_t seed = m_rdev();
        m_rnd = std::mt19937(seed);

        m_d_prob = std::uniform_real_distribution<COSTTYPE>(0, 1);
        m_d_val = std::uniform_real_distribution<COSTTYPE>(0, max_val);

        create_graph();
        create_label_set();
        create_cost_bundle();
        create_unaries();
        create_task_data();
        fill_data();
    }

    void
    TearDown()
    {

    }

    void
    create_graph()
    {
        m_graph = std::unique_ptr<Graph<COSTTYPE>>(new Graph<COSTTYPE>(2));
        m_graph->add_edge(0, 1, e_weight);
    }

    void
    create_label_set()
    {
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl0, lbl1;

        for(int i = 0; i < num_labels; ++i)
        {
            if(m_d_prob(m_rnd) <= lbl_prob)
                lbl0.push_back(i);
            if(m_d_prob(m_rnd) <= lbl_prob)
                lbl1.push_back(i);
        }

        m_label_set = std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>>(
            new LabelSet<COSTTYPE, SIMDWIDTH>(m_graph->num_nodes(), true));
        m_label_set->set_label_set_for_node(0, lbl0);
        m_label_set->set_label_set_for_node(1, lbl1);
    }

    void
    create_cost_bundle()
    {
        m_cbundle = std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>>(new
            CostBundle<COSTTYPE, SIMDWIDTH>(m_graph.get()));
    }

    void
    create_unaries()
    {
        m_unaries.reserve(2);

        std::vector<COSTTYPE> costs0(m_label_set->label_set_size(0));
        for(luint_t i = 0; i < costs0.size(); ++i)
            costs0[i] = m_d_val(m_rnd);
        m_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
            SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(0,
            m_label_set.get())));
        m_unaries.back()->set_costs(costs0);
        m_cbundle->set_unary_costs(0, m_unaries.back().get());

        std::vector<COSTTYPE> costs1(m_label_set->label_set_size(1));
        for(luint_t i = 0; i < costs1.size(); ++i)
            costs1[i] = m_d_val(m_rnd);
        m_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
            SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(1,
            m_label_set.get())));
        m_unaries.back()->set_costs(costs1);
        m_cbundle->set_unary_costs(1, m_unaries.back().get());
    }

    void
    create_task_data()
    {
        m_c_node.is_in_tree = true;

        m_c_node.parent_id = 0;
        m_c_node.to_parent_edge_id = 0;
        m_c_node.to_parent_weight = e_weight;

        m_c_node.node_id = 1;
        m_c_node.degree = 0;
        m_c_node.dependency_degree = 0;
    }

    void
    fill_data()
    {
        m_pairwise = std::unique_ptr<PAIRWISE>(new PAIRWISE({c, d}));
        m_cbundle->set_pairwise_costs(m_pairwise.get());

        const luint_t num_parent_labels = m_label_set->label_set_size(0);
        const luint_t num_parent_labels_padded = DIV_UP(num_parent_labels,
            SIMDWIDTH) * SIMDWIDTH;

        m_gp_cost.resize(num_parent_labels_padded);
        m_gp_lbl.resize(num_parent_labels_padded);

        m_sp_cost.resize(num_parent_labels_padded);
        m_sp_lbl.resize(num_parent_labels_padded);

        /* create optimization task and allocate scratch space */
        DPNode<COSTTYPE, SIMDWIDTH> node;
        node.c_node = m_c_node;
        node.c_graph = m_graph.get();
        node.c_labels = m_label_set.get();
        node.c_unary = m_cbundle->get_unary_costs(1);
        node.c_pairwise = m_cbundle->get_pairwise_costs(0);
        node.respect_dependencies = false;

        /* solve with general-purpose solver */
        DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH> gp_solver =
            DPNodeSolverFactory<COSTTYPE, SIMDWIDTH>::
            get_enumerative_solver(&node);

        const luint_t gp_scratch_needed =
            gp_solver->scratch_bytes_needed();
        const luint_t gp_scratch_padded = DIV_UP(gp_scratch_needed,
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>)) *
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>);
        std::vector<char> scratch(gp_scratch_padded);
        node.c_scratch = (_s_t<COSTTYPE, SIMDWIDTH> *) scratch.data();

        node.c_opt_values = m_gp_cost.data();
        node.c_opt_labels = m_gp_lbl.data();
        gp_solver->optimize_node();

        /* solve with specific solver */
        DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH> sp_solver =
        DPNodeSolverFactory<COSTTYPE, SIMDWIDTH>::
            get_solver(&node);

        const luint_t sp_scratch_needed =
            sp_solver->scratch_bytes_needed();
        const luint_t sp_scratch_padded = DIV_UP(sp_scratch_needed,
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>)) *
            sizeof(_s_t<COSTTYPE, SIMDWIDTH>);
        scratch.resize(sp_scratch_padded);
        node.c_scratch = (_s_t<COSTTYPE, SIMDWIDTH> *) scratch.data();

        node.c_opt_values = m_sp_cost.data();
        node.c_opt_labels = m_sp_lbl.data();
        sp_solver->optimize_node();
    }

protected:
    std::unique_ptr<Graph<COSTTYPE>> m_graph;
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> m_label_set;
    std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>> m_cbundle;
    std::vector<std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>>> m_unaries;
    std::unique_ptr<PAIRWISE> m_pairwise;
    TreeNode<COSTTYPE> m_c_node;

    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_gp_cost;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_gp_lbl;

    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_sp_cost;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_sp_lbl;

    /* create dice */
    std::random_device m_rdev;
    std::mt19937 m_rnd;

    std::uniform_real_distribution<COSTTYPE> m_d_prob;
    std::uniform_real_distribution<COSTTYPE> m_d_val;
};
TYPED_TEST_CASE_P(mapMAPTestEnvelope);

TYPED_TEST_P(mapMAPTestEnvelope, TestCompareLeafOpt)
{
    const luint_t num_parent_labels = this->m_label_set->label_set_size(0);

    /* compare results */
    for(luint_t i = 0; i < num_parent_labels; ++i)
        ASSERT_NEAR(this->m_sp_cost[i], this->m_gp_cost[i], 0.01);
}

/* register test cases */
REGISTER_TYPED_TEST_CASE_P(mapMAPTestEnvelope,
    TestCompareLeafOpt);

/* instantiate tests */
typedef ::testing::Types<
    #if defined(__SSE4_2__)
    TestTuple<float, 4, PairwisePotts<float, 4>>,
    TestTuple<float, 4, PairwiseTruncatedLinear<float, 4>>,
    TestTuple<float, 4, PairwiseTruncatedQuadratic<float, 4>>,
    TestTuple<float, 4, PairwiseAntipotts<float, 4>>,
    TestTuple<float, 4, PairwiseLinearPeak<float, 4>>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<float, 8, PairwisePotts<float, 8>>,
    TestTuple<float, 8, PairwiseTruncatedLinear<float, 8>>,
    TestTuple<float, 8, PairwiseTruncatedQuadratic<float, 8>>,
    TestTuple<float, 8, PairwiseAntipotts<float, 8>>,
    TestTuple<float, 8, PairwiseLinearPeak<float, 8>>,
    #endif /* __AVX__ */
    #if defined(__SSE4_2__)
    TestTuple<double, 2, PairwisePotts<double, 2>>,
    TestTuple<double, 2, PairwiseTruncatedLinear<double, 2>>,
    TestTuple<double, 2, PairwiseTruncatedQuadratic<double, 2>>,
    TestTuple<double, 2, PairwiseAntipotts<double, 2>>,
    TestTuple<double, 2, PairwiseLinearPeak<double, 2>>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<double, 4, PairwisePotts<double, 4>>,
    TestTuple<double, 4, PairwiseTruncatedLinear<double, 4>>,
    TestTuple<double, 4, PairwiseTruncatedQuadratic<double, 4>>,
    TestTuple<double, 4, PairwiseAntipotts<double, 4>>,
    TestTuple<double, 4, PairwiseLinearPeak<double, 4>>,
    #endif /* __AVX__ */
    /* Scalar */
    TestTuple<float, 1, PairwisePotts<float, 1>>,
    TestTuple<float, 1, PairwiseTruncatedLinear<float, 1>>,
    TestTuple<float, 1, PairwiseTruncatedQuadratic<float, 1>>,
    TestTuple<float, 1, PairwiseAntipotts<float, 1>>,
    TestTuple<float, 1, PairwiseLinearPeak<float, 1>>,
    TestTuple<double, 1, PairwisePotts<double, 1>>,
    TestTuple<double, 1, PairwiseTruncatedLinear<double, 1>>,
    TestTuple<double, 1, PairwiseTruncatedQuadratic<double, 1>>,
    TestTuple<double, 1, PairwiseAntipotts<double, 1>>,
    TestTuple<double, 1, PairwiseLinearPeak<double, 1>>
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(EnvelopeTest,
    mapMAPTestEnvelope, TestTupleInstances);

NS_MAPMAP_END
