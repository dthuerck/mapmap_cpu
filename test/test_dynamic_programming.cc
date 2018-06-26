/**
 * Copyright (C) 2016, Daniel Thuerck
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
#include "header/cost_bundle.h"
#include "header/graph.h"

#include "header/optimizer_instances/dynamic_programming.h"
#include "header/cost_instances/pairwise_antipotts.h"
#include "header/cost_instances/pairwise_linear_peak.h"
#include "header/cost_instances/pairwise_potts.h"
#include "header/cost_instances/pairwise_truncated_linear.h"
#include "header/cost_instances/pairwise_truncated_quadratic.h"
#include "header/cost_instances/pairwise_table.h"
#include "header/cost_instances/unary_table.h"

NS_MAPMAP_BEGIN

template<typename T, uint_t N>
class TestTuple {
public:
    typedef T Type;
    static const uint_t Value = N;
};

template<typename T>
class mapMAPTestDynamicProgramming : public ::testing::Test
{
public:
    /* rename template parameters for convenience */
    typedef typename T::Type COSTTYPE;
    static const uint_t SIMDWIDTH = T::Value;

public:
    mapMAPTestDynamicProgramming()
    {

    }

    ~mapMAPTestDynamicProgramming()
    {

    }

    void
    SetUp()
    {
        /* create input data for test tree-shaped MRF */
        create_graph();
        create_tree();
        create_label_set();
        create_cost_bundle();
        create_unaries();

        /* create data and storage for pairwise cost tests */
        create_pairwise_data();
    }

    void
    TearDown()
    {

    }

protected:
    void
    create_graph()
    {
        /* graph: 4-layered full binary tree */
        m_graph = std::unique_ptr<Graph<COSTTYPE>>(new Graph<COSTTYPE>(15));

        m_graph->add_edge(0, 1, 1.0); /* 0 */
        m_graph->add_edge(0, 2, 1.0); /* 1 */

        m_graph->add_edge(1, 3, 1.0); /* 2 */
        m_graph->add_edge(1, 4, 1.0); /* 3 */
        m_graph->add_edge(2, 5, 1.0); /* 4 */
        m_graph->add_edge(2, 6, 1.0); /* 5 */

        m_graph->add_edge(3, 7, 1.0); /* 6 */
        m_graph->add_edge(3, 8, 1.0); /* 7 */
        m_graph->add_edge(4, 9, 1.0); /* 8 */
        m_graph->add_edge(4, 10, 1.0); /* 9 */
        m_graph->add_edge(5, 11, 1.0); /* 10 */
        m_graph->add_edge(5, 12, 1.0); /* 11 */
        m_graph->add_edge(6, 13, 1.0); /* 12 */
        m_graph->add_edge(6, 14, 1.0); /* 13 */
    }

    void
    create_tree()
    {
         /* graph is a tree, hence just can copy parent relations */
        m_tree = std::unique_ptr<Tree<COSTTYPE>>(new Tree<COSTTYPE>(15, 14));

        /* parent ids */
        m_tree->raw_parent_ids()[0] = 0;
        m_tree->raw_to_parent_edge_ids()[0] = -1;

        m_tree->raw_parent_ids()[1] = 0;
        m_tree->raw_to_parent_edge_ids()[1] = 0;
        m_tree->raw_parent_ids()[2] = 0;
        m_tree->raw_to_parent_edge_ids()[2] = 1;

        m_tree->raw_parent_ids()[3] = 1;
        m_tree->raw_to_parent_edge_ids()[3] = 2;
        m_tree->raw_parent_ids()[4] = 1;
        m_tree->raw_to_parent_edge_ids()[4] = 3;
        m_tree->raw_parent_ids()[5] = 2;
        m_tree->raw_to_parent_edge_ids()[5] = 4;
        m_tree->raw_parent_ids()[6] = 2;
        m_tree->raw_to_parent_edge_ids()[6] = 5;

        m_tree->raw_parent_ids()[7] = 3;
        m_tree->raw_to_parent_edge_ids()[7] = 6;
        m_tree->raw_parent_ids()[8] = 3;
        m_tree->raw_to_parent_edge_ids()[8] = 7;
        m_tree->raw_parent_ids()[9] = 4;
        m_tree->raw_to_parent_edge_ids()[9] = 8;
        m_tree->raw_parent_ids()[10] = 4;
        m_tree->raw_to_parent_edge_ids()[10] = 9;
        m_tree->raw_parent_ids()[11] = 5;
        m_tree->raw_to_parent_edge_ids()[11] = 10;
        m_tree->raw_parent_ids()[12] = 5;
        m_tree->raw_to_parent_edge_ids()[12] = 11;
        m_tree->raw_parent_ids()[13] = 6;
        m_tree->raw_to_parent_edge_ids()[13] = 12;
        m_tree->raw_parent_ids()[14] = 6;
        m_tree->raw_to_parent_edge_ids()[14] = 13;

        /* node weights */
        for(luint_t i = 0; i < 15; ++i)
            m_tree->raw_degrees()[i] = 1.0;

        /* create remaining data for tree */
        m_tree->finalize(false, m_graph.get());
    }

    void
    create_label_set()
    {
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> all_label_set(15, 0);
        for(uint_t i = 0; i < 15; ++i)
            all_label_set[i] = i;

        m_label_set = std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>>(
            new LabelSet<COSTTYPE, SIMDWIDTH>(m_graph->num_nodes(), true));
        for(luint_t i = 0; i < 15; ++i)
            m_label_set->set_label_set_for_node(i, all_label_set);
    }

    void
    create_cost_bundle()
    {
        m_cbundle = std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>>(
            new CostBundle<COSTTYPE, SIMDWIDTH>(m_graph.get()));
    }

    void
    create_unaries()
    {
        m_unaries.reserve(15);

        /* root / inner node costs */
        std::vector<COSTTYPE> node_costs(15, (COSTTYPE) 0);
        for(uint_t n = 0; n < 7; ++n)
        {
            m_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
                SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(
                n, m_label_set.get())));
            m_unaries.back()->set_costs(node_costs);
        }

        /* leaf nodes - cost 1 for one label per node, 2 for all others */
        for(uint_t n = 0; n < 8; ++n)
        {
            std::vector<COSTTYPE> leaf_costs(15, (COSTTYPE) 2.0);
            leaf_costs[n] = (COSTTYPE) 1.0;

            m_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
                SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(
                7 + n, m_label_set.get())));
            m_unaries.back()->set_costs(leaf_costs);
        }

        for(uint_t n = 0; n < 15; ++n)
            m_cbundle->set_unary_costs(n, m_unaries[n].get());
    }

    void
    create_pairwise_data()
    {
         /**
         * label vector for pairwise cost test - label set size
         * divisible by all used SIMDWIDTHs
         */
        m_labels.resize(16, 0);
        for(_iv_st<COSTTYPE, SIMDWIDTH> l = 0; l < 16; ++l)
            m_labels[l] = l;

        /* storage for cost output */
        m_cost_out.resize(16, std::numeric_limits<
            _s_t<COSTTYPE, SIMDWIDTH>>::max());
    }

protected:
    std::unique_ptr<Graph<COSTTYPE>> m_graph;
    std::unique_ptr<Tree<COSTTYPE>> m_tree;
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> m_label_set;
    std::vector<std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>>> m_unaries;
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> m_pairwise;
    std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>> m_cbundle;

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_labels;
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_cost_out;
};
TYPED_TEST_CASE_P(mapMAPTestDynamicProgramming);

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwisePotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwisePotts<COSTTYPE, SIMDWIDTH>> pairwise(
        new PairwisePotts<COSTTYPE, SIMDWIDTH>);

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(15, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
            ASSERT_NEAR(this->m_cost_out[i], pairwise->get_c() *
                (i != l_i), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseAntiPotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(
        new PairwiseAntipotts<COSTTYPE, SIMDWIDTH>);

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(15, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
            ASSERT_NEAR(this->m_cost_out[i], 1.0 * (i == l_i), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseLinearPeak)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(
        new PairwiseLinearPeak<COSTTYPE, SIMDWIDTH>);

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(15, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
            ASSERT_NEAR(this->m_cost_out[i], 
                std::max(0.0, 1.0 * (2 - std::abs(i - l_i))),
                0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseTruncatedLinear)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>> pairwise(
        new PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>({2.0, 5.0}));

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
            ASSERT_NEAR(this->m_cost_out[i], std::min((COSTTYPE)
                pairwise->get_c() * std::abs(i - l_i),
                (COSTTYPE) pairwise->get_label_diff_cap()), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseTruncatedQuadratic)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>> pairwise(
        new PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH>({1.5, 9.0}));

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
        ASSERT_NEAR(this->m_cost_out[i], std::min((COSTTYPE)
            pairwise->get_c() * std::abs(i - l_i) * std::abs(i - l_i),
            (COSTTYPE) pairwise->get_label_diff_cap()), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseIndependentTable)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create dense cost table */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(15 * 15);

    /* fill table as if it was a truncated linear function with c = 2 */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_1 = 0; l_1 < 15; ++l_1)
        for(_iv_st<COSTTYPE, SIMDWIDTH> l_2 = 0; l_2 < 15; ++l_2)
            costs[l_1 * 15 + l_2] = std::min((COSTTYPE)
                std::abs(l_1 - l_2), (COSTTYPE) 2.0);

    /* test like the linear truncated function */
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(new
        PairwiseTable<COSTTYPE, SIMDWIDTH>(0, 1, this->m_label_set.get(),
        costs));

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 15; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(15, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);

            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_pairwise_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost,
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 15; ++i)
            ASSERT_NEAR(this->m_cost_out[i], std::min((COSTTYPE)
                std::abs(i - l_i), (COSTTYPE) 2.0), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreePotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwisePotts<COSTTYPE, SIMDWIDTH>);
    this->m_cbundle->set_pairwise_costs(this->m_pairwise.get());

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(this->m_cbundle.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);

    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 15, (COSTTYPE) 0.1);
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreeTruncatedLinear)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>({2.0}));
    this->m_cbundle->set_pairwise_costs(this->m_pairwise.get());

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(this->m_cbundle.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);

    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 15, (COSTTYPE) 0.1);
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreeIndependentTable)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create dense cost table */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(15 * 15);

    /* fill table as if it was a truncated linear function with c = 2 */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_1 = 0; l_1 < 15; ++l_1)
        for(_iv_st<COSTTYPE, SIMDWIDTH> l_2 = 0; l_2 < 15; ++l_2)
            costs[l_1 * 15 + l_2] = std::min((COSTTYPE)
                std::abs(l_1 - l_2), (COSTTYPE) 2.0);

    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwiseTable<COSTTYPE, SIMDWIDTH>(0, 1, this->m_label_set.get(),
        costs));
    this->m_cbundle->set_pairwise_costs(this->m_pairwise.get());

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(this->m_cbundle.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);

    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 15, (COSTTYPE) 0.1);
}

/* register test cases */
REGISTER_TYPED_TEST_CASE_P(mapMAPTestDynamicProgramming,
    TestPairwisePotts,
    TestPairwiseAntiPotts,
    TestPairwiseLinearPeak,
    TestPairwiseTruncatedLinear,
    TestPairwiseTruncatedQuadratic,
    TestPairwiseIndependentTable,
    TestTreePotts,
    TestTreeTruncatedLinear,
    TestTreeIndependentTable);

/* instantiate tests */
typedef ::testing::Types<
    #if defined(__SSE4_2__)
    TestTuple<float, 4>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<float, 8>,
    #endif /* __AVX__ */
    #if defined(__SSE4_2__)
    TestTuple<double, 2>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<double, 4>,
    #endif /* __AVX__ */
    TestTuple<float, 1>,
    TestTuple<double, 1>
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(DynamicProgrammingTest,
    mapMAPTestDynamicProgramming, TestTupleInstances);

NS_MAPMAP_END