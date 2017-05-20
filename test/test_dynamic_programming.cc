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
#include "header/dynamic_programming.h"
#include "header/costs.h"
#include "header/graph.h"
#include "header/cost_instances/pairwise_potts.h"
#include "header/cost_instances/pairwise_antipotts.h"
#include "header/cost_instances/pairwise_truncated_linear.h"
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

        m_graph->add_edge(0, 1, 1.0);
        m_graph->add_edge(0, 2, 1.0);

        m_graph->add_edge(1, 3, 1.0);
        m_graph->add_edge(1, 4, 1.0);
        m_graph->add_edge(2, 5, 1.0);
        m_graph->add_edge(2, 6, 1.0);

        m_graph->add_edge(3, 7, 1.0);
        m_graph->add_edge(3, 8, 1.0);
        m_graph->add_edge(4, 9, 1.0);
        m_graph->add_edge(4, 10, 1.0);
        m_graph->add_edge(5, 11, 1.0);
        m_graph->add_edge(5, 12, 1.0);
        m_graph->add_edge(6, 13, 1.0);
        m_graph->add_edge(6, 14, 1.0);
    }

    void 
    create_tree()
    {
         /* graph is a tree, hence just can copy parent relations */
        m_tree = std::unique_ptr<Tree<COSTTYPE>>(new Tree<COSTTYPE>(15, 14));

        /* parent ids */
        m_tree->raw_parent_ids()[0] = 0;

        m_tree->raw_parent_ids()[1] = 0;
        m_tree->raw_parent_ids()[2] = 0;

        m_tree->raw_parent_ids()[3] = 1;
        m_tree->raw_parent_ids()[4] = 1;
        m_tree->raw_parent_ids()[5] = 2;
        m_tree->raw_parent_ids()[6] = 2;

        m_tree->raw_parent_ids()[7] = 3;
        m_tree->raw_parent_ids()[8] = 3;
        m_tree->raw_parent_ids()[9] = 4;
        m_tree->raw_parent_ids()[10] = 4;
        m_tree->raw_parent_ids()[11] = 5;
        m_tree->raw_parent_ids()[12] = 5;
        m_tree->raw_parent_ids()[13] = 6;
        m_tree->raw_parent_ids()[14] = 6;

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
    create_unaries()
    {
        m_unaries = std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>>(
            new UnaryTable<COSTTYPE, SIMDWIDTH>(
            m_graph.get(), m_label_set.get()));

        /* leaf nodes - cost 1 for one label per node, 2 for all others */
        for(uint_t n = 0; n < 8; ++n)
        {
            std::vector<COSTTYPE> leaf_costs(15, (COSTTYPE) 2.0);
            leaf_costs[n] = (COSTTYPE) 1.0;

            m_unaries->set_costs_for_node(7 + n, leaf_costs);
        }

        std::vector<COSTTYPE> node_costs(15, (COSTTYPE) 0);
        for(uint_t n = 0; n < 7; ++n)
            m_unaries->set_costs_for_node(n, node_costs);
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
    std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>> m_unaries;
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> m_pairwise;

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_labels;
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_cost_out;
};
TYPED_TEST_CASE_P(mapMAPTestDynamicProgramming);

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwisePotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise( 
        new PairwisePotts<COSTTYPE, SIMDWIDTH>);

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 16; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);
            
            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_binary_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost, 
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 16; ++i)
            ASSERT_NEAR(this->m_cost_out[i], 1.0 * (i != l_i), 0.01);

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
    for(l_i = 0; l_i < 16; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);
            
            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_binary_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost, 
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 16; ++i)
            ASSERT_NEAR(this->m_cost_out[i], 1.0 * (i == l_i), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseTruncatedLinear)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise( 
        new PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>(2.0));

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 16; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);
            
            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_binary_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost, 
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 16; ++i)
            ASSERT_NEAR(this->m_cost_out[i], std::min((COSTTYPE) 
                std::abs(i - l_i), (COSTTYPE) 2.0), 0.01);

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
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(16 * 16);

    /* fill table as if it was a truncated linear function with c = 2 */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_1 = 0; l_1 < 16; ++l_1)
        for(_iv_st<COSTTYPE, SIMDWIDTH> l_2 = 0; l_2 < 16; ++l_2)
            costs[l_1 * 16 + l_2] = std::min((COSTTYPE) 
                std::abs(l_1 - l_2), (COSTTYPE) 2.0);

    /* test like the linear truncated function */
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(new 
        PairwiseTable<COSTTYPE, SIMDWIDTH>(16, costs));

    _iv_st<COSTTYPE, SIMDWIDTH> l_i, i;
    for(l_i = 0; l_i < 16; ++l_i)
    {
        /* compute costs with one label fixed */
        for(uint_t c = 0; c < DIV_UP(16, SIMDWIDTH); ++c)
        {
            _iv_t<COSTTYPE, SIMDWIDTH> l = iv_load<COSTTYPE, SIMDWIDTH>(
                &this->m_labels[c * SIMDWIDTH]);
            
            _v_t<COSTTYPE, SIMDWIDTH> cost = pairwise->get_binary_costs(l,
                iv_init<COSTTYPE, SIMDWIDTH>(l_i));
            v_store<COSTTYPE, SIMDWIDTH>(cost, 
                &this->m_cost_out[c * SIMDWIDTH]);
        }

        /* compare costs to ground truth */
        for(i = 0; i < 16; ++i)
            ASSERT_NEAR(this->m_cost_out[i], std::min((COSTTYPE) 
                std::abs(i - l_i), (COSTTYPE) 2.0), 0.01);

        /* reset cost values for next loop iteration */
        std::fill(this->m_cost_out.begin(), this->m_cost_out.end(),
            std::numeric_limits<_s_t<COSTTYPE, SIMDWIDTH>>::max());
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseDependentTable)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /**
     * use tree graph as in the other tests and each node's label set
     * is the node's ID + 3 labels left and right of it (max. 7 labels).
     */
    
    /* create label set */
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> sp_label_set(
        new LabelSet<COSTTYPE, SIMDWIDTH>(15, true));
    for(luint_t n = 0; n < 15; ++n)
    {
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls;

        const _iv_st<COSTTYPE, SIMDWIDTH> min_label = (n < 3) ? 0 : (n - 3);
        const _iv_st<COSTTYPE, SIMDWIDTH> max_label = (n > 12) ? 16 : (n + 4);

        for(_iv_st<COSTTYPE, SIMDWIDTH> ll = min_label; ll < max_label; ++ll)
            ls.push_back(ll);

        sp_label_set->set_label_set_for_node(n, ls);
    }

    /* create packed cost table */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs;
    for(const GraphEdge<COSTTYPE>& e : this->m_graph->edges())
    {
        const luint_t node_a = e.node_a;
        const luint_t node_b = e.node_b;

        const _iv_st<COSTTYPE, SIMDWIDTH> node_a_lsize = 
            sp_label_set->label_set_size(node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> node_b_lsize = 
            sp_label_set->label_set_size(node_b);

        _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i;
        for(l_a_i = 0; l_a_i < node_a_lsize; ++l_a_i)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> l_a = 
                sp_label_set->label_from_offset(node_a, l_a_i);

            for(l_b_i = 0; l_b_i < node_b_lsize; ++l_b_i)
            {
                const _iv_st<COSTTYPE, SIMDWIDTH> l_b = 
                    sp_label_set->label_from_offset(node_b, l_b_i);

                costs.push_back(std::min((COSTTYPE) 
                    std::abs(l_a - l_b), (COSTTYPE) 2.0));
            }
        }
    }

    /* test like the linear truncated function */
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(new 
        PairwiseTable<COSTTYPE, SIMDWIDTH>(sp_label_set.get(), 
        this->m_graph.get(), costs));

    _iv_st<COSTTYPE, SIMDWIDTH> l_out[SIMDWIDTH];
    _s_t<COSTTYPE, SIMDWIDTH> c_out[SIMDWIDTH];
    for(const GraphEdge<COSTTYPE>& e : this->m_graph->edges())
    {
        const luint_t node_a = e.node_a;
        const luint_t node_b = e.node_b;

        const _iv_st<COSTTYPE, SIMDWIDTH> node_a_lsize = 
            sp_label_set->label_set_size(node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> node_b_lsize = 
            sp_label_set->label_set_size(node_b);
        const _iv_st<COSTTYPE, SIMDWIDTH> aug_node_b_lsize = 
            DIV_UP(sp_label_set->label_set_size(node_b), SIMDWIDTH);

        _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i;
        for(l_a_i = 0; l_a_i < node_a_lsize; l_a_i += SIMDWIDTH)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> ll_a = 
                sp_label_set->label_from_offset(node_a, l_a_i);
            const _iv_t<COSTTYPE, SIMDWIDTH> l_a = 
                iv_init<COSTTYPE, SIMDWIDTH>(ll_a);

            for(l_b_i = 0; l_b_i < aug_node_b_lsize; l_b_i += SIMDWIDTH)
            {
                const _iv_t<COSTTYPE, SIMDWIDTH> l_b = 
                    sp_label_set->labels_from_offset(node_b, l_b_i);

                _v_t<COSTTYPE, SIMDWIDTH> costs = 
                    pairwise->get_binary_costs(node_a, l_a, node_b, l_b);

                /* store costs and check them */
                v_store<COSTTYPE, SIMDWIDTH>(costs, c_out);
                iv_store<COSTTYPE, SIMDWIDTH>(l_b, l_out);

                const uint_t to_check = std::min(node_b_lsize - l_b_i,
                    (_iv_st<COSTTYPE, SIMDWIDTH>) SIMDWIDTH);
                for(uint_t i = 0; i < to_check; ++i)
                    ASSERT_NEAR(c_out[i], std::min((COSTTYPE) 
                        std::abs(ll_a - l_out[i]), (COSTTYPE) 2.0), 0.01);
            }
        }
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestPairwiseDependentTableWrongOrder)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /**
     * use tree graph as in the other tests and each node's label set
     * is the node's ID + 3 labels left and right of it (max. 7 labels).
     */
    
    /* create label set */
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> sp_label_set(new LabelSet<COSTTYPE,
        SIMDWIDTH>(15, true));
    for(luint_t n = 0; n < 15; ++n)
    {
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls;

        const _iv_st<COSTTYPE, SIMDWIDTH> min_label = (n < 3) ? 0 : (n - 3);
        const _iv_st<COSTTYPE, SIMDWIDTH> max_label = (n > 12) ? 16 : (n + 4);

        for(_iv_st<COSTTYPE, SIMDWIDTH> ll = min_label; ll < max_label; ++ll)
            ls.push_back(ll);

        sp_label_set->set_label_set_for_node(n, ls);
    }

    /* create packed cost table */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs;
    for(const GraphEdge<COSTTYPE>& e : this->m_graph->edges())
    {
        const luint_t node_a = e.node_a;
        const luint_t node_b = e.node_b;

        const _iv_st<COSTTYPE, SIMDWIDTH> node_a_lsize = 
            sp_label_set->label_set_size(node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> node_b_lsize = 
            sp_label_set->label_set_size(node_b);

        _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i;
        for(l_a_i = 0; l_a_i < node_a_lsize; ++l_a_i)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> l_a = 
                sp_label_set->label_from_offset(node_a, l_a_i);

            for(l_b_i = 0; l_b_i < node_b_lsize; ++l_b_i)
            {
                const _iv_st<COSTTYPE, SIMDWIDTH> l_b = 
                    sp_label_set->label_from_offset(node_b, l_b_i);

                costs.push_back(std::min((COSTTYPE) 
                    std::abs(l_a - l_b), (COSTTYPE) 2.0));
            }
        }
    }

    /* test like the linear truncated function */
    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> pairwise(new 
        PairwiseTable<COSTTYPE, SIMDWIDTH>(sp_label_set.get(), 
        this->m_graph.get(), costs));

    _iv_st<COSTTYPE, SIMDWIDTH> l_out[SIMDWIDTH];
    _s_t<COSTTYPE, SIMDWIDTH> c_out[SIMDWIDTH];
    for(const GraphEdge<COSTTYPE>& e : this->m_graph->edges())
    {
        const luint_t node_a = e.node_a;
        const luint_t node_b = e.node_b;

        const _iv_st<COSTTYPE, SIMDWIDTH> node_a_lsize = 
            sp_label_set->label_set_size(node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> node_b_lsize = 
            sp_label_set->label_set_size(node_b);
        const _iv_st<COSTTYPE, SIMDWIDTH> aug_node_b_lsize = 
            DIV_UP(sp_label_set->label_set_size(node_b), SIMDWIDTH);

        _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i;
        for(l_a_i = 0; l_a_i < node_a_lsize; l_a_i += SIMDWIDTH)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> ll_a = 
                sp_label_set->label_from_offset(node_a, l_a_i);
            const _iv_t<COSTTYPE, SIMDWIDTH> l_a = 
                iv_init<COSTTYPE, SIMDWIDTH>(ll_a);

            for(l_b_i = 0; l_b_i < aug_node_b_lsize; l_b_i += SIMDWIDTH)
            {
                const _iv_t<COSTTYPE, SIMDWIDTH> l_b = 
                    sp_label_set->labels_from_offset(node_b, l_b_i);

                /* change order of ndoe a/b such that a > b */
                _v_t<COSTTYPE, SIMDWIDTH> costs = 
                    pairwise->get_binary_costs(node_b, l_b, node_a, l_a);

                /* store costs and check them */
                v_store<COSTTYPE, SIMDWIDTH>(costs, c_out);
                iv_store<COSTTYPE, SIMDWIDTH>(l_b, l_out);

                const uint_t to_check = std::min(node_b_lsize - l_b_i,
                    (_iv_st<COSTTYPE, SIMDWIDTH>) SIMDWIDTH);
                for(uint_t i = 0; i < to_check; ++i)
                    ASSERT_NEAR(c_out[i], std::min((COSTTYPE) 
                        std::abs(ll_a - l_out[i]), (COSTTYPE) 2.0), 0.01);
            }
        }
    }
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreePotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwisePotts<COSTTYPE, SIMDWIDTH>);

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH,
        UnaryTable<COSTTYPE, SIMDWIDTH>,
        PairwisePotts<COSTTYPE, SIMDWIDTH>> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(
        (const UnaryTable<COSTTYPE, SIMDWIDTH>*) this->m_unaries.get(), 
        (const PairwisePotts<COSTTYPE, SIMDWIDTH>*) this->m_pairwise.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);

    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 15, (COSTTYPE) 0.1);
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreeAntiPotts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwiseAntipotts<COSTTYPE, SIMDWIDTH>);

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH,
        UnaryTable<COSTTYPE, SIMDWIDTH>,
        PairwiseAntipotts<COSTTYPE, SIMDWIDTH>> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(
        (const UnaryTable<COSTTYPE, SIMDWIDTH>*) this->m_unaries.get(), 
        (const PairwiseAntipotts<COSTTYPE, SIMDWIDTH>*) this->m_pairwise.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);
   
    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 8, (COSTTYPE) 0.1);
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreeTruncatedLinear)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

   this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>(2.0));

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH,
        UnaryTable<COSTTYPE, SIMDWIDTH>,
        PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(
        (const UnaryTable<COSTTYPE, SIMDWIDTH>*) this->m_unaries.get(), 
        (const PairwiseTruncatedLinear<COSTTYPE, SIMDWIDTH>*) 
        this->m_pairwise.get());

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
        new PairwiseTable<COSTTYPE, SIMDWIDTH>(15, costs));

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH,
        UnaryTable<COSTTYPE, SIMDWIDTH>,
        PairwiseTable<COSTTYPE, SIMDWIDTH>> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(this->m_label_set.get());
    cdp.set_costs(
        (const UnaryTable<COSTTYPE, SIMDWIDTH>*) this->m_unaries.get(), 
        (const PairwiseTable<COSTTYPE, SIMDWIDTH>*) this->m_pairwise.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);
   
    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 15, (COSTTYPE) 0.1);
}

TYPED_TEST_P(mapMAPTestDynamicProgramming, TestTreeDependentTable)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /**
     * use tree graph as in the other tests and each node's label set
     * is the node's ID + 3 labels left and right of it (max. 7 labels).
     */
    
    /* create sparse label set and unary costs */
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> sp_label_set(
        new LabelSet<COSTTYPE, SIMDWIDTH>(15, true));
    for(luint_t n = 0; n < 15; ++n)
    {
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls;

        const _iv_st<COSTTYPE, SIMDWIDTH> min_label = (n < 3) ? 0 : (n - 3);
        const _iv_st<COSTTYPE, SIMDWIDTH> max_label = (n > 12) ? 16 : (n + 4);

        for(_iv_st<COSTTYPE, SIMDWIDTH> ll = min_label; ll < max_label; ++ll)
            ls.push_back(ll);

        sp_label_set->set_label_set_for_node(n, ls);
    }

    std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>> sp_unaries(
        new UnaryTable<COSTTYPE, SIMDWIDTH>(this->m_graph.get(), 
        sp_label_set.get()));
    for(luint_t n = 0; n < 15; ++n)
    {
        std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary;

        const _iv_st<COSTTYPE, SIMDWIDTH> min_label = (n < 3) ? 0 : (n - 3);
        const _iv_st<COSTTYPE, SIMDWIDTH> max_label = (n > 12) ? 16 : (n + 4);

        for(_iv_st<COSTTYPE, SIMDWIDTH> ll = min_label; ll < max_label; ++ll)
        {
            if(n >= 7)
                unary.push_back((((_iv_st<COSTTYPE, SIMDWIDTH>) n == ll)) ? 1 : 2);
            else
                unary.push_back(0);
        }

        sp_unaries->set_costs_for_node(n, unary);
    }
    
    /* create packed cost table */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs;
    for(const GraphEdge<COSTTYPE>& e : this->m_graph->edges())
    {
        const luint_t node_a = e.node_a;
        const luint_t node_b = e.node_b;

        const _iv_st<COSTTYPE, SIMDWIDTH> node_a_lsize = 
            sp_label_set->label_set_size(node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> node_b_lsize = 
            sp_label_set->label_set_size(node_b);

        _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i;
        for(l_a_i = 0; l_a_i < node_a_lsize; ++l_a_i)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> l_a = 
                sp_label_set->label_from_offset(node_a, l_a_i);

            for(l_b_i = 0; l_b_i < node_b_lsize; ++l_b_i)
            {
                const _iv_st<COSTTYPE, SIMDWIDTH> l_b = 
                    sp_label_set->label_from_offset(node_b, l_b_i);

                costs.push_back(std::min((COSTTYPE) 
                    std::abs(l_a - l_b), (COSTTYPE) 2.0));
            }
        }
    }

    /* test like the linear truncated function */
    this->m_pairwise = std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>(
        new PairwiseTable<COSTTYPE, SIMDWIDTH>(sp_label_set.get(), 
        this->m_graph.get(), costs));

    CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH,
        UnaryTable<COSTTYPE, SIMDWIDTH>,
        PairwiseTable<COSTTYPE, SIMDWIDTH>> cdp;
    cdp.set_graph(this->m_graph.get());
    cdp.set_tree(this->m_tree.get());
    cdp.set_label_set(sp_label_set.get());
    cdp.set_costs(
        (const UnaryTable<COSTTYPE, SIMDWIDTH>*) sp_unaries.get(), 
        (const PairwiseTable<COSTTYPE, SIMDWIDTH>*) this->m_pairwise.get());

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> solution(15, 0);
    COSTTYPE tree_result = cdp.optimize(solution);
   
    /* check results */
    ASSERT_NEAR(tree_result, (COSTTYPE) 24, (COSTTYPE) 0.1);
}

/* register test cases */
REGISTER_TYPED_TEST_CASE_P(mapMAPTestDynamicProgramming, 
    TestPairwisePotts,
    TestPairwiseAntiPotts,
    TestPairwiseTruncatedLinear,
    TestPairwiseIndependentTable,
    TestPairwiseDependentTable,
    TestPairwiseDependentTableWrongOrder,
    TestTreePotts,
    TestTreeAntiPotts,
    TestTreeTruncatedLinear,
    TestTreeIndependentTable,
    TestTreeDependentTable);

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