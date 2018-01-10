/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <list>
#include <utility>
#include <tuple>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include "header/defines.h"
#include "header/graph.h"
#include "header/multilevel.h"
#include "header/multilevel_instances/group_same_label.h"
#include "header/cost_instances/unary_table.h"
#include "header/cost_instances/pairwise_table.h"
#include "test/util_test.h"

NS_MAPMAP_BEGIN

template<typename T, uint_t N>
class TestTuple {
public:
    typedef T Type;
    static const uint_t Value = N;
};

template<typename T>
class mapMAPTestMultilevel : public ::testing::Test
{
public:
    /* rename template parameters for convenience */
    typedef typename T::Type COSTTYPE;
    static const uint_t SIMDWIDTH = T::Value;

    mapMAPTestMultilevel()
    {

    }

    ~mapMAPTestMultilevel()
    {

    }

    void
    SetUp()
    {
        /* create original graph - a 6x6 regular 4-grid */
        m_original_graph = createComponentGrid<COSTTYPE>(1, 6);

        /* create label set */
        create_label_set();

        /* create unary and pairwise costs */
        create_cost_bundle();
        create_unary_costs();
        create_pairwise_costs();

        /* create a labelling for grouping nodes */
        m_first_labelling = { /* indices; labels only as comments */
            3, 3, 3, 2, 1, 0, /* 3, 3, 3, 2, 1, 0 */
            3, 3, 4, 3, 0, 0, /* 3, 3, 4, 3, 0, 0 */
            1, 1, 2, 1, 0, 0, /* 1, 1, 2, 1, 0, 0 */
            0, 0, 3, 5, 2, 1, /* 0, 0, 3, 5, 2, 1 */
            1, 0, 4, 5, 0, 1, /* 1, 0, 4, 5, 0, 1 */
            3, 4, 3, 4, 1, 0  /* 3, 4, 3, 4, 1, 0 */
        };
        m_second_labelling = { /* indices; labels only as comments */
            4, 0, 0, 0, /* 4, 1, 0, 4 */
            1, 1, 1, 2, /* 2, 2, 2, 2 */
            2, 1, 0 /* 5, 6, 1 */
        };

        /* execute first round of region graph building, grouping same labels */
        m_grouping = std::unique_ptr<MultilevelCriterion<COSTTYPE, SIMDWIDTH>>(
            new GroupSameLabel<COSTTYPE, SIMDWIDTH>);
        m_multilevel = std::unique_ptr<Multilevel<COSTTYPE, SIMDWIDTH>>(
            new Multilevel<COSTTYPE, SIMDWIDTH>(
                m_original_graph.get(),
                m_original_label_set.get(),
                m_original_cbundle.get(),
                m_grouping.get(),
                true));

        m_multilevel->next_level(m_first_labelling, this->m_dep_labelling);
    }

    void
    TearDown()
    {

    }

protected:
    void
    create_label_set()
    {
        /* supernode I (0) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_0 = {0, 1, 2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_1 = {0, 1, 2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_2 = {0, 1, 2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_6 = {0, 1, 2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_7 = {0, 1, 2, 3, 4, 5, 6};

        /* supernode II (1) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_3 = {0, 1, 2};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_4 = {1, 2, 3};

        /* supernode III (2) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_5 = {0, 2, 4, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_10 = {0, 1, 3, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_11 = {0, 1, 3, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_17 = {0, 1, 3, 5};

        /* supernode IV (3) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_8 = {0, 1, 2, 3, 4};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_9 = {1, 2, 3, 4, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_14 = {2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_15 = {3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_16 = {4, 5, 6};

        /* supernode V (4) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_12 = {0, 1, 2, 3};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_13 = {0, 1, 2, 3};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_18 = {1, 2, 3};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_19 = {1, 2, 3};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_24 = {0, 1, 2, 3, 4};

        /* supernode VI (5) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_20 = {0, 2, 4, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_21 = {0, 1, 2, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_26 = {0, 2, 3, 4, 6};

        /* supernode VII (6) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_22 = {0, 1, 2, 4};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_23 = {1, 2, 4, 6};

        /* supernode VIII (7) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_25 = {0, 1, 2, 3};

        /* supernode IX (8) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_27 = {0, 1, 2, 3, 4, 5, 6};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_30 = {0, 1, 3, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_31 = {1, 2, 3, 4, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_32 = {0, 1, 3, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_33 = {0, 1, 3, 4, 5};

        /* supernode X (9) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_28 = {0, 6};

        /* supernode XI (10) */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_29 = {0, 1, 2, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_34 = {0, 1, 3, 4, 5};
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> ls_35 = {1, 2, 5};

        /* assign label sets */
        m_original_label_set = std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>>(
            new LabelSet<COSTTYPE, SIMDWIDTH>(36, true));

        m_original_label_set->set_label_set_for_node(0, ls_0);
        m_original_label_set->set_label_set_for_node(1, ls_1);
        m_original_label_set->set_label_set_for_node(2, ls_2);
        m_original_label_set->set_label_set_for_node(3, ls_3);
        m_original_label_set->set_label_set_for_node(4, ls_4);
        m_original_label_set->set_label_set_for_node(5, ls_5);
        m_original_label_set->set_label_set_for_node(6, ls_6);
        m_original_label_set->set_label_set_for_node(7, ls_7);
        m_original_label_set->set_label_set_for_node(8, ls_8);
        m_original_label_set->set_label_set_for_node(9, ls_9);
        m_original_label_set->set_label_set_for_node(10, ls_10);
        m_original_label_set->set_label_set_for_node(11, ls_11);
        m_original_label_set->set_label_set_for_node(12, ls_12);
        m_original_label_set->set_label_set_for_node(13, ls_13);
        m_original_label_set->set_label_set_for_node(14, ls_14);
        m_original_label_set->set_label_set_for_node(15, ls_15);
        m_original_label_set->set_label_set_for_node(16, ls_16);
        m_original_label_set->set_label_set_for_node(17, ls_17);
        m_original_label_set->set_label_set_for_node(18, ls_18);
        m_original_label_set->set_label_set_for_node(19, ls_19);
        m_original_label_set->set_label_set_for_node(20, ls_20);
        m_original_label_set->set_label_set_for_node(21, ls_21);
        m_original_label_set->set_label_set_for_node(22, ls_22);
        m_original_label_set->set_label_set_for_node(23, ls_23);
        m_original_label_set->set_label_set_for_node(24, ls_24);
        m_original_label_set->set_label_set_for_node(25, ls_25);
        m_original_label_set->set_label_set_for_node(26, ls_26);
        m_original_label_set->set_label_set_for_node(27, ls_27);
        m_original_label_set->set_label_set_for_node(28, ls_28);
        m_original_label_set->set_label_set_for_node(29, ls_29);
        m_original_label_set->set_label_set_for_node(30, ls_30);
        m_original_label_set->set_label_set_for_node(31, ls_31);
        m_original_label_set->set_label_set_for_node(32, ls_32);
        m_original_label_set->set_label_set_for_node(33, ls_33);
        m_original_label_set->set_label_set_for_node(34, ls_34);
        m_original_label_set->set_label_set_for_node(35, ls_35);
    }

    void
    create_cost_bundle()
    {
        m_original_cbundle = std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>>(
            new CostBundle<COSTTYPE, SIMDWIDTH>(m_original_graph.get()));
    }

    void
    create_unary_costs()
    {
        m_original_unaries.reserve(36);

        /* unary costs l for label l */
        for(luint_t n = 0; n < 36; ++n)
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> label_set_size =
                m_original_label_set->label_set_size(n);
            std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(label_set_size, 0);

            _iv_st<COSTTYPE, SIMDWIDTH> l_i, l;
            for(l_i = 0; l_i < label_set_size; ++l_i)
            {
                l = m_original_label_set->label_from_offset(n, l_i);
                costs[l_i] = (_s_t<COSTTYPE, SIMDWIDTH>) l;
            }

            m_original_unaries.emplace_back(std::unique_ptr<UnaryTable<COSTTYPE,
                SIMDWIDTH>>(new UnaryTable<COSTTYPE, SIMDWIDTH>(n,
                m_original_label_set.get())));
            m_original_unaries.back()->set_costs(costs);
            m_original_cbundle->set_unary_costs(n,
                m_original_unaries.back().get());
        }
    }

    void
    create_pairwise_costs()
    {
        m_original_pairwise.reserve(m_original_graph->num_edges());

        /* pairwise costs: antipotts multiplied by sum of incident node ids */
        for(luint_t e_id = 0; e_id < m_original_graph->num_edges(); ++e_id)
        {
            const GraphEdge<COSTTYPE>& e = m_original_graph->edges()[e_id];

            const luint_t n_a = e.node_a;
            const luint_t n_b = e.node_b;

            const _iv_st<COSTTYPE, SIMDWIDTH> labels_a =
                m_original_label_set->label_set_size(n_a);
            const _iv_st<COSTTYPE, SIMDWIDTH> labels_b =
                m_original_label_set->label_set_size(n_b);

            std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(labels_a * labels_b);

            _iv_st<COSTTYPE, SIMDWIDTH> l_a_i, l_b_i, l_a, l_b;
            for(l_a_i = 0; l_a_i < labels_a; ++l_a_i)
            {
                l_a = m_original_label_set->label_from_offset(n_a, l_a_i);
                for(l_b_i = 0; l_b_i < labels_b; ++l_b_i)
                {
                    l_b = m_original_label_set->label_from_offset(n_b, l_b_i);
                    costs[l_a_i * labels_b + l_b_i] =
                        (_s_t<COSTTYPE, SIMDWIDTH>) (n_a + n_b) * (l_a == l_b);
                }
            }

            m_original_pairwise.emplace_back(std::unique_ptr<PairwiseTable<
                COSTTYPE, SIMDWIDTH>>(new PairwiseTable<COSTTYPE, SIMDWIDTH>(
                n_a, n_b, m_original_label_set.get(), costs)));
            m_original_cbundle->set_pairwise_costs(e_id,
                m_original_pairwise.back().get());
        }
    }

protected:
    std::unique_ptr<Graph<COSTTYPE>> m_original_graph;
    std::unique_ptr<LabelSet<COSTTYPE, SIMDWIDTH>> m_original_label_set;
    std::unique_ptr<CostBundle<COSTTYPE, SIMDWIDTH>> m_original_cbundle;
    std::vector<std::unique_ptr<UnaryTable<COSTTYPE, SIMDWIDTH>>>
        m_original_unaries;
    std::vector<std::unique_ptr<PairwiseTable<COSTTYPE, SIMDWIDTH>>>
        m_original_pairwise;

    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> m_pw_costs;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_first_labelling;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_second_labelling;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_dep_labelling;

    std::unique_ptr<Multilevel<COSTTYPE, SIMDWIDTH>> m_multilevel;
    std::unique_ptr<MultilevelCriterion<COSTTYPE, SIMDWIDTH>> m_grouping;
};
TYPED_TEST_CASE_P(mapMAPTestMultilevel);

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelNumNodes)
{
    typedef typename TypeParam::Type COSTTYPE;

    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    ASSERT_EQ(graph->num_nodes(), 11);
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelNumEdges)
{
    typedef typename TypeParam::Type COSTTYPE;

    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    ASSERT_EQ(graph->edges().size(), 22);
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelEdges)
{
    typedef typename TypeParam::Type COSTTYPE;

    typedef std::tuple<luint_t, luint_t, luint_t> edge;
    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    /**
     * check if all edges are there (and no more) -- since we are using
     * node-dependent pairwise costs, all weights are 1
     */
    std::list<edge> required =
    {
        edge(0, 1, 1), edge(1, 2, 1), edge(0, 3, 1),
        edge(0, 4, 1), edge(4, 5, 1), edge(5, 6, 1),
        edge(4, 8, 1), edge(8, 9, 1), edge(6, 10, 1),
        edge(9, 10, 1), edge(2, 6, 1), edge(1, 3, 1),
        edge(2, 3, 1), edge(3, 4, 1), edge(4, 7, 1),
        edge(3, 5, 1), edge(5, 7, 1), edge(5, 8, 1),
        edge(6, 9, 1), edge(8, 10, 1), edge(3, 6, 1),
        edge(7, 8, 1)
    };

    /* necessary condition: equal size */
    ASSERT_EQ(graph->edges().size(), required.size());

    for(const GraphEdge<COSTTYPE>& e : graph->edges())
    {
        edge te = edge(e.node_a, e.node_b, (luint_t) e.weight);

        /* try to find edge in required list */
        auto it = std::find(required.begin(), required.end(), te);
        ASSERT_NE(it, required.end());

        required.erase(it);
    }

    /* after removing all edges in the graph, required should be empty */
    ASSERT_TRUE(required.empty());
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelLabelSet)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* define label sets for comparison */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_I =
        {0, 1, 2, 3, 4, 5, 6};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_II =
        {1, 2};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_III =
        {0};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_IV =
        {4};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_V =
        {1, 2, 3};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_VI =
        {0, 2, 4, 6};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_VII =
        {1, 2, 4};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_VIII =
        {0, 1, 2, 3};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_IX =
        {1, 3, 5};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_X =
        {0, 6};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_XI =
        {1, 5};

    std::vector<std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>> lbl_sets =
        {
            lbl_set_I, lbl_set_II, lbl_set_III, lbl_set_IV,
            lbl_set_V, lbl_set_VI, lbl_set_VII, lbl_set_VIII,
            lbl_set_IX, lbl_set_X, lbl_set_XI
        };

    /* check computed label sets against golden data */
    const LabelSet<COSTTYPE, SIMDWIDTH> * lset =
        this->m_multilevel->get_level_label_set();

    for(luint_t i = 0; i < 11; ++i)
    {
        /* assemble label set for this node */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> cmp_set;

        const _iv_st<COSTTYPE, SIMDWIDTH> lset_size =
            lset->label_set_size(i);
        _iv_st<COSTTYPE, SIMDWIDTH> j;
        for(j = 0; j < lset_size; ++j)
            cmp_set.push_back(lset->label_from_offset(i, j));

        /* compare label set */
        ASSERT_TRUE(cmp_vector(cmp_set, lbl_sets[i]));
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelUnaryCosts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* define cost vectors for comparison */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_I =
        {31, 36, 41, 46, 51, 56, 61};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_II =
        {9, 11};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_III =
        {65};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_IV =
        {143};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_V =
        {171, 176, 181};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_VI =
        {87, 93, 99, 105};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_VII =
        {47, 49, 53};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_VIII =
        {0, 1, 2, 3};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_IX =
        {254, 264, 274};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_X =
        {0, 6};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_XI =
        {136, 148};

    std::vector<std::vector<_s_t<COSTTYPE, SIMDWIDTH>>> unary_set =
        {
            unary_I, unary_II, unary_III, unary_IV,
            unary_V, unary_VI, unary_VII, unary_VIII,
            unary_IX, unary_X, unary_XI
        };

    /* check computed unary costs against golden data */
    const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle =
        this->m_multilevel->get_level_cost_bundle();
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set =
        this->m_multilevel->get_level_label_set();

    for(luint_t n = 0; n < 11; ++n)
    {
        const UnaryCosts<COSTTYPE, SIMDWIDTH> * unaries =
            cbundle->get_unary_costs(n);

        std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(8);

        const luint_t lset_size = label_set->label_set_size(n);
        const luint_t chunk = SIMDWIDTH * DIV_UP(lset_size, SIMDWIDTH);

        _v_t<COSTTYPE, SIMDWIDTH> c;
        for(luint_t i = 0; i < chunk; i += SIMDWIDTH)
        {
            c = unaries->get_unary_costs_enum_offset(i);
            v_store<COSTTYPE, SIMDWIDTH>(c, &costs[i]);
        }

        /* compare costs */
        ASSERT_TRUE(cmp_vector(costs, unary_set[n], lset_size));
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelPairwiseCosts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    typedef std::tuple<luint_t, luint_t, luint_t> eweight;

    /* save node-dependent factors for superedges */
    std::vector<eweight> weights =
        {
            eweight(0, 1, 5), eweight(1, 3, 12),
            eweight(1, 2, 23), eweight(2, 3, 78),
            eweight(0, 3, 25), eweight(3, 4, 27),
            eweight(0, 4, 38), eweight(4, 7, 93),
            eweight(4, 5, 39), eweight(3, 5, 70),
            eweight(5, 6, 43), eweight(5, 7, 51),
            eweight(4, 8, 54), eweight(5, 8, 101),
            eweight(8, 9, 55), eweight(6, 9, 50),
            eweight(6, 10, 52), eweight(8, 10, 67),
            eweight(9, 10, 119), eweight(3, 6, 38),
            eweight(2, 6, 40), eweight(7, 8, 56)
        };

    /* compare node-dependent anti-potts model with c = 1 */
    const Graph<COSTTYPE> * graph =
        this->m_multilevel->get_level_graph();
    const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle =
        this->m_multilevel->get_level_cost_bundle();
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set =
        this->m_multilevel->get_level_label_set();

    _iv_t<COSTTYPE, SIMDWIDTH> l_a, l_b;
    _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> ll_a[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> ll_b[SIMDWIDTH];
    _v_t<COSTTYPE, SIMDWIDTH> c;

    /* compare costs per (super-)edge */
    for(luint_t e_id = 0; e_id < graph->num_edges(); ++e_id)
    {
        const GraphEdge<COSTTYPE>& e = graph->edges()[e_id];
        const PairwiseCosts<COSTTYPE, SIMDWIDTH> * pairwise =
            cbundle->get_pairwise_costs(e_id);

        const luint_t n_a = e.node_a;
        const luint_t n_b = e.node_b;

        luint_t weight = invalid_luint_t;
        for(const eweight& ew : weights)
            if(std::get<0>(ew) == n_a && std::get<1>(ew) == n_b)
                weight = std::get<2>(ew);

        ASSERT_NE(weight, invalid_luint_t);
        const _s_t<COSTTYPE, SIMDWIDTH> se_weight =
            (_s_t<COSTTYPE, SIMDWIDTH>) weight;

        const _iv_st<COSTTYPE, SIMDWIDTH> ls_a = label_set->label_set_size(n_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> ls_b = label_set->label_set_size(n_b);

        _iv_st<COSTTYPE, SIMDWIDTH> li_a, li_b;
        for(li_a = 0; li_a < ls_a; ++li_a)
        {
            l_a = iv_init<COSTTYPE, SIMDWIDTH>(
                    label_set->label_from_offset(n_a, li_a));

            for(li_b = 0; li_b < ls_b; ++li_b)
            {
                l_b = iv_init<COSTTYPE, SIMDWIDTH>(
                    label_set->label_from_offset(n_b, li_b));
                c = pairwise->get_pairwise_costs(l_a, l_b);
                v_store<COSTTYPE, SIMDWIDTH>(c, tmp);
                iv_store<COSTTYPE, SIMDWIDTH>(l_a, ll_a);
                iv_store<COSTTYPE, SIMDWIDTH>(l_b, ll_b);

                const _s_t<COSTTYPE, SIMDWIDTH> ref_costs =
                    se_weight * (ll_a[0] == ll_b[0]);

                ASSERT_NEAR(tmp[0], ref_costs, 1e-6);
            }
        }
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestFirstLevelReprojection)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* take a labelling (indices!) for the first level... */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> labelling_first =
    {
        3, 1, 0, 0, /* 3, 2, 0, 4 */
        0, 2, 1, 0, /* 1, 4, 2, 0 */
        2, 1, 1 /* 5, 6, 5 */
    };

    /* and compare its reprojection to the golden original labelling */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> labelling_original =
    {
        3, 3, 3, 2, /* 3, 3, 3, 2 */
        1, 0, 3, 3, /* 2, 0, 3, 3 */
        4, 3, 0, 0, /* 4, 4, 0, 0 */
        1, 1, 2, 1, /* 1, 1, 4, 4 */
        0, 0, 0, 0, /* 4, 0, 1, 1 */
        2, 3, 2, 1, /* 4, 4, 2, 2 */
        1, 0, 3, 5, /* 1, 0, 4, 5 */
        1, 3, 3, 4, /* 6, 5, 5, 5 */
        3, 4, 4, 2 /* 5, 5, 5, 5 */
    };

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> reprojection;
    this->m_multilevel->reproject_solution(labelling_first, reprojection);

    /* check equivalence */
    ASSERT_TRUE(cmp_vector(reprojection, labelling_original, 36));
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelNumNodes)
{
    typedef typename TypeParam::Type COSTTYPE;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    ASSERT_EQ(graph->num_nodes(), 7);
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelNumEdges)
{
    typedef typename TypeParam::Type COSTTYPE;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    ASSERT_EQ(graph->edges().size(), 11);
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelEdges)
{
    typedef typename TypeParam::Type COSTTYPE;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    typedef std::tuple<luint_t, luint_t, luint_t> edge;
    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();

    /**
     * check if all edges are there (and no more) -- for node-dependent costs,
     * all weights must be one, since costs are summed up per edge
     */
    std::list<edge> required =
    {
        edge(0, 1, 1), edge(0, 2, 1), edge(0, 3, 1),
        edge(1, 2, 1), edge(2, 3, 1), edge(3, 4, 1),
        edge(3, 5, 1), edge(3, 6, 1), edge(4, 5, 1),
        edge(4, 6, 1), edge(5, 6, 1)
    };

    /* necessary condition: equal size */
    ASSERT_EQ(graph->edges().size(), required.size());

    for(const GraphEdge<COSTTYPE>& e : graph->edges())
    {
        edge te = edge(e.node_a, e.node_b, e.weight);

        /* try to find edge in required list */
        auto it = std::find(required.begin(), required.end(), te);
        ASSERT_NE(it, required.end());

        required.erase(it);
    }

    /* after removing all edges in the graph, required should be empty */
    ASSERT_TRUE(required.empty());
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelLabelSet)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_0 =
        {4};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_1 =
        {1, 2};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_2 =
        {0};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_3 =
        {2};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_4 =
        {1, 3, 5};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_5 =
        {0, 6};
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> lbl_set_6 =
        {1, 5};

    std::vector<std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>> lbl_sets =
        {
            lbl_set_0, lbl_set_1, lbl_set_2, lbl_set_3,
            lbl_set_4, lbl_set_5, lbl_set_6
        };

    /* check computed label sets against golden data */
    const LabelSet<COSTTYPE, SIMDWIDTH> * lset =
        this->m_multilevel->get_level_label_set();

    for(luint_t i = 0; i < 7; ++i)
    {
        /* assemble label set for this node */
        std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> cmp_set;

        const _iv_st<COSTTYPE, SIMDWIDTH> lset_size =
            lset->label_set_size(i);
        _iv_st<COSTTYPE, SIMDWIDTH> j;
        for(j = 0; j < lset_size; ++j)
            cmp_set.push_back(lset->label_from_offset(i, j));

        /* compare label set */
        ASSERT_TRUE(cmp_vector(cmp_set, lbl_sets[i]));
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelUnaryCosts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    /* golden cost data */
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_0 = {219};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_1 = {9, 11};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_2 = {65};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_3 = {546};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_4 = {254, 264, 274};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_5 = {0, 6};
    std::vector<_s_t<COSTTYPE, SIMDWIDTH>> unary_6 = {136, 148};

    std::vector<std::vector<_s_t<COSTTYPE, SIMDWIDTH>>> golden_unaries =
    {
        unary_0, unary_1, unary_2, unary_3,
        unary_4, unary_5, unary_6
    };

    /* check computed unary costs against golden data */
    const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle =
        this->m_multilevel->get_level_cost_bundle();
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set =
        this->m_multilevel->get_level_label_set();

    for(luint_t n = 0; n < 7; ++n)
    {
        const UnaryCosts<COSTTYPE, SIMDWIDTH> * unaries =
            cbundle->get_unary_costs(n);

        const luint_t lset_size = label_set->label_set_size(n);
        const luint_t chunk = SIMDWIDTH * DIV_UP(lset_size, SIMDWIDTH);

        std::vector<_s_t<COSTTYPE, SIMDWIDTH>> costs(chunk);

        _v_t<COSTTYPE, SIMDWIDTH> c;
        for(luint_t i = 0; i < chunk; i += SIMDWIDTH)
        {
            c = unaries->get_unary_costs_enum_offset(i);
            v_store<COSTTYPE, SIMDWIDTH>(c, &costs[i]);
        }

        /* compare costs */
        ASSERT_TRUE(cmp_vector(costs, golden_unaries[n], lset_size));
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelPairwiseCosts)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    typedef std::tuple<luint_t, luint_t, _s_t<COSTTYPE, SIMDWIDTH>> edge;
    const Graph<COSTTYPE> * graph = this->m_multilevel->get_level_graph();
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set = this->m_multilevel->
        get_level_label_set();
    const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle = this->m_multilevel->
        get_level_cost_bundle();

    /* golden data for edge costs */
    std::vector<edge> golden_pairwise =
    {
        edge(0, 1, 5), edge(0, 2, 78),
        edge(0, 3, 183), edge(1, 2, 23),
        edge(2, 3, 40), edge(3, 4, 211),
        edge(3, 5, 50), edge(3, 6, 52),
        edge(4, 5, 55), edge(4, 6, 67),
        edge(5, 6, 119)
    };

    /* check edge cost against golden data */
    /* necessary condition: equal size */
    ASSERT_EQ(graph->edges().size(), golden_pairwise.size());

    _iv_st<COSTTYPE, SIMDWIDTH> tmpi1[SIMDWIDTH], tmpi2[SIMDWIDTH];
    _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
    for(luint_t e_id = 0; e_id < graph->num_edges(); ++e_id)
    {
        const GraphEdge<COSTTYPE>& e = graph->edges()[e_id];

        /* find edge's cost factor */
        _s_t<COSTTYPE, SIMDWIDTH> costfactor = -1;
        for(const edge& golden_e : golden_pairwise)
            if(std::get<0>(golden_e) == e.node_a &&
                std::get<1>(golden_e) == e.node_b)
                costfactor = std::get<2>(golden_e);

        ASSERT_NE(costfactor, -1);

        const _iv_st<COSTTYPE, SIMDWIDTH> lset_size_1 =
            label_set->label_set_size(e.node_a);
        const _iv_st<COSTTYPE, SIMDWIDTH> lset_size_2 =
            label_set->label_set_size(e.node_b);

        const PairwiseCosts<COSTTYPE, SIMDWIDTH> * pcosts =
            cbundle->get_pairwise_costs(e_id);

        _iv_st<COSTTYPE, SIMDWIDTH> l1_i, l2_i;
        _iv_t<COSTTYPE, SIMDWIDTH> l1, l2;
        _v_t<COSTTYPE, SIMDWIDTH> c;
        for(l1_i = 0; l1_i < lset_size_1; ++l1_i)
        {
            l1 = iv_init<COSTTYPE, SIMDWIDTH>(
                label_set->label_from_offset(e.node_a, l1_i));

            for(l2_i = 0; l2_i < lset_size_2; ++l2_i)
            {
                l2 = iv_init<COSTTYPE, SIMDWIDTH>(
                    label_set->label_from_offset(e.node_b, l2_i));
                c = pcosts->get_pairwise_costs(l1, l2);

                v_store<COSTTYPE, SIMDWIDTH>(c, tmp);
                iv_store<COSTTYPE, SIMDWIDTH>(l1, tmpi1);
                iv_store<COSTTYPE, SIMDWIDTH>(l2, tmpi2);

                /* represent sum of L1 antipotts costs */
                const COSTTYPE res_costs = costfactor * (tmpi1[0] == tmpi2[0]);
                ASSERT_NEAR(tmp[0], res_costs, 0.01);
            }
        }
    }
}

TYPED_TEST_P(mapMAPTestMultilevel, TestSecondLevelReprojection)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* create second level */
    this->m_multilevel->next_level(this->m_second_labelling,
        this->m_dep_labelling);

    /* take a labelling (indices!) for the second level... */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> labelling_second =
    {
        0, 1, 0, 0, /* 4, 2, 0, 2 */
        1, 1, 1 /* 3, 6, 5 */
    };

    /* and compare its reprojection to the golden original labelling */
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> labelling_original =
    {
        4, 4, 4, 2, /* 4, 4, 4, 2 */
        1, 0, 4, 4, /* 2, 0, 4, 4 */
        4, 3, 0, 0, /* 4, 4, 0, 0 */
        2, 2, 2, 1, /* 2, 2, 4, 4 */
        0, 0, 1, 1, /* 4, 0, 2, 2 */
        1, 2, 2, 1, /* 2, 2, 2, 2 */
        2, 2, 1, 3, /* 2, 2, 2, 3 */
        1, 3, 2, 2, /* 6, 5, 3, 3 */
        2, 2, 4, 2 /* 3, 3, 5, 5 */
    };

    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> reprojection;
    this->m_multilevel->reproject_solution(labelling_second, reprojection);

    /* check equivalence */
    ASSERT_TRUE(cmp_vector(reprojection, labelling_original, 36));
}

/* register test cases */
REGISTER_TYPED_TEST_CASE_P(mapMAPTestMultilevel,
    TestFirstLevelNumNodes,
    TestFirstLevelNumEdges,
    TestFirstLevelEdges,
    TestFirstLevelLabelSet,
    TestFirstLevelUnaryCosts,
    TestFirstLevelPairwiseCosts,
    TestFirstLevelReprojection,
    TestSecondLevelNumNodes,
    TestSecondLevelNumEdges,
    TestSecondLevelEdges,
    TestSecondLevelLabelSet,
    TestSecondLevelUnaryCosts,
    TestSecondLevelPairwiseCosts,
    TestSecondLevelReprojection);

/* instantiate tests */
typedef ::testing::Types<
    // #if defined(__SSE4_2__)
    // TestTuple<float, 4>,
    // #endif /* __SSE4_2__ */
    // #if defined(__AVX__)
    // TestTuple<float, 8>,
    // #endif /* __AVX__ */
    // #if defined(__SSE4_2__)
    // TestTuple<double, 2>,
    // #endif /* __SSE4_2__ */
    // #if defined(__AVX__)
    // TestTuple<double, 4>,
    // #endif /* __AVX__ */
    TestTuple<float, 1>,
    TestTuple<double, 1>
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(MultilevelTest,
    mapMAPTestMultilevel, TestTupleInstances);


NS_MAPMAP_END