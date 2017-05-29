/**
 * Copyright (C) 2016, Daniel Thuerck
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

/**
 * This file is meant for demo purposes - it allows the execution of mapMAP
 * on datasets stored in the (deprecated) binary SHIMF format. mapMAP is
 * used as it would be if used as a library; pairwise costs must be
 * set manually.
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <memory>

#include "mapmap/full.h"

#include "tbb/task_scheduler_init.h"

using namespace NS_MAPMAP;

int
main(
    int argc,
    char * argv[])
{
    /* retrieve console parameter -> number of threads */
    int num_threads = 32;
    const luint_t num_roots = 128;
    const char * mrf_path = "../demo/planesweep_320_256_96.bin";
    //const TREE_SAMPLER_ALGORITHM ts_algo = OPTIMISTIC_TREE_SAMPLER;
    const TREE_SAMPLER_ALGORITHM ts_algo = LOCK_FREE_TREE_SAMPLER;

    std::cout << "Using dataset: " << mrf_path << std::endl;

    if(argc > 1)
        num_threads = std::atoi(argv[1]);
    std::cout << "Using number of threads = " << num_threads << std::endl;

    /* TBB control */
    tbb::task_scheduler_init schedule(num_threads);
    std::cout << "Schedule active = " << schedule.is_active() << std::endl;

    /* mapMAP template parameters */
    using cost_t = float;
    const uint_t simd_w = sys_max_simd_width<cost_t>();
    std::cout << "Using SIMD width = " << simd_w << std::endl;

    using unary_t = UnaryTable<cost_t, simd_w>;
    using pairwise_t = PairwiseTruncatedLinear<cost_t, simd_w>;

    /* path to dataset */
    uint32_t num_nodes, num_edges, num_labels, num_label_sets,
        use_label_costs, use_ptx;

    /* pointer to data structures */
    std::unique_ptr<Graph<cost_t>> graph;
    std::unique_ptr<LabelSet<cost_t, simd_w>> label_set;

    std::unique_ptr<unary_t> unaries;
    std::unique_ptr<pairwise_t> pairwise;

    /* read packed dataset */
    std::ifstream io(mrf_path, std::ifstream::in | std::ifstream::binary);

    if(io.good())
    {
        std::cout << "Start loading dataset..." << std::endl;
        io.read((char *) &num_nodes, sizeof(uint32_t));
        io.read((char *) &num_edges, sizeof(uint32_t));
        io.read((char *) &num_labels, sizeof(uint32_t));
        io.read((char *) &num_label_sets, sizeof(uint32_t));

        /* skip number of extensions */
        io.seekg(sizeof(uint32_t), std::ifstream::cur);

        io.read((char *) &use_label_costs, sizeof(uint32_t));
        io.read((char *) &use_ptx, sizeof(uint32_t));

        std::cout << "Nodes = " << num_nodes << std::endl;
        std::cout << "Edges = " << num_edges << std::endl;

        /* parse edges and construct graph */
        graph = std::unique_ptr<Graph<cost_t>>(new Graph<cost_t>(num_nodes));

        uint32_t min_id, max_id;
        float weight;
        for(uint32_t e_id = 0; e_id < num_edges; ++e_id)
        {
            io.read((char *) &min_id, sizeof(uint32_t));
            io.read((char *) &max_id, sizeof(uint32_t));
            io.read((char *) &weight, sizeof(float));

            graph->add_edge(min_id, max_id, weight);
        }

        /* finalize component lists for base graph */
        graph->update_components();

                /* skip label cost table */
        if(use_label_costs > 0u)
            io.seekg(num_labels * sizeof(float), std::ifstream::cur);

        /* read label set */
        uint32_t tmp_total_label_entries;
        io.read((char *) &tmp_total_label_entries, sizeof(uint32_t));

        std::vector<uint16_t> tmp_label_table(tmp_total_label_entries);
        std::vector<uint32_t> tmp_node_label_offsets(num_nodes);
        std::vector<uint32_t> tmp_node_label_counts(num_nodes);

        io.read((char *) &tmp_label_table[0], tmp_total_label_entries *
            sizeof(uint16_t));
        io.read((char *) &tmp_node_label_offsets[0], num_nodes *
            sizeof(uint32_t));
        io.read((char *) &tmp_node_label_counts[0], num_nodes *
            sizeof(uint32_t));

        /* construct and compress label set */
        label_set = std::unique_ptr<LabelSet<cost_t, simd_w>>(
            new LabelSet<cost_t, simd_w>(num_nodes, false));
        for(uint32_t n = 0; n < num_nodes; ++n)
        {
            const uint32_t n_labels = tmp_node_label_counts[n];

            std::vector<_iv_st<cost_t, simd_w>> lset(n_labels);
            for(uint32_t l = 0; l < n_labels; ++l)
                lset[l] = tmp_label_table[tmp_node_label_offsets[n] + l];

            label_set->set_label_set_for_node(n, lset);
        }

        /* read unary cost table */
        uint32_t tmp_total_unary_entries;
        io.read((char *) &tmp_total_unary_entries, sizeof(uint32_t));

        std::vector<uint32_t> tmp_node_unary_offsets(num_nodes);
        std::vector<float> tmp_unary_table(tmp_total_unary_entries);

        io.read((char *) &tmp_node_unary_offsets[0], num_nodes *
            sizeof(uint32_t));
        io.read((char *) &tmp_unary_table[0], tmp_total_unary_entries *
            sizeof(float));

        /* construct unary cost table */
        unaries = std::unique_ptr<unary_t>(new unary_t
            (graph.get(), label_set.get()));

        for(uint32_t n = 0; n < num_nodes; ++n)
        {
            const uint32_t n_labels = tmp_node_label_counts[n];
            std::vector<_s_t<cost_t, simd_w>> costs(n_labels);

            for(uint32_t l = 0; l < n_labels; ++l)
                costs[l] = tmp_unary_table[tmp_node_unary_offsets[n] + l];

            unaries->set_costs_for_node(n, costs);
        }

        for(luint_t n = 0; n < num_nodes; ++n)
            tmp_node_unary_offsets[n] = n * num_labels;

        /* pairwise costs */
        pairwise = std::unique_ptr<pairwise_t>(new pairwise_t(2));

        /* skip extensions and PTX files */
        io.close();

        /* initial solution: first label */
        std::vector<int_t> sol(graph->num_nodes());
        for(luint_t i = 0; i < graph->num_nodes(); ++i)
            sol[i] = label_set->label_from_offset(i, 0);

        std::cout << "Finished loading dataset..." << std::endl;

        /* color the graph */
        START_TIMER("Coloring");
        std::vector<luint_t> coloring;
        Color<cost_t> pen(*graph);
        pen.color_graph(coloring);
        graph->set_coloring(coloring);
        STOP_TIMER("Coloring");
        double color_time = GET_TIMER("Coloring");

        double acyclic_time = 0;
        double dp_time = 0;
        const luint_t warmup_iterations = 1;
        const luint_t benchmark_iterations = 5;

        for(luint_t it = 0; it < warmup_iterations + benchmark_iterations; ++it)
        {
            std::vector<int_t> sol2(sol.begin(), sol.end());

            START_TIMER("Acyclic");

            /* sample a tree (forest) without dependencies */
            std::unique_ptr<TreeSampler<cost_t, true>> sampler =
                InstanceFactory<cost_t, true>::get_sampler_instance(
                ts_algo, graph.get());

            std::vector<luint_t> roots;
            sampler->select_random_roots(num_roots, roots);

            /* grow trees in forest */
            std::unique_ptr<Tree<cost_t>> tree = sampler->sample(roots, true,
                true);

            STOP_TIMER("Acyclic");

            if(it >= warmup_iterations)
                acyclic_time += GET_TIMER("Acyclic");

            START_TIMER("DP");

            /* create optimizer */
            CombinatorialDynamicProgramming<cost_t, simd_w, unary_t,
                pairwise_t> opt;
            opt.set_graph(graph.get());
            opt.set_tree(tree.get());
            opt.set_label_set(label_set.get());
            opt.set_costs(unaries.get(), pairwise.get());
            opt.use_dependencies(sol);

            /* optimize! */
            opt.optimize(sol2);

            STOP_TIMER("DP");

            if(it >= warmup_iterations)
                dp_time += GET_TIMER("DP");
        }

        acyclic_time /= benchmark_iterations;

        std::cout << "Coloring time: " << color_time << std::endl;
        std::cout << "Acyclic time: " << acyclic_time << std::endl;
        std::cout << "DP time: " << dp_time << std::endl;
    }

    return EXIT_SUCCESS;
}
