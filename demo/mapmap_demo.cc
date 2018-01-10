/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
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
    int num_threads = 8;

    if(argc > 1)
        num_threads = std::atoi(argv[1]);
    std::cout << "Using number of threads: " << num_threads << std::endl;

    /* TBB control */
    tbb::task_scheduler_init schedule(num_threads);

    /* mapMAP template parameters */
    using cost_t = float;
    const uint_t simd_w = sys_max_simd_width<cost_t>();
    std::cout << "Using SIMD width = " << simd_w << std::endl;

    /* cost types */
    using unary_t = UnaryTable<cost_t, simd_w>;
    using pairwise_t = PairwiseTruncatedLinear<cost_t, simd_w>;

    /* path to dataset */
    uint32_t num_nodes, num_edges, num_labels, num_label_sets,
        use_label_costs, use_ptx;
    const char * mrf_path = "../demo/planesweep_320_256_96.bin";

    /* pointer to data structures */
    std::unique_ptr<Graph<cost_t>> graph;
    std::unique_ptr<LabelSet<cost_t, simd_w>> label_set;

    std::vector<std::unique_ptr<unary_t>> unaries;
    std::unique_ptr<pairwise_t> pairwise;

    /* termination criterion and control flow */
    std::unique_ptr<TerminationCriterion<cost_t, simd_w>> terminate;
    mapMAP_control ctr;

    /* solver instance */
    mapMAP<cost_t, simd_w> mapmap;

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

        std::cout << "Nodes: " << num_nodes << std::endl;
        std::cout << "Edges: " << num_edges << std::endl;
        std::cout << "Labels: " << num_labels << std::endl;
        std::cout << "Number of label sets: " << num_label_sets << std::endl;
        std::cout << "Use label costs: " << use_label_costs << std::endl;

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

        /* construct unary cost tables */
        unaries.reserve(num_nodes);

        for(uint32_t n = 0; n < num_nodes; ++n)
        {
            unaries.emplace_back(std::unique_ptr<unary_t>(new unary_t(
                n, label_set.get())));

            const uint32_t n_labels = tmp_node_label_counts[n];
            std::vector<_s_t<cost_t, simd_w>> costs(n_labels);

            for(uint32_t l = 0; l < n_labels; ++l)
                costs[l] = tmp_unary_table[tmp_node_unary_offsets[n] + l];

            unaries.back()->set_costs(costs);
        }

        /* construct pairwise costs */
        pairwise = std::unique_ptr<pairwise_t>(new pairwise_t({1.0, 2.0}));

        /* skip extensions and PTX files */
        io.close();

        /* create termination criterion */
        terminate = std::unique_ptr<TerminationCriterion<cost_t, simd_w>>(
            new StopWhenReturnsDiminish<cost_t, simd_w>(5, 0.0001));

        /* create (optional) control flow settings */
        ctr.use_multilevel = true;
        ctr.use_spanning_tree = true;
        ctr.use_acyclic = true;
        ctr.spanning_tree_multilevel_after_n_iterations = 5;
        ctr.force_acyclic = true;
        ctr.min_acyclic_iterations = 5;
        ctr.relax_acyclic_maximal = true;
        ctr.tree_algorithm = LOCK_FREE_TREE_SAMPLER;

        /* set to true and select a seed for (serial) deterministic sampling */
        ctr.sample_deterministic = false;
        ctr.initial_seed = 548923723;

        /* construct optimizer */
        mapmap.set_graph(graph.get());
        mapmap.set_label_set(label_set.get());
        for(luint_t n = 0; n < num_nodes; ++n)
            mapmap.set_unary(n, unaries[n].get());
        mapmap.set_pairwise(pairwise.get());
        mapmap.set_termination_criterion(terminate.get());

        std::cout << "Finished loading dataset." << std::endl;
    }
    else
    {
        /* no MRF? Abort... */
        std::cout << UNIX_COLOR_RED
                  << "Could not open MRF "
                  << UNIX_COLOR_WHITE
                  << mrf_path
                  << UNIX_COLOR_RED
                  << ", exiting..."
                  << UNIX_COLOR_RESET
                  << std::endl;

        return EXIT_FAILURE;
    }

    /* use standard multilevel and termination criterion and start */
    std::vector<_iv_st<cost_t, simd_w>> solution;

    /* catch errors thrown during optimization */
    try
    {
        mapmap.optimize(solution, ctr);
    }
    catch(std::runtime_error& e)
    {
        std::cout << UNIX_COLOR_RED
                  << "Caught an exception: "
                  << UNIX_COLOR_WHITE
                  << e.what()
                  << UNIX_COLOR_RED
                  << ", exiting..."
                  << UNIX_COLOR_RESET
                  << std::endl;
    }
    catch(std::domain_error& e)
    {
        std::cout << UNIX_COLOR_RED
                  << "Caught an exception: "
                  << UNIX_COLOR_WHITE
                  << e.what()
                  << UNIX_COLOR_RED
                  << ", exiting..."
                  << UNIX_COLOR_RESET
                  << std::endl;
    }

    /* Extract lables from solution (vector of label indices) */
    std::vector<_iv_st<cost_t, simd_w>> labeling(num_nodes);
    for(uint32_t n = 0; n < num_nodes; ++n)
    {
        labeling[n] = label_set->label_from_offset(n, solution[n]);
    }

    return EXIT_SUCCESS;
}
