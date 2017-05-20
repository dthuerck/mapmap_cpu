/**
 * Copyright (C) 2016, Daniel Thuerck
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

/**
 * This file is meant for demos purposes - it allows the execution of mapMAP
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

    /* path to dataset */
    uint32_t num_nodes, num_edges, num_labels, num_label_sets,
        use_label_costs, use_ptx;

    /* pointer to data structures */
    std::unique_ptr<Graph<cost_t>> graph;

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
        const luint_t warmup_iterations = 1;
        const luint_t benchmark_iterations = 1;
        for(luint_t it = 0; it < warmup_iterations + benchmark_iterations; ++it)
        {
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
        }

        acyclic_time /= benchmark_iterations;

        std::cout << "Coloring time: " << color_time << std::endl;
        std::cout << "Acyclic time: " << acyclic_time << std::endl;
    }

    return EXIT_SUCCESS;
}
