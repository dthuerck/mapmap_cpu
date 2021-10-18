/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_FULL_H_
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#define __MAPMAP_FULL_H_

/* basic classes */
#include <mapmap/header/color.h>
#include <mapmap/header/costs.h>
#include <mapmap/header/defines.h>
#include <mapmap/header/graph.h>
#include <mapmap/header/mapmap.h>
#include <mapmap/header/multilevel.h>
#include <mapmap/header/parallel_templates.h>
#include <mapmap/header/termination_criterion.h>
#include <mapmap/header/timer.h>
#include <mapmap/header/tree_optimizer.h>
#include <mapmap/header/tree_sampler.h>
#include <mapmap/header/tree.h>
#include <mapmap/header/vector_math.h>
#include <mapmap/header/vector_types.h>

/* cost functions */
#include <mapmap/header/cost_instances/pairwise_antipotts.h>
#include <mapmap/header/cost_instances/pairwise_linear_peak.h>
#include <mapmap/header/cost_instances/pairwise_potts.h>
#include <mapmap/header/cost_instances/pairwise_table.h>
#include <mapmap/header/cost_instances/pairwise_truncated_linear.h>
#include <mapmap/header/cost_instances/pairwise_truncated_quadratic.h>
#include <mapmap/header/cost_instances/unary_table.h>

/* multilevel instances */
#include <mapmap/header/multilevel_instances/group_same_label.h>

/* optimizer instances */
#include <mapmap/header/optimizer_instances/dp_node_solver_factory.h>
#include <mapmap/header/optimizer_instances/dp_node_solver.h>
#include <mapmap/header/optimizer_instances/dp_node.h>
#include <mapmap/header/optimizer_instances/dynamic_programming.h>
#include <mapmap/header/optimizer_instances/envelope.h>

/* termination criteria */
#include <mapmap/header/termination_instances/stop_after_iterations.h>
#include <mapmap/header/termination_instances/stop_after_time.h>
#include <mapmap/header/termination_instances/stop_when_flat.h>
#include <mapmap/header/termination_instances/stop_when_returns_diminish.h>

/* tree sampler instances */
#include <mapmap/header/tree_sampler_instances/lock_free_tree_sampler.h>
#include <mapmap/header/tree_sampler_instances/optimistic_tree_sampler.h>

#endif /* __MAPMAP_FULL_H_ */