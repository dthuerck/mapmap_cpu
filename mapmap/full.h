/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_FULL_H_
#define __MAPMAP_FULL_H_

/* basic classes */
#include "header/color.h"
#include "header/costs.h"
#include "header/defines.h"
#include "header/graph.h"
#include "header/mapmap.h"
#include "header/multilevel.h"
#include "header/parallel_templates.h"
#include "header/termination_criterion.h"
#include "header/timer.h"
#include "header/tree_optimizer.h"
#include "header/tree_sampler.h"
#include "header/tree.h"
#include "header/vector_math.h"
#include "header/vector_types.h"

/* cost functions */
#include "header/cost_instances/pairwise_antipotts.h"
#include "header/cost_instances/pairwise_linear_peak.h"
#include "header/cost_instances/pairwise_potts.h"
#include "header/cost_instances/pairwise_table.h"
#include "header/cost_instances/pairwise_truncated_linear.h"
#include "header/cost_instances/pairwise_truncated_quadratic.h"
#include "header/cost_instances/unary_table.h"

/* multilevel instances */
#include "header/multilevel_instances/group_same_label.h"

/* optimizer instances */
#include "header/optimizer_instances/dp_node_solver_factory.h"
#include "header/optimizer_instances/dp_node_solver.h"
#include "header/optimizer_instances/dp_node.h"
#include "header/optimizer_instances/dynamic_programming.h"
#include "header/optimizer_instances/envelope.h"

/* termination criteria */
#include "header/termination_instances/stop_after_iterations.h"
#include "header/termination_instances/stop_after_time.h"
#include "header/termination_instances/stop_when_flat.h"
#include "header/termination_instances/stop_when_returns_diminish.h"

/* tree sampler instances */
#include "header/tree_sampler_instances/lock_free_tree_sampler.h"
#include "header/tree_sampler_instances/optimistic_tree_sampler.h"

#endif /* __MAPMAP_FULL_H_ */