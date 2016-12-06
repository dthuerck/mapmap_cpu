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
#include "header/costs.h"
#include "header/defines.h"
#include "header/dynamic_programming.h"
#include "header/graph.h"
#include "header/mapmap.h"
#include "header/multilevel.h"
#include "header/parallel_templates.h"
#include "header/termination_criterion.h"
#include "header/tree.h"
#include "header/tree_optimizer.h"
#include "header/tree_sampler.h"
#include "header/vector_math.h"
#include "header/vector_types.h"

/* cost functions */
#include "header/cost_instances/pairwise_antipotts.h"
#include "header/cost_instances/pairwise_potts.h"
#include "header/cost_instances/pairwise_table.h"
#include "header/cost_instances/pairwise_truncated_linear.h"
#include "header/cost_instances/unary_table.h"

/* termination criteria */
#include "header/termination_instances/stop_after_iterations.h"
#include "header/termination_instances/stop_after_time.h"
#include "header/termination_instances/stop_when_flat.h"

/* node grouping criteria */
#include "header/multilevel_instances/group_same_label.h"

#endif /* __MAPMAP_FULL_H_ */