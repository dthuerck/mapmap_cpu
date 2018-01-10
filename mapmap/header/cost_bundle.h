/**
 * Copyright (C) 2017, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_COST_BUNDLE_H_
#define __MAPMAP_HEADER_COST_BUNDLE_H_

#include "header/defines.h"
#include "header/costs.h"
#include "header/graph.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class CostBundle
{
public:
    CostBundle(const Graph<COSTTYPE> * graph);
    ~CostBundle();

    /* global cost setting */
    void set_unary_costs(const UnaryCosts<COSTTYPE, SIMDWIDTH> * costs);
    void set_pairwise_costs(const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs);

    /* individual cost setting */
    void set_unary_costs(const luint_t node_id, 
        const UnaryCosts<COSTTYPE, SIMDWIDTH> * costs);
    void set_pairwise_costs(const luint_t edge_id, 
        const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs);
    
    /* individual cost retrieval */
    const UnaryCosts<COSTTYPE, SIMDWIDTH> * get_unary_costs(const luint_t 
        node_id) const;
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * get_pairwise_costs(const luint_t
        edge_id) const;

protected:
    const Graph<COSTTYPE> * m_graph;

    std::vector<const UnaryCosts<COSTTYPE, SIMDWIDTH>*> m_unary;
    std::vector<const PairwiseCosts<COSTTYPE, SIMDWIDTH>*> m_pairwise;
};

NS_MAPMAP_END

/* include template instances */
#include "source/cost_bundle.impl.h"

#endif /* __MAPMAP_HEADER_COST_BUNDLE_H_ */