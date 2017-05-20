/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_COST_INSTANCES_PAIRWISE_ANTIPOTTS_H_
#define __MAPMAP_COST_INSTANCES_PAIRWISE_ANTIPOTTS_H_

#include "header/costs.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwiseAntipotts : public PairwiseCosts<COSTTYPE, SIMDWIDTH>
{
public:
    PairwiseAntipotts();
    PairwiseAntipotts(const _s_t<COSTTYPE, SIMDWIDTH>& c);
    ~PairwiseAntipotts();

    virtual bool node_dependent() const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(
        const luint_t& node_id_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
        const luint_t& node_id_2, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw();
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw();
    
protected: 
    COSTTYPE m_c = (COSTTYPE) 1;

    _v_t<COSTTYPE, SIMDWIDTH> m_full_cost;
};

NS_MAPMAP_END

#include "source/cost_instances/pairwise_antipotts.impl.h"

#endif /* __MAPMAP_COST_INSTANCES_PAIRWISE_ANTIPOTTS_H_ */
