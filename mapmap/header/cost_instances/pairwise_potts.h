/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_COST_INSTANCES_PAIRWISE_POTTS_H_
#define __MAPMAP_COST_INSTANCES_PAIRWISE_POTTS_H_

#include "header/costs.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwisePotts : public PairwiseCosts<COSTTYPE, SIMDWIDTH>
{
public:
    PairwisePotts();
    PairwisePotts(const _s_t<COSTTYPE, SIMDWIDTH>& c);
    ~PairwisePotts();

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
    _s_t<COSTTYPE, SIMDWIDTH> m_c = (_s_t<COSTTYPE, SIMDWIDTH>) 1;
};

template<typename COSTTYPE, uint_t SIMDWIDTH>
using PairwisePotts_ptr = std::shared_ptr<PairwisePotts<COSTTYPE, SIMDWIDTH>>; 

NS_MAPMAP_END

#include "source/cost_instances/pairwise_potts.impl.h"

#endif /* __MAPMAP_COST_INSTANCES_PAIRWISE_POTTS_H_ */
