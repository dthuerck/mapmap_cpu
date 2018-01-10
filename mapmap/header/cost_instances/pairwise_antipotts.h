/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_PAIRWISE_ANTIPOTTS_H_
#define __MAPMAP_PAIRWISE_ANTIPOTTS_H_

#include "header/costs.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwiseAntipotts : public PairwiseCosts<COSTTYPE, SIMDWIDTH>
{
public:
    PairwiseAntipotts();
    PairwiseAntipotts(const _s_t<COSTTYPE, SIMDWIDTH>& c);
    PairwiseAntipotts(
            const std::initializer_list<_s_t<COSTTYPE, SIMDWIDTH>>& ps);
    ~PairwiseAntipotts();

    _s_t<COSTTYPE, SIMDWIDTH> get_c() const;
    _s_t<COSTTYPE, SIMDWIDTH> get_label_diff_cap() const;

    virtual std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> copy() const;

    virtual bool supports_enumerable_costs() const;
    virtual bool eq(const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs) const;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_pairwise_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const;
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_pairwise_costs_enum_offsets(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_1,
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_ix_vec_2) const;

protected:
    _s_t<COSTTYPE, SIMDWIDTH> m_c = (_s_t<COSTTYPE, SIMDWIDTH>) 1;
};

NS_MAPMAP_END

/* include function implementations */
#include "source/cost_instances/pairwise_antipotts.impl.h"

#endif /* __MAPMAP_PAIRWISE_ANTIPOTTS_H_ */