/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_HEADER_COSTS_H_
#define __MAPMAP_HEADER_COSTS_H_

#include <memory>
#include <exception>
#include <string>
#include <vector>
#include <limits>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

/**
 * Represents sparse label tables, independent from costs. Each node
 * has its assigned label set (can be shared between nodes), represented
 * by two lists with label IDs and actual labels.
 */
template<typename COSTTYPE, uint_t SIMDWIDTH>
class LabelSet
{
public:
    LabelSet(const luint_t& num_graph_nodes, const bool compress);
    ~LabelSet();

    /**
     * Retrieval functions for optimization
     */
    const _iv_st<COSTTYPE, SIMDWIDTH> max_label_set_size() const;
    const _iv_st<COSTTYPE, SIMDWIDTH> max_label() const;
    const _iv_st<COSTTYPE, SIMDWIDTH> label_set_size(const luint_t& node_id)
        const;
    _iv_t<COSTTYPE, SIMDWIDTH> labels_from_offset(const luint_t& node_id, 
        const _iv_st<COSTTYPE, SIMDWIDTH>& offset) const;
    _iv_st<COSTTYPE, SIMDWIDTH> label_from_offset(const luint_t& node_id,
        const _iv_st<COSTTYPE, SIMDWIDTH>& offset) const;
    const _iv_st<COSTTYPE, SIMDWIDTH> offset_for_label(const luint_t& node_id,
        const _iv_st<COSTTYPE, SIMDWIDTH>& offset) const;

    /**
     * Label set construction functions.
     */
    void set_label_set_for_node(const luint_t& node_id,
        const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& label_set);

protected:
    const uint_t hash(
        const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& label_set);

protected:
    luint_t m_graph_num_nodes;
    _iv_st<COSTTYPE, SIMDWIDTH> m_max_label_set_size;
    _iv_st<COSTTYPE, SIMDWIDTH> m_max_label;

    /**
     * Note: for aligned vector access, all addresses must 
     * be aligned.
     * 
     * - For most compilers, all std::vectors are aligned. 
     */ 
    std::vector<std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>> m_label_sets;
    std::vector<luint_t> m_label_set_ids;
    std::vector<uint_t> m_label_set_hashes;
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>> m_label_set_sizes;

    bool m_compress = false;
    bool m_mutable = true;
};

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
class UnaryCosts
{
public:
    virtual ~UnaryCosts() {};

    /* enumerable: pass indices instead of labels */
    virtual bool supports_enumerable_costs() const = 0;

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs(const luint_t& node_id, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec) const throw() = 0;
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_unary_costs_enum_offset(
        const luint_t& node_id, const _iv_st<COSTTYPE, SIMDWIDTH>& offset) 
        const throw() = 0;
};

/******************************************************************************/

template<typename COSTTYPE, uint_t SIMDWIDTH>
class PairwiseCosts
{
public:
    virtual ~PairwiseCosts() {};

    virtual bool node_dependent() const = 0; 

    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(const luint_t& node_id_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, const luint_t& node_id_2, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw() = 0;
    virtual _v_t<COSTTYPE, SIMDWIDTH> get_binary_costs(
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1, 
        const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2) const throw() = 0;
};

/******************************************************************************/

class ModeNotSupportedException : public std::exception
{
public:
    ModeNotSupportedException(const char* err_msg);
    ModeNotSupportedException(const std::string& err_msg);
    ~ModeNotSupportedException() throw();

    const char* what() const throw();

protected:
    std::string m_err_msg;
};

NS_MAPMAP_END

#include "source/costs.impl.h"

#endif /* __MAPMAP_HEADER_COSTS_H_ */
