/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/cost_instances/pairwise_table.h"

#include <limits>
#include <iostream>

#include "header/parallel_templates.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
PairwiseTable(
    const luint_t node_a,
    const luint_t node_b,
    const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set)
: m_node_a(node_a),
  m_node_b(node_b),
  m_lbl_set(lbl_set)
{
    const luint_t padded_size = 
        lbl_set->label_set_size(node_a) *
        (DIV_UP(lbl_set->label_set_size(node_b), SIMDWIDTH) * SIMDWIDTH);
    m_packed_table_storage = std::unique_ptr<_s_t<COSTTYPE, SIMDWIDTH>[]>(new
        _s_t<COSTTYPE, SIMDWIDTH>[padded_size]);

    m_packed_table = m_packed_table_storage.get();
    std::fill(m_packed_table, m_packed_table + padded_size, 0);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
PairwiseTable(
    const luint_t node_a,
    const luint_t node_b,
    const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set,
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table)
: PairwiseTable<COSTTYPE, SIMDWIDTH>(node_a, node_b, lbl_set)
{
    set_costs(packed_table);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
PairwiseTable(
    const luint_t node_a,
    const luint_t node_b,
    const LabelSet<COSTTYPE, SIMDWIDTH> * lbl_set,
    _s_t<COSTTYPE, SIMDWIDTH> * packed_table)
: m_node_a(node_a),
  m_node_b(node_b),
  m_lbl_set(lbl_set),
  m_packed_table(packed_table)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
PairwiseTable<COSTTYPE, SIMDWIDTH>::
~PairwiseTable()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
PairwiseTable<COSTTYPE, SIMDWIDTH>::
set_costs(
    const std::vector<_s_t<COSTTYPE, SIMDWIDTH>>& packed_table)
{
    const _iv_st<COSTTYPE, SIMDWIDTH> len_a = m_lbl_set->label_set_size(
        m_node_a);
    const _iv_st<COSTTYPE, SIMDWIDTH> len_b = m_lbl_set->label_set_size(
        m_node_b);
    const _iv_st<COSTTYPE, SIMDWIDTH> padded_b = DIV_UP(len_b, SIMDWIDTH) * 
        SIMDWIDTH;

    /* expand table into aligned storage */
    for(_iv_st<COSTTYPE, SIMDWIDTH> li_a = 0; li_a < len_a; ++li_a)
        std::copy(&packed_table[li_a * len_b], 
            &packed_table[(li_a + 1) * len_b],
            &m_packed_table[li_a * padded_b]);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH> *
PairwiseTable<COSTTYPE, SIMDWIDTH>::
get_raw_costs()
{
    return m_packed_table;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>>
PairwiseTable<COSTTYPE, SIMDWIDTH>::
copy()
const
{
    PairwiseTable<COSTTYPE, SIMDWIDTH> * pwtab = new
        PairwiseTable<COSTTYPE, SIMDWIDTH>(m_node_a, m_node_b, m_lbl_set);

    const _iv_st<COSTTYPE, SIMDWIDTH> len_a = m_lbl_set->label_set_size(
        m_node_a);
    const _iv_st<COSTTYPE, SIMDWIDTH> len_b = m_lbl_set->label_set_size(
        m_node_b);
    const _iv_st<COSTTYPE, SIMDWIDTH> padded_b = DIV_UP(len_b, SIMDWIDTH) * 
        SIMDWIDTH;

    std::copy(m_packed_table, m_packed_table + len_a * padded_b, 
        pwtab->get_raw_costs());

    std::unique_ptr<PairwiseCosts<COSTTYPE, SIMDWIDTH>> uptr(pwtab);
    return std::move(uptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTable<COSTTYPE, SIMDWIDTH>::
supports_enumerable_costs()
const
{
    return true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
bool
PairwiseTable<COSTTYPE, SIMDWIDTH>::
eq(
    const PairwiseCosts<COSTTYPE, SIMDWIDTH> * costs)
const
{
    /* avoids long comparisons for multilevel */
    return false;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTable<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs(
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& label_vec_2)
const
{
    /* find label vector indices */
    _iv_t<COSTTYPE, SIMDWIDTH> l_ix_1 = m_lbl_set->offsets_for_labels(
        m_node_a, label_vec_1);
    _iv_t<COSTTYPE, SIMDWIDTH> l_ix_2 = m_lbl_set->offsets_for_labels(
        m_node_b, label_vec_2);

    /* directly query full table */
    const _iv_st<COSTTYPE, SIMDWIDTH> padded_b = 
        DIV_UP(m_lbl_set->label_set_size(m_node_b), SIMDWIDTH) * SIMDWIDTH;
    _iv_t<COSTTYPE, SIMDWIDTH> addresses =
        iv_init<COSTTYPE, SIMDWIDTH>(padded_b);
    addresses = iv_mult<COSTTYPE, SIMDWIDTH>(l_ix_1, addresses);
    addresses = iv_add<COSTTYPE, SIMDWIDTH>(l_ix_2, addresses);

    return v_gather<COSTTYPE, SIMDWIDTH>(m_packed_table, addresses);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
PairwiseTable<COSTTYPE, SIMDWIDTH>::
get_pairwise_costs_enum_offsets(
    const _iv_t<COSTTYPE, SIMDWIDTH>& offset_vec_1,
    const _iv_t<COSTTYPE, SIMDWIDTH>& offset_vec_2)
const
{
    /* directly query full table */
    const _iv_st<COSTTYPE, SIMDWIDTH> padded_b = 
        DIV_UP(m_lbl_set->label_set_size(m_node_b), SIMDWIDTH) * SIMDWIDTH;
    _iv_t<COSTTYPE, SIMDWIDTH> addresses =
        iv_init<COSTTYPE, SIMDWIDTH>(padded_b);
    addresses = iv_mult<COSTTYPE, SIMDWIDTH>(offset_vec_1, addresses);
    addresses = iv_add<COSTTYPE, SIMDWIDTH>(offset_vec_2, addresses);

    return v_gather<COSTTYPE, SIMDWIDTH>(m_packed_table, addresses);
}

NS_MAPMAP_END
