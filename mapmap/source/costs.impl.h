/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <iostream>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "header/costs.h"

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * ******************************** LabelSet ***********************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
LabelSet<COSTTYPE, SIMDWIDTH>::
LabelSet(
    const luint_t& num_graph_nodes,
    const bool compress)
: m_graph_num_nodes(num_graph_nodes),
  m_max_label_set_size(0),
  m_max_label(0),
  m_label_sets(),
  m_label_set_ids(m_graph_num_nodes),
  m_label_set_hashes(),
  m_is_ordered(),
  m_compress(compress)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
LabelSet<COSTTYPE, SIMDWIDTH>::
~LabelSet()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
max_label_set_size()
const
{
    return m_max_label_set_size;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
max_label()
const
{
    return m_max_label;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
label_set_size(
    const luint_t& node_id)
const
{
    if(node_id >= m_graph_num_nodes)
        return 0;

    return m_label_set_sizes[m_label_set_ids[node_id]];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
labels_from_offset(
    const luint_t& node_id,
    const _iv_st<COSTTYPE, SIMDWIDTH>& offset)
const
{
    if(node_id >= m_graph_num_nodes)
        return iv_init<COSTTYPE, SIMDWIDTH>();

    /* determine label set for node and calculate pointer */
    const luint_t label_set_id = m_label_set_ids[node_id];
    const _iv_st<COSTTYPE, SIMDWIDTH>* ptr = (_iv_st<COSTTYPE, SIMDWIDTH>*)
        (&m_label_sets[label_set_id][offset]);

    /* load vector from ptr */
    return iv_load<COSTTYPE, SIMDWIDTH>(ptr);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_st<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
label_from_offset(
    const luint_t& node_id,
    const _iv_st<COSTTYPE, SIMDWIDTH>& offset)
const
{
    if(node_id >= m_graph_num_nodes)
        return (uint_t) 0;

    /* determine label set index and load single label from list */
    const luint_t label_set_id = m_label_set_ids[node_id];

    return m_label_sets[label_set_id][offset];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
offset_for_label(
    const luint_t& node_id,
    const _iv_st<COSTTYPE, SIMDWIDTH>& label)
const
{
    if(node_id >= m_graph_num_nodes)
        return (uint_t) 0;

    /* determine label set for node */
    const luint_t label_set_id = m_label_set_ids[node_id];

    /* find label in label set - binary search is sorted, otherwise linear */
    if(m_is_ordered[label_set_id])
    {
        /* binary search */
        const auto it = std::lower_bound(m_label_sets[label_set_id].begin(),
            m_label_sets[label_set_id].end(), label);

        if(it != m_label_sets[label_set_id].end() && !(label < *it))
            return (const _iv_st<COSTTYPE, SIMDWIDTH>)
                (it - m_label_sets[label_set_id].begin());
    }
    else
    {
        /* linear search */
        const auto it = std::find(m_label_sets[label_set_id].begin(),
            m_label_sets[label_set_id].end(), label);

        if(it != m_label_sets[label_set_id].end())
            return (const _iv_st<COSTTYPE, SIMDWIDTH>)
                (it - m_label_sets[label_set_id].begin());
    }

    return -1;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_t<COSTTYPE, SIMDWIDTH>
LabelSet<COSTTYPE, SIMDWIDTH>::
offsets_for_labels(
    const luint_t& node_id,
    const _iv_t<COSTTYPE, SIMDWIDTH>& labels)
const
{
    if(node_id >= m_graph_num_nodes)
        return iv_init<COSTTYPE, SIMDWIDTH>(0);

    /* determine label set for node */
    const luint_t label_set_id = m_label_set_ids[node_id];

    /* find label in label set - binary search is sorted, otherwise linear */
    _iv_st<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
    if(m_is_ordered[label_set_id])
    {
        for(uint_t i = 0; i < SIMDWIDTH; ++i)
        {
            tmp[i] = 0;
            const _iv_st<COSTTYPE, SIMDWIDTH> label = iv_extract<COSTTYPE,
                SIMDWIDTH>(labels, i);

            /* binary search */
            const auto it = std::lower_bound(m_label_sets[label_set_id].begin(),
                m_label_sets[label_set_id].end(), label);

            if(it != m_label_sets[label_set_id].end() && !(label < *it))
                tmp[i] = (const _iv_st<COSTTYPE, SIMDWIDTH>)
                    (it - m_label_sets[label_set_id].begin());
        }

        return iv_load<COSTTYPE, SIMDWIDTH>(tmp);
    }
    else
    {
        for(uint_t i = 0; i < SIMDWIDTH; ++i)
        {
            tmp[i] = 0;
            const _iv_st<COSTTYPE, SIMDWIDTH> label = iv_extract<COSTTYPE,
                SIMDWIDTH>(labels, i);

            /* linear search */
            const auto it = std::find(m_label_sets[label_set_id].begin(),
                m_label_sets[label_set_id].end(), label);

            if(it != m_label_sets[label_set_id].end())
                tmp[i] = (const _iv_st<COSTTYPE, SIMDWIDTH>)
                    (it - m_label_sets[label_set_id].begin());
        }

        return iv_load<COSTTYPE, SIMDWIDTH>(tmp);
    }

    return iv_init<COSTTYPE, SIMDWIDTH>();
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
LabelSet<COSTTYPE, SIMDWIDTH>::
set_label_set_for_node(
    const luint_t& node_id,
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& label_set)
{
    m_max_label_set_size = std::max(m_max_label_set_size,
        (_iv_st<COSTTYPE, SIMDWIDTH>) label_set.size());

    /* update max label */
    for(const _iv_st<COSTTYPE, SIMDWIDTH>& l : label_set)
        m_max_label = std::max(m_max_label, l);

    /**
     * if compression is activated, each newly passed label set is checked
     * against all other label sets and only added if it is new
     */
    if(m_compress)
    {
        const uint_t set_hash = hash(label_set);

        /* check all other hashes serially (might be a nested loop here!) */
        uint_t exists = invalid_uint_t;
        for(uint_t hash_ix = 0; hash_ix < m_label_set_hashes.size(); ++hash_ix)
        {
            if(m_label_set_hashes[hash_ix] == set_hash)
            {
                const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& other_set =
                    m_label_sets[hash_ix];

                /* now check label set for equivalence */
                bool is_equal = (label_set.size() == other_set.size());
                for(uint_t i = 0; (i < label_set.size()) && is_equal; ++i)
                    is_equal &= (label_set[i] == other_set[i]);

                if(is_equal)
                    exists = hash_ix;
            }
        }

        if(exists != invalid_uint_t)
        {
            m_label_set_ids[node_id] = exists;
            return;
        }

        /* in case set is new: add hash to list of known hashes */
        m_label_set_hashes.push_back(set_hash);
    }

    /* add label set as new label set (new ID: m_label_set_ids.size()) */
    m_label_set_sizes.push_back(label_set.size());
    m_label_set_ids[node_id] = m_label_sets.size();
    m_label_sets.push_back(label_set);

    /* detect whether label set is ordered */
    bool ordered = true;
    for(luint_t i = 0; ordered && i < (label_set.size() - 1); ++i)
        ordered &= (label_set[i] < label_set[i + 1]);
    m_is_ordered.push_back(ordered);

    /* expand label set to facilitate vectorized DP */
    m_label_sets.back().resize(SIMDWIDTH * DIV_UP(m_label_set_sizes.back(),
        SIMDWIDTH), std::numeric_limits<_iv_st<COSTTYPE, SIMDWIDTH>>::max());
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const uint_t
LabelSet<COSTTYPE, SIMDWIDTH>::
hash(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& label_set)
{
    /**
     * Hash function: \bigor_{l \in label_set} (1 << \mod(l, 32)),
     * i.e. OR over all label's modulo-32 bit position
     */
    uint_t hash = 0;
    for(const uint_t& l : label_set)
    {
        hash |= (1 << (l & 0x1f));
    }

    return hash;
}

/**
 * *****************************************************************************
 * ************************* ModeNotSupportedException *************************
 * *****************************************************************************
 */

FORCEINLINE
ModeNotSupportedException::
ModeNotSupportedException(
    const char* err_msg)
: m_err_msg(err_msg)
{

}

/* ************************************************************************** */

FORCEINLINE
ModeNotSupportedException::
ModeNotSupportedException(
    const std::string& err_msg)
: m_err_msg(err_msg)
{

}

/* ************************************************************************** */

FORCEINLINE
ModeNotSupportedException::
~ModeNotSupportedException()
{

}

/* ************************************************************************** */

FORCEINLINE
const char*
ModeNotSupportedException::
what()
const throw()
{
    return m_err_msg.c_str();
}

NS_MAPMAP_END
