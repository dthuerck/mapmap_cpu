/**
 * Copyright (C) 2017, Daniel Thuerck, Nick Heppert
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.h"

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
PairwiseTruncatedQuadraticEnvelope(
    const PairwiseTruncatedQuadratic<COSTTYPE, SIMDWIDTH> * base)
: m_label_diff_cap(base->get_label_diff_cap())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
~PairwiseTruncatedQuadraticEnvelope()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
void
PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
compute_m_primes(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    const _s_t<COSTTYPE, SIMDWIDTH> * icosts,
    const _iv_st<COSTTYPE, SIMDWIDTH> * lbl_union,
    const _iv_st<COSTTYPE, SIMDWIDTH> lbl_union_size,
    _s_t<COSTTYPE, SIMDWIDTH> * m_prime,
    _iv_st<COSTTYPE, SIMDWIDTH> * m_prime_ix,
    _s_t<COSTTYPE, SIMDWIDTH> * scratch)
{
    const luint_t node_id = node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        node->c_labels->label_set_size(node_id);

    const _s_t<COSTTYPE, SIMDWIDTH> c = node->c_node.to_parent_weight *
        ((PAIRWISE *) node->c_pairwise)->get_c();

    _iv_st<COSTTYPE, SIMDWIDTH> current_label = 0;
    _iv_st<COSTTYPE, SIMDWIDTH> lowest_label = 0;

    /* retrieve array adresses in scratch space */
    const luint_t offset = node_num_labels *
        sizeof(_iv_st<COSTTYPE, SIMDWIDTH>);

    /* v: indices for lower envelope labels */
    _iv_st<COSTTYPE, SIMDWIDTH> * v = (_iv_st<COSTTYPE, SIMDWIDTH> *) scratch;

    /* z: intersection of lower envelope between two parabolas */
    _s_t<COSTTYPE, SIMDWIDTH> * z = (_s_t<COSTTYPE, SIMDWIDTH> *) (
        (char *) scratch + offset);

    v[0] = 0;
    z[0] = -std::numeric_limits<COSTTYPE>::max();
    z[1] = std::numeric_limits<COSTTYPE>::max();

    _iv_st<COSTTYPE, SIMDWIDTH> j = 0;
    _s_t<COSTTYPE, SIMDWIDTH> s = 0.;

    /* Constructing envelope */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 1; l_i < node_num_labels; ++l_i)
    {
        current_label = node->c_labels->label_from_offset(node_id, l_i);

        while(true)
        {
            lowest_label = node->c_labels->label_from_offset(node_id, v[j]);

            s = ((icosts[l_i] + c * current_label * current_label) -
                (icosts[v[j]] + c * lowest_label * lowest_label))
                / (2 * c * current_label - 2 * c * lowest_label);

            /**
             * Skim to the left as long as the current intersection 
             * is lower then previous 
             */
            if(s < z[j])
            {
                --j;
            }
            else
            {
                /* Insert a new lowest intersection */
                ++j;
                v[j] = l_i;
                z[j] = s;
                z[j + 1] = std::numeric_limits<COSTTYPE>::max();
                break;
            }
        }
    }

    j = 0;

    /* Sample the previously constructed lowest envelope */
    for(_iv_st<COSTTYPE, SIMDWIDTH> l_i = 0; l_i < lbl_union_size; ++l_i)
    {
        current_label = lbl_union[l_i];

        /* Search the nearest intersection to the right */
        while(z[j + 1] < current_label)
            ++j;

        lowest_label = node->c_labels->label_from_offset(node_id, v[j]);

        m_prime[l_i] = c * (current_label - lowest_label) * (current_label -
            lowest_label) + icosts[v[j]];
        m_prime_ix[l_i] = v[j];
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
cost_bound_d()
{
    return m_label_diff_cap;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH, typename UNARY, typename PAIRWISE>
FORCEINLINE
luint_t
PairwiseTruncatedQuadraticEnvelope<COSTTYPE, SIMDWIDTH, UNARY, PAIRWISE>::
scratch_bytes_needed(
    DPNode<COSTTYPE, SIMDWIDTH> * node)
{
    const luint_t node_id = node->c_node.node_id;
    const _iv_st<COSTTYPE, SIMDWIDTH> node_num_labels =
        node->c_labels->label_set_size(node_id);

    return ((node_num_labels + 1) * sizeof(_s_t<COSTTYPE, SIMDWIDTH>) +
        node_num_labels * sizeof(_iv_st<COSTTYPE, SIMDWIDTH>));
}

NS_MAPMAP_END
