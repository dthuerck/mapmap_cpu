/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include "header/parallel_templates.h"

#include <limits>

NS_MAPMAP_BEGIN

template<typename VALTYPE, typename INDEXTYPE>
PlusReduction<VALTYPE, INDEXTYPE>::
PlusReduction(
    VALTYPE* in)
    : m_sum((VALTYPE) 0),
      m_in(in)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
PlusReduction<VALTYPE, INDEXTYPE>::
PlusReduction(
    PlusReduction<VALTYPE, INDEXTYPE>& lhs,
    tbb::split)
    : PlusReduction<VALTYPE, INDEXTYPE>(lhs.m_in)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
PlusReduction<VALTYPE, INDEXTYPE>::
~PlusReduction()
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
VALTYPE
PlusReduction<VALTYPE, INDEXTYPE>::
get_sum()
{
    return m_sum;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusReduction<VALTYPE, INDEXTYPE>::
operator()(
    const tbb::blocked_range<INDEXTYPE>& r)
{
    for(INDEXTYPE i = r.begin(); i != r.end(); ++i)
        m_sum += m_in[i];
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusReduction<VALTYPE, INDEXTYPE>::
join(
    const PlusReduction<VALTYPE, INDEXTYPE>& rhs)
{
    m_sum += rhs.m_sum;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
MaxReduction<VALTYPE, INDEXTYPE>::
MaxReduction(
    VALTYPE * in)
    : m_max((VALTYPE) std::numeric_limits<VALTYPE>::min()),
      m_in(in)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
MaxReduction<VALTYPE, INDEXTYPE>::
MaxReduction(
    MaxReduction<VALTYPE, INDEXTYPE>& lhs,
    tbb::split)
    : MaxReduction<VALTYPE, INDEXTYPE>(lhs.m_in)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
MaxReduction<VALTYPE, INDEXTYPE>::
~MaxReduction()
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
VALTYPE
MaxReduction<VALTYPE, INDEXTYPE>::
get_max()
{
    return m_max;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
MaxReduction<VALTYPE, INDEXTYPE>::
operator()(
    const tbb::blocked_range<INDEXTYPE>& r)
{
    for(INDEXTYPE i = r.begin(); i != r.end(); ++i)
        m_max = std::max(m_max, m_in[i]);
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
MaxReduction<VALTYPE, INDEXTYPE>::
join(
    const MaxReduction<VALTYPE, INDEXTYPE>& rhs)
{
    m_max = std::max(m_max, rhs.m_max);
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
PlusScan<VALTYPE, INDEXTYPE>::
PlusScan(
    VALTYPE* in,
    VALTYPE* out)
    : m_sum((VALTYPE) 0),
      m_in(in),
      m_out(out)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
PlusScan<VALTYPE, INDEXTYPE>::
PlusScan(
    PlusScan<VALTYPE, INDEXTYPE>& lhs,
    tbb::split)
    : PlusScan<VALTYPE, INDEXTYPE>(lhs.m_in, lhs.m_out)
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
PlusScan<VALTYPE, INDEXTYPE>::
~PlusScan()
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusScan<VALTYPE, INDEXTYPE>::
operator()(
    const tbb::blocked_range<INDEXTYPE>& r,
    tbb::pre_scan_tag)
{
    VALTYPE tmp = m_sum;
    for(INDEXTYPE i = r.begin(); i != r.end(); ++i)
        tmp += m_in[i];

    m_sum = tmp;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusScan<VALTYPE, INDEXTYPE>::
operator()(
    const tbb::blocked_range<INDEXTYPE>& r,
    tbb::final_scan_tag)
{
    VALTYPE tmp = m_sum;
    for(INDEXTYPE i = r.begin(); i != r.end(); ++i)
    {
        m_out[i] = tmp;
        tmp += m_in[i];
    }

    m_sum = tmp;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusScan<VALTYPE, INDEXTYPE>::
assign(
    PlusScan<VALTYPE, INDEXTYPE>& rhs)
{
    m_sum = rhs.m_sum;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
void
PlusScan<VALTYPE, INDEXTYPE>::
reverse_join(
    PlusScan<VALTYPE, INDEXTYPE>& rhs)
{
    m_sum = rhs.m_sum + m_sum;
}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
Histogram<VALTYPE, INDEXTYPE>::
Histogram(
    VALTYPE* in,
    const VALTYPE max_val)
    : m_in(in),
      m_max_val(max_val),
      m_histogram()
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
Histogram<VALTYPE, INDEXTYPE>::
~Histogram()
{

}

/* ************************************************************************** */

template<typename VALTYPE, typename INDEXTYPE>
std::vector<VALTYPE>&
Histogram<VALTYPE, INDEXTYPE>::
operator()(
    const tbb::blocked_range<INDEXTYPE>& r)
{
    m_histogram.resize(r.end());
    std::fill(m_histogram.begin(), m_histogram.end(), 0);

    m_final_histogram.clear();
    m_final_histogram.resize(r.end(), (VALTYPE) 0);

    tbb::parallel_for(r,
        [&](const tbb::blocked_range<INDEXTYPE>& ri)
        {
            for(INDEXTYPE n = ri.begin(); n != ri.end(); ++n)
            {
                const INDEXTYPE number = m_in[n];
                if (number < m_max_val)
                    ++m_histogram[number];
            }
        });

    /* copy to fixed data */
    tbb::parallel_for(tbb::blocked_range<INDEXTYPE>(0, m_max_val),
        [&](const tbb::blocked_range<INDEXTYPE>& ri)
        {
            for(INDEXTYPE n = ri.begin(); n != ri.end(); ++n)
                m_final_histogram[n] = m_histogram[n];
        });

    return m_final_histogram;
}

NS_MAPMAP_END
