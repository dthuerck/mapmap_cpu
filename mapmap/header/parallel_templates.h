/*
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_PARALLEL_TEMPLATES_H_
#define __MAPMAP_PARALLEL_TEMPLATES_H_

#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <tbb/atomic.h>

#include "header/defines.h"
#include "header/vector_types.h"

NS_MAPMAP_BEGIN

/**
 * Reduction body (operator: +) for tbb::parallel_reduce.
 */
template<typename VALTYPE, typename INDEXTYPE>
class PlusReduction
{
public:
    PlusReduction(VALTYPE* in);
    PlusReduction(PlusReduction<VALTYPE, INDEXTYPE>& lhs, tbb::split);
    ~PlusReduction();

    VALTYPE get_sum();
    void operator()(const tbb::blocked_range<INDEXTYPE>& r);
    void join(const PlusReduction<VALTYPE, INDEXTYPE>& rhs);

public:
    VALTYPE m_sum;
    VALTYPE* m_in;
};

/**
 * Reduction body (operator: max) for tbb::parallel_reduce.
 */
template<typename VALTYPE, typename INDEXTYPE>
class MaxReduction
{
public:
    MaxReduction(VALTYPE * in);
    MaxReduction(MaxReduction<VALTYPE, INDEXTYPE>& lhs, tbb::split);
    ~MaxReduction();

    VALTYPE get_max();
    void operator()(const tbb::blocked_range<INDEXTYPE>& r);
    void join(const MaxReduction<VALTYPE, INDEXTYPE>& rhs);

public:
    VALTYPE m_max;
    VALTYPE * m_in;
};

/**
 * Exclusive scan body (operator: +) for tbb::parallel_scan.
 */
template<typename VALTYPE, typename INDEXTYPE>
class PlusScan
{
public:
    PlusScan(VALTYPE* in, VALTYPE* out);
    PlusScan(PlusScan<VALTYPE, INDEXTYPE>& lhs, tbb::split);
    ~PlusScan();

    void operator()(const tbb::blocked_range<INDEXTYPE>& r, tbb::pre_scan_tag);
    void operator()(const tbb::blocked_range<INDEXTYPE>& r,
        tbb::final_scan_tag);
    void assign(PlusScan<VALTYPE, INDEXTYPE>& rhs);
    void reverse_join(PlusScan<VALTYPE, INDEXTYPE>& rhs);

public:
    VALTYPE m_sum;
    VALTYPE* m_in;
    VALTYPE* m_out;
};

/**
 * Histogram helper (count how many times each number occurs).
 *
 * Results in a list for values (must be discrete type) [0, length).
 */
template<typename VALTYPE, typename INDEXTYPE>
class Histogram
{
public:
    Histogram(VALTYPE* in, const VALTYPE max_val);
    ~Histogram();

    std::vector<VALTYPE>& operator()(const tbb::blocked_range<INDEXTYPE>& r);

protected:
    VALTYPE* m_in;
    VALTYPE m_max_val;
    INDEXTYPE m_length;

    std::vector<tbb::atomic<VALTYPE>> m_histogram;
    std::vector<VALTYPE> m_final_histogram;
};

NS_MAPMAP_END

/* include templated implementation */
#include "source/parallel_templates.impl.h"

#endif /* __MAPMAP_PARALLEL_TEMPLATES_H_ */
