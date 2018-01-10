/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
 
#ifndef __MAPMAP_HEADER_VECTOR_MATH_H_
#define __MAPMAP_HEADER_VECTOR_MATH_H_
 
#include <cmath>
#include <cstdio>
#include <cstring>

#include "header/defines.h"
#include "header/vector_types.h"

NS_MAPMAP_BEGIN

template<typename A, uint_t B>
using _v_t = vector_t<A, B>;

template<typename A, uint_t B> 
using _iv_t = ivector_t<A, B>;

template<typename A, uint_t B>
using _iv_st = ivector_st<A, B>;

template<typename A, uint_t B>
using _iv_ust = ivector_ust<A, B>;

template<typename A, uint_t B>
using _s_t = scalar_t<A, B>;

/**
 * Utility: get maximum supported vector size
 */
template<typename COSTTYPE>
FORCEINLINE
constexpr
uint_t
sys_max_simd_width();

/**
 * Constructors
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_init();

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_init(const _s_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_init();

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_init(const _iv_st<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_sequence(const _iv_st<COSTTYPE, SIMDWIDTH>& start);

/** 
 * Real arithmetique functions
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
v_add(const _v_t<COSTTYPE, SIMDWIDTH>& a, 
    const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
v_sub(const _v_t<COSTTYPE, SIMDWIDTH>& a, 
    const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */
	
template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH> 
v_mult(const _v_t<COSTTYPE, SIMDWIDTH>& a, 
    const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_abs(const _v_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_min(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_max(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b);

/** 
 * Integer arithmetique functions
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_add(const _iv_t<COSTTYPE, SIMDWIDTH>& a, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_sub(const _iv_t<COSTTYPE, SIMDWIDTH>& a, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_mult(const _iv_t<COSTTYPE, SIMDWIDTH>& a,
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_abs(const _iv_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_min(const _iv_t<COSTTYPE, SIMDWIDTH>& a, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_max(const _iv_t<COSTTYPE, SIMDWIDTH>& a, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/**
 * Logical functions
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
iv_reinterpret_v(const _iv_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
v_reinterpret_iv(const _v_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
iv_convert_v(const _iv_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
v_convert_iv(const _v_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_eq(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_not(const _v_t<COSTTYPE, SIMDWIDTH>& a);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_and(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_or(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_le(const _v_t<COSTTYPE, SIMDWIDTH>& a, 
    const _v_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_le(const _iv_t<COSTTYPE, SIMDWIDTH>& a,
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_eq(const _iv_t<COSTTYPE, SIMDWIDTH>& a,
    const _iv_t<COSTTYPE, SIMDWIDTH>& b);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_blend(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _v_t<COSTTYPE, SIMDWIDTH>& b, 
    const _v_t<COSTTYPE, SIMDWIDTH>& mask);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_blend(const _iv_t<COSTTYPE, SIMDWIDTH>& a, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& b, 
    const _iv_t<COSTTYPE, SIMDWIDTH>& mask);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
v_extract(
    const _v_t<COSTTYPE, SIMDWIDTH>& a,
    const int8_t imm);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_st<COSTTYPE, SIMDWIDTH>
iv_extract(
    const _iv_t<COSTTYPE, SIMDWIDTH>& a,
    const int8_t imm);

/**
 * Memory functions
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_load(const _s_t<COSTTYPE, SIMDWIDTH>* ptr);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_t<COSTTYPE, SIMDWIDTH>
iv_load(const _iv_st<COSTTYPE, SIMDWIDTH>* ptr);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
v_store(const _v_t<COSTTYPE, SIMDWIDTH>& a, _s_t<COSTTYPE, SIMDWIDTH>* ptr);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
iv_store(const _iv_t<COSTTYPE, SIMDWIDTH>& a, _iv_st<COSTTYPE, SIMDWIDTH>* ptr);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
v_masked_store(const _v_t<COSTTYPE, SIMDWIDTH>& a, const _iv_t<COSTTYPE,
    SIMDWIDTH>& mask, _s_t<COSTTYPE, SIMDWIDTH> * ptr);

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_v_t<COSTTYPE, SIMDWIDTH>
v_gather(const _s_t<COSTTYPE, SIMDWIDTH> * base,
    const _iv_t<COSTTYPE, SIMDWIDTH>& offsets);

/**
 * Miscelleanous functions
 */

/* alignment in multiples of sizeof(COSTTYPE) */
template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
uint_t 
v_alignment();

NS_MAPMAP_END 

#include "source/vector_math.impl.h"
 
#endif /* MAPMAP_HEADER_VECTOR_MATH_H_ */
