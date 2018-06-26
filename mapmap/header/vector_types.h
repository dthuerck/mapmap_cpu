/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_VECTOR_TYPES_H_
#define __MAPMAP_VECTOR_TYPES_H_

#include <cstdint>

#include "header/defines.h"

#if defined(__SSE4_2__)
    #include <smmintrin.h>  
#endif

#if defined(__AVX__)
    #include <immintrin.h>
#endif

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * *************************** Scalar types (scalar_t) *************************
 * *****************************************************************************
 */

using int_t = int32_t;
using uint_t = uint32_t;
using lint_t = int64_t;
using luint_t = uint64_t;

#define invalid_uint_t UINT32_MAX
#define invalid_luint_t UINT64_MAX

template<typename COSTTYPE, uint_t SIMDWIDTH = 1>
using scalar_t = COSTTYPE;

/**
 * *****************************************************************************
 * ************************** Real vector types (vector_t) *********************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
struct _vector_traits
{
    typedef COSTTYPE type;
    unsigned int mask;
};

template<>
struct _vector_traits<float, 1>
{
    typedef float type;
    unsigned int mask = 0x4;
};
#if defined(__SSE4_2__)
template<>
struct _vector_traits<float, 4>
{
    typedef __m128 type;
    unsigned int mask = 0xF;
}; /* __SSE4_2__ */
#if defined(__AVX__)
template<>
struct _vector_traits<float, 8>
{
    typedef __m256 type;
    unsigned int mask = 0x1F;
}; /* __AVX__ */
#endif
#endif

template<>
struct _vector_traits<double, 1>
{
    typedef double type;
    unsigned int mask = 0x8;
};
#if defined(__SSE4_2__)
template<>
struct _vector_traits<double, 2>
{
    typedef __m128d type;
    unsigned int mask = 0xF;
}; /* __SSE4_2__ */
#if defined(__AVX__)
template<>
struct _vector_traits<double, 4>
{
    typedef __m256d type;
    unsigned int mask = 0x1F;
}; /* __AVX__ */
#endif
#endif

template<typename COSTTYPE, uint_t SIMDWIDTH>
using vector_t = typename _vector_traits<COSTTYPE, SIMDWIDTH>::type;

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
uint_t
v_get_mask()
{
    _vector_traits<COSTTYPE, SIMDWIDTH> vt;
    return vt.mask;
}

/**
 * *****************************************************************************
 * ************************* Int vector types (ivector_t) **********************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
struct _i_vector_traits
{
    typedef uint32_t utype;
    typedef int32_t itype;
    typedef int32_t type;
};

template<>
struct _i_vector_traits<float, 1>
{
    typedef int32_t uitype;
    typedef int32_t itype;
    typedef int32_t type;
    unsigned int mask = 0x4;
};
#if defined(__SSE4_2__)
template<>
struct _i_vector_traits<float, 4>
{
    typedef uint32_t uitype;
    typedef int32_t itype;
    typedef __m128i type;
    unsigned int mask = 0xF;
};
#endif /* __SSE4_2__ */
#if defined(__AVX__)
template<>
struct _i_vector_traits<float, 8>
{
    typedef uint32_t uitype;
    typedef int32_t itype;
    typedef __m256i type;
    unsigned int mask = 0x1F;
}; /* __AVX__ */
#endif

template<>
struct _i_vector_traits<double, 1>
{
    typedef uint64_t uitype;
    typedef int64_t itype;
    typedef int64_t type;
    unsigned int mask = 0x4;
};
#if defined(__SSE4_2__)
template<>
struct _i_vector_traits<double, 2>
{
    typedef uint64_t uitype;
    typedef int64_t itype;
    typedef __m128i type;
    unsigned int mask = 0xF;
};
#endif /* __SSE3 */
#if defined(__AVX__)
template<>
struct _i_vector_traits<double, 4>
{
    typedef uint64_t uitype;
    typedef int64_t itype;
    typedef __m256i type;
    unsigned int mask = 0x1F;
}; /* __AVX__ */
#endif

template<typename COSTTYPE, uint_t SIMDWIDTH>
using ivector_st = typename _i_vector_traits<COSTTYPE, SIMDWIDTH>::itype;

template<typename COSTTYPE, uint_t SIMDWIDTH>
using ivector_ust = typename _i_vector_traits<COSTTYPE, SIMDWIDTH>::uitype;

template<typename COSTTYPE, uint_t SIMDWIDTH>
using ivector_t = typename _i_vector_traits<COSTTYPE, SIMDWIDTH>::type;

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
uint_t
iv_get_mask()
{
    _i_vector_traits<COSTTYPE, SIMDWIDTH> ivt;
    return ivt.mask;
}

NS_MAPMAP_END

#endif /* __MAPMAP_VECTOR_TYPES_H_ */
