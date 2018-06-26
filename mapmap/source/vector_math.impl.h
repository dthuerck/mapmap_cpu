/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
 
#include "header/vector_math.h"

#include <cstring>
#include <algorithm>

NS_MAPMAP_BEGIN

/** 
 * *****************************************************************************
 * ******************************* Utility *************************************
 * *****************************************************************************
 */

template<>
FORCEINLINE
constexpr
uint_t
sys_max_simd_width<float>()
{
	#if defined(__AVX__)
		return 8;
	#elif defined(__SSE4_2__)
		return 4;
	#else
		return 1;
	#endif
}

template<>
FORCEINLINE
constexpr
uint_t
sys_max_simd_width<double>()
{
	#if defined(__AVX__)
		return 4;
	#elif defined(__SSE4_2__)
		return 2;
	#else
		return 1;
	#endif
}

/** 
 * *****************************************************************************
 * **************************** float, width = 1 *******************************
 * *****************************************************************************
 */

/* Constructors */
template<>
FORCEINLINE
_v_t<float, 1>
v_init<float, 1>()
{
	return ((_v_t<float,1>) 0);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_init<float, 1>(
	const _s_t<float, 1>& a)
{
	return ((_v_t<float, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_init<float, 1>()
{
	return ((_iv_t<float, 1>) 0);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_init<float, 1>(
	const _iv_st<float, 1>& a)
{
	return ((_iv_t<float, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_sequence<float, 1>(
	const _iv_st<float, 1>& start)
{
	return start;
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<float, 1>
v_add<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	return (a + b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_sub<float, 1>(
	const _v_t<float, 1>& a,
	const _v_t<float, 1>& b)
{
	return (a - b);
}

/* ************************************************************************** */

/* _s_t<float, 1> = _v_t<float, 1>, hence only one instance */
template<>
FORCEINLINE
_v_t<float, 1>
v_mult<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	return (a * b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_abs<float, 1>(
	const _v_t<float, 1>& a)
{
	return std::abs(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_min<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	return std::min(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_max<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	return std::max(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_add<float, 1>(
	const _iv_t<float, 1>& a, 
	const _iv_t<float, 1>& b)
{
	return (a + b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_sub<float, 1>(
	const _iv_t<float, 1>& a, 
	const _iv_t<float, 1>& b)
{
	return (a - b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_mult<float, 1>(
	const _iv_t<float, 1>& a,
	const _iv_t<float, 1>& b)
{
	return (a * b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_abs<float, 1>(
	const _iv_t<float, 1>& a)
{
	return std::abs(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_min<float, 1>(
	const _iv_t<float, 1>& a, 
	const _iv_t<float, 1>& b)
{
	return std::min(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_max<float, 1>(
	const _iv_t<float, 1>& a, 
	const _iv_t<float, 1>& b)
{
	return std::max(a, b);
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<float, 1>
iv_reinterpret_v<float, 1>(
	const _iv_t<float, 1>& a)
{
	_v_t<float, 1> aa;
	std::memcpy(&aa, &a, sizeof(_iv_t<float, 1>));
	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
v_reinterpret_iv<float, 1>(
	const _v_t<float, 1>& a)
{
	_iv_t<float, 1> aa;
	std::memcpy(&aa, &a, sizeof(_v_t<float, 1>));
	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
iv_convert_v<float, 1>(
	const _iv_t<float, 1>& a)
{
	return ((_v_t<float, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
v_convert_iv<float, 1>(
	const _v_t<float, 1>& a)
{
	return ((_iv_t<float, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_eq<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	_iv_t<float, 1> result = ((a == b) ? ~0x0 : 0x0);
	_v_t<float, 1> aa;
	std::memcpy(&aa, &result, sizeof(_iv_t<float, 1>));
	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_not<float, 1>(
	const _v_t<float, 1>& a)
{
	const uint32_t ifloat = reinterpret_cast<const uint32_t&>(a);
	_iv_st<float, 1> fint = ~ifloat; 
	_v_t<float, 1> aa;
	std::memcpy(&aa, &fint, sizeof(_iv_st<float, 1>));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_and<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	const uint32_t ia = reinterpret_cast<const uint32_t&>(a);
	const uint32_t ib = reinterpret_cast<const uint32_t&>(b);

	uint32_t iand = ia & ib;
	_v_t<float, 1> aa;
	std::memcpy(&aa, &iand, sizeof(uint32_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_or<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	const uint32_t ia = reinterpret_cast<const uint32_t&>(a);
	const uint32_t ib = reinterpret_cast<const uint32_t&>(b);

	uint32_t iand = ia | ib;
	_v_t<float, 1> aa;
	std::memcpy(&aa, &iand, sizeof(uint32_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float,1>
v_le<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b)
{
	uint32_t result = ((a <= b) ? 0xFFFFFFFF : 0x0);
	_v_t<float, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint32_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_le<float, 1>(
	const _iv_t<float, 1>& a,
	const _iv_t<float, 1>& b)
{
	uint32_t result = (a <= b ? 0xFFFFFFFF : 0x0);
	_iv_t<float, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint32_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_eq<float, 1>(
	const _iv_t<float, 1>& a,
	const _iv_t<float, 1>& b)
{
	uint32_t result = (a == b ? 0xFFFFFFFF : 0x0);
	_iv_t<float, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint32_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float,1>
v_blend<float, 1>(
	const _v_t<float, 1>& a, 
	const _v_t<float, 1>& b, 
	const _v_t<float, 1>& mask)
{
	const uint32_t imask = reinterpret_cast<const uint32_t&>(mask);

	return (imask == 0xFFFFFFFF ? b : a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float,1>
iv_blend<float, 1>(
	const _iv_t<float, 1>& a, 
	const _iv_t<float, 1>& b, 
	const _iv_t<float, 1>& mask)
{
	const uint32_t imask = reinterpret_cast<const uint32_t&>(mask);

	return (imask == 0xFFFFFFFF ? b : a); 
}

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<float, 1>
v_extract<float, 1>(
	const _v_t<float, 1>& a,
	const int8_t imm)
{
	return a; 
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<float, 1>
iv_extract<float, 1>(
	const _iv_t<float, 1>& a,
	const int8_t imm)
{
	return a; 
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<float, 1>
v_load<float, 1>(
	const _s_t<float, 1>* ptr)
{
	return *ptr;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 1>
iv_load<float, 1>(
	const _iv_st<float, 1>* ptr)
{
	return *ptr;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<float, 1>(
	const _v_t<float, 1>& a,
	_s_t<float, 1>* ptr)
{
	*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<float, 1>(
	const _v_t<float, 1>& a,
	const _iv_t<float, 1>& mask,
	_s_t<float, 1>* ptr)
{
	if(mask)
		*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<float, 1>(
	const _iv_t<float, 1>& a,
	_iv_st<float, 1>* ptr)
{
	*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 1>
v_gather<float, 1>(
	const _s_t<float, 1> * base,
	const _iv_t<float, 1>& offsets)
{
	return base[offsets];
}

/* ************************************************************************** */

/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<float, 1>()
{
	return 1;
}

/** 
 * *****************************************************************************
 * **************************** float, width = 4 *******************************
 * *****************************************************************************
 */
#if defined(__SSE4_2__)

/* Constructors */

template<>
FORCEINLINE
_v_t<float,4>
v_init<float, 4>()
{
	return _mm_setzero_ps();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_init<float, 4>(
	const _s_t<float, 4>& a)
{
	return _mm_set1_ps(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_init<float, 4>()
{
	return _mm_setzero_si128();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_init<float, 4>(
	const _iv_st<float, 4>& a)
{
	return _mm_set1_epi32(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_sequence<float, 4>(
	const _iv_st<float, 4>& start)
{
	return _mm_set_epi32(start + 3, start + 2, start + 1, start);
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<float,4>
v_add<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_add_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_sub<float, 4>(
	const _v_t<float, 4>& a,
	const _v_t<float, 4>& b)
{
	return _mm_sub_ps(a, b);
}

/* ************************************************************************** */
	
template<>
FORCEINLINE
_v_t<float,4>
v_mult<float, 4>(
	const _v_t<float,4>& a, 
	const _v_t<float,4>& b)
{
	return _mm_mul_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_abs<float, 4>(
	const _v_t<float, 4>& a)
{
	/* for floats, just delete the sign bit.... */
	__m128i imask = _mm_set1_epi32(0x7FFFFFFF);
	__m128 mask = _mm_castsi128_ps(imask);

	return _mm_and_ps(a, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_min<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_min_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_max<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_max_ps(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_add<float, 4>(
	const _iv_t<float, 4>& a, 
	const _iv_t<float, 4>& b)
{
	return _mm_add_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_sub<float, 4>(
	const _iv_t<float, 4>& a, 
	const _iv_t<float, 4>& b)
{
	return _mm_sub_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_mult<float, 4>(
	const _iv_t<float, 4>& a,
	const _iv_t<float, 4>& b)
{
	/* throws away the upper 32bit of the multiplication result */
	return _mm_mullo_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_abs<float, 4>(
	const _iv_t<float, 4>& a)
{
	return _mm_abs_epi32(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_min<float, 4>(
	const _iv_t<float, 4>& a, 
	const _iv_t<float, 4>& b)
{
	return _mm_min_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_max<float, 4>(
	const _iv_t<float, 4>& a, 
	const _iv_t<float, 4>& b)
{
	return _mm_max_epi32(a, b);
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<float, 4>
iv_reinterpret_v<float, 4>(
	const _iv_t<float, 4>& a)
{
	return _mm_castsi128_ps(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
v_reinterpret_iv<float, 4>(
	const _v_t<float, 4>& a)
{
	return _mm_castps_si128(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
iv_convert_v<float, 4>(
	const _iv_t<float, 4>& a)
{
	return _mm_cvtepi32_ps(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
v_convert_iv<float, 4>(
	const _v_t<float, 4>& a)
{
	return _mm_cvtps_epi32(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_eq<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_cmpeq_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_not<float, 4>(
	const _v_t<float, 4>& a)
{
	/* ~a = (~a & 0xFFFFFFFF) */
	return _mm_andnot_ps(a, 
		_mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFF)));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_and<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_and_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_or<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_or_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_le<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b)
{
	return _mm_cmple_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_le<float, 4>(
	const _iv_t<float, 4>& a,
	const _iv_t<float, 4>& b)
{
	return v_reinterpret_iv<float, 4>(v_not<float, 4>(
		iv_reinterpret_v<float, 4>(_mm_cmpgt_epi32(a, b))));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_eq<float, 4>(
	const _iv_t<float, 4>& a,
	const _iv_t<float, 4>& b)
{
	return _mm_cmpeq_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_blend<float, 4>(
	const _v_t<float, 4>& a, 
	const _v_t<float, 4>& b, 
	const _v_t<float, 4>& mask)
{
	return _mm_blendv_ps(a, b, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_blend<float, 4>(
	const _iv_t<float, 4>& a, 
	const _iv_t<float, 4>& b, 
	const _iv_t<float, 4>& mask)
{
	return _mm_blendv_epi8(a, b, mask); 
}

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<float, 4>
v_extract<float, 4>(
	const _v_t<float, 4>& a,
	const int8_t imm)
{
	switch(imm)
	{
		case 0:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(a, 0));
		case 1:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(a, 1));
		case 2:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(a, 2));
		case 3:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(a, 3));
		default:
			return ((_s_t<float, 4>) 0);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<float, 4>
iv_extract<float, 4>(
	const _iv_t<float, 4>& a,
	const int8_t imm)
{
	switch(imm)
	{
		case 0:
			return _mm_extract_epi32(a, 0);
		case 1:
			return _mm_extract_epi32(a, 1);
		case 2:
			return _mm_extract_epi32(a, 2);
		case 3:
			return _mm_extract_epi32(a, 3);
		default:
			return ((_iv_st<float, 4>) 0);
	}
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<float, 4>
v_load<float, 4>(
	const _s_t<float, 4>* ptr)
{
	return (!((unsigned long) ptr & v_get_mask<float, 4>()) ? 
		_mm_load_ps(ptr) : _mm_loadu_ps(ptr));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 4>
iv_load<float, 4>(
	const _iv_st<float, 4>* ptr)
{
	return (!((unsigned long) ptr & iv_get_mask<float, 4>()) ? 
		_mm_load_si128((__m128i*) ptr) : _mm_loadu_si128((__m128i*)ptr));
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<float, 4>(
	const _v_t<float, 4>& a,
	_s_t<float, 4>* ptr)
{
	if(!((unsigned long) ptr & v_get_mask<float, 4>()))
		_mm_store_ps(ptr, a);
	else
		_mm_storeu_ps(ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<float, 4>(
	const _iv_t<float, 4>& a,
	_iv_st<float, 4>* ptr)
{
	if(!((unsigned long) ptr & iv_get_mask<float, 4>()))
		_mm_store_si128((__m128i*) ptr, a);
	else
		_mm_storeu_si128((__m128i*) ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<float, 4>(
	const _v_t<float, 4>& a,
	const _iv_t<float, 4>& mask,
	_s_t<float, 4>* ptr)
{
	union {
		int64_t i;
		__m64 v;
	} a1, m1, a2, m2;

	if(!((unsigned long) ptr & v_get_mask<float, 4>()))
	{
		/* split and store both parts */
		const __m128i aa = v_reinterpret_iv<float, 4>(a);
		a1.i = _mm_extract_epi64(aa, 0);
		a2.i = _mm_extract_epi64(aa, 1);
		m1.i = _mm_extract_epi64(mask, 0);
		m2.i = _mm_extract_epi64(mask, 1);
		_mm_maskmove_si64(a1.v, m1.v, (char *) ptr);
		_mm_maskmove_si64(a2.v, m2.v, (char *) ptr + 8);
	}
	else
	{
		_mm_maskmoveu_si128(v_reinterpret_iv<float, 4>(a), mask, (char *) ptr);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 4>
v_gather<float, 4>(
	const _s_t<float, 4> * base,
	const _iv_t<float, 4>& offsets)
{
#ifndef HAS___AVX2__
	/* no intrinsic available */
	_s_t<float, 4> tmp[4];
	_iv_st<float, 4> tmpi[4];

	iv_store<float, 4>(offsets, tmpi);
	tmp[0] = base[tmpi[0]];
	tmp[1] = base[tmpi[1]];
	tmp[2] = base[tmpi[2]];
	tmp[3] = base[tmpi[3]];

	return v_load<float, 4>(tmp);
#else
	return _mm_i32gather_ps(base, offsets, 1);
#endif
}

/* ************************************************************************** */


/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<float, 4>()
{
	return 4;
}

#endif /* __SSE4_2__ */

/**
 * ***************************************************************************** 
 * ***************************** float, width = 8 ******************************
 * *****************************************************************************
 */
#if defined(__AVX__)

/* Constructors */

template<>
FORCEINLINE
_v_t<float, 8>
v_init<float, 8>()
{
	return _mm256_setzero_ps();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_init<float, 8>(
	const _s_t<float, 8>& a)
{
	const _v_t<float, 8> aa =  _mm256_set1_ps(a);
	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_init<float, 8>()
{
	return _mm256_setzero_si256();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_init<float, 8>(
	const _iv_st<float, 8>& a)
{
	return _mm256_set1_epi32(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_sequence<float, 8>(
	const _iv_st<float, 8>& start)
{
	return _mm256_set_epi32(start + 7, start + 6, start + 5, start + 4,
		start + 3, start + 2, start + 1, start);
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<float, 8>
v_add<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_add_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_sub<float, 8>(
	const _v_t<float, 8>& a,
	const _v_t<float, 8>& b)
{
	return _mm256_sub_ps(a, b);
}

/* ************************************************************************** */
	
template<>
FORCEINLINE
_v_t<float, 8>
v_mult<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_mul_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_abs<float, 8>(
	const _v_t<float, 8>& a)
{
	/* for floats, just delete the sign bit.... */
	__m256i imask = _mm256_set1_epi32(0x7FFFFFFF);
	__m256 mask = _mm256_castsi256_ps(imask);

	return _mm256_and_ps(a, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_min<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_min_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_max<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_max_ps(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_add<float, 8>(
	const _iv_t<float, 8>& a, 
	const _iv_t<float, 8>& b)
{
#ifndef	HAS___AVX2__
	/* before AVX2, there is no 256-bit add, hence process in two steps */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);
	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);
 
	a1 = _mm_add_epi32(a1, b1);
	a2 = _mm_add_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_add_epi32(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_sub<float, 8>(
	const _iv_t<float, 8>& a, 
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* before AVX2, there is no 256-bit subtract, hence process in two steps */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);
	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);
 
	a1 = _mm_sub_epi32(a1, b1);
	a2 = _mm_sub_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_sub_epi32(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_mult<float, 8>(
	const _iv_t<float, 8>& a,
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* before AVX2, there is no 256-bit multiply, hence split register */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);
	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);
 
	a1 = _mm_mullo_epi32(a1, b1);
	a2 = _mm_mullo_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_mullo_epi32(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_abs<float, 8>(
	const _iv_t<float, 8>& a)
{
#ifndef HAS___AVX2__
	/* no mm256 abs, hence process in two batches */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	a1 = _mm_abs_epi32(a1);
	a2 = _mm_abs_epi32(a2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_abs_epi32(a);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_min<float, 8>(
	const _iv_t<float, 8>& a, 
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* no mm256 abs, hence process in two batches */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_min_epi32(a1, b1);
	a2 = _mm_min_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_min_epi32(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_max<float, 8>(
	const _iv_t<float, 8>& a, 
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* no mm256 abs, hence process in two batches */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_max_epi32(a1, b1);
	a2 = _mm_max_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_max_epi32(a, b);
#endif
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<float, 8>
iv_reinterpret_v<float, 8>(
	const _iv_t<float, 8>& a)
{
	return _mm256_castsi256_ps(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
v_reinterpret_iv<float, 8>(
	const _v_t<float, 8>& a)
{
	return _mm256_castps_si256(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
iv_convert_v<float, 8>(
	const _iv_t<float, 8>& a)
{
	return _mm256_cvtepi32_ps(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
v_convert_iv<float, 8>(
	const _v_t<float, 8>& a)
{
	return _mm256_cvtps_epi32(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_eq<float, 8>(
	const _v_t<float, 8>& a,
 	const _v_t<float, 8>& b)
{
	return _mm256_cmp_ps(a, b, _CMP_EQ_UQ);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_not<float, 8>(
	const _v_t<float, 8>& a)
{
	return _mm256_andnot_ps(a, 
		_mm256_castsi256_ps(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_and<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_and_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_or<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_or_ps(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_le<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b)
{
	return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_le<float, 8>(
	const _iv_t<float, 8>& a,
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* no integer compare before __AVX2__, need to split vector in half */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = v_reinterpret_iv<float, 4>(v_not<float, 4>(iv_reinterpret_v<float, 4>(
		_mm_cmpgt_epi32(a1, b1)))); 
	a2 = v_reinterpret_iv<float, 4>(v_not<float, 4>(iv_reinterpret_v<float, 4>(
		_mm_cmpgt_epi32(a2, b2)))); 

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return v_reinterpret_iv<float, 8>(v_not<float, 8>(
		iv_reinterpret_v<float, 8>(_mm256_cmpgt_epi32(a, b)))); 
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_eq<float, 8>(
	const _iv_t<float, 8>& a,
	const _iv_t<float, 8>& b)
{
#ifndef HAS___AVX2__
	/* no integer compare before __AVX2__, need to split vector in half */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_cmpeq_epi32(a1, b1); 
	a2 = _mm_cmpeq_epi32(a2, b2); 

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_cmpeq_epi32(a, b); 
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_blend<float, 8>(
	const _v_t<float, 8>& a, 
	const _v_t<float, 8>& b, 
	const _v_t<float, 8>& mask)
{
	return _mm256_blendv_ps(a, b, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_blend<float, 8>(
	const _iv_t<float, 8>& a, 
	const _iv_t<float, 8>& b, 
	const _iv_t<float, 8>& mask)
{
#ifndef HAS___AVX2__
	/* no blending before __AVX2__, need to split vector in half */
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	__m128i mask1 = _mm256_extractf128_si256(mask, 0);
	__m128i mask2 = _mm256_extractf128_si256(mask, 1);

	a1 = _mm_blendv_epi8(a1, b1, mask1);
	a2 = _mm_blendv_epi8(a2, b2, mask2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_blendv_epi8(a, b, mask);
#endif
} 

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<float, 8>
v_extract<float, 8>(
	const _v_t<float, 8>& a,
	const int8_t imm)
{
	_v_t<float, 4> tmp = v_init<float, 4>();

	if(imm < 4)
		tmp = _mm256_extractf128_ps(a, 0);
	else if (imm < 8)
		tmp = _mm256_extractf128_ps(a, 1);

	switch(imm)
	{
		case 0:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 0));
		case 1:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 1));
		case 2:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 2));
		case 3:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 3));
		case 4:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 0));
		case 5:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 1));
		case 6:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 2));
		case 7:
			return iv_reinterpret_v<float, 1>(_mm_extract_ps(tmp, 3));
		default:
			return ((_s_t<float, 8>) 0);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<float, 8>
iv_extract<float, 8>(
	const _iv_t<float, 8>& a,
	const int8_t imm)
{
	switch(imm)
	{
		case 0:
			return _mm256_extract_epi32(a, 0);
		case 1:
			return _mm256_extract_epi32(a, 1);
		case 2:
			return _mm256_extract_epi32(a, 2);
		case 3:
			return _mm256_extract_epi32(a, 3);
		case 4:
			return _mm256_extract_epi32(a, 4);
		case 5:
			return _mm256_extract_epi32(a, 5);
		case 6:
			return _mm256_extract_epi32(a, 6);
		case 7:
			return _mm256_extract_epi32(a, 7);
		default:
			return ((_iv_st<float, 8>) 0);
	}
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<float, 8>
v_load<float, 8>(
	const _s_t<float, 8>* ptr)
{
	if (!((unsigned long) ptr & v_get_mask<float, 8>())) 
		return _mm256_load_ps(ptr);
	else
		return _mm256_loadu_ps(ptr);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<float, 8>
iv_load<float, 8>(
	const _iv_st<float, 8>* ptr)
{
	if (!((unsigned long) ptr & iv_get_mask<float, 8>())) 
		return _mm256_load_si256((__m256i*) ptr);
	else
		return _mm256_loadu_si256((__m256i*) ptr);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<float, 8>(
	const _v_t<float, 8>& a,
	_s_t<float, 8>* ptr)
{
	if(!((unsigned long) ptr & v_get_mask<float, 8>()))
		_mm256_store_ps(ptr, a);
	else
		_mm256_storeu_ps(ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<float, 8>(
	const _v_t<float, 8>& a,
	const _iv_t<float, 8>& mask,
	_s_t<float, 8> * ptr)
{
	if(!((unsigned long) ptr & v_get_mask<float, 8>()))
	{
		_mm256_maskstore_ps(ptr, mask, a);
	}
	else
	{
		const _iv_t<float, 8> ia = v_reinterpret_iv<float, 8>(a);

		/* split and store both parts */
		const __m128i a1 = _mm256_extractf128_si256(ia, 0);
		const __m128i a2 = _mm256_extractf128_si256(ia, 1);

		const __m128i mask1 = _mm256_extractf128_si256(mask, 0);
		const __m128i mask2 = _mm256_extractf128_si256(mask, 1);

		_mm_maskmoveu_si128(a1, mask1, (char *) ptr);
		_mm_maskmoveu_si128(a2, mask2, (char *) (ptr + 4));
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<float, 8>(
	const _iv_t<float, 8>& a,
	_iv_st<float, 8>* ptr)
{
	if(!((unsigned long) ptr & iv_get_mask<float, 8>()))
		_mm256_store_si256((__m256i*) ptr, a);
	else
		_mm256_storeu_si256((__m256i*) ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<float, 8>
v_gather<float, 8>(
	const _s_t<float, 8> * base,
	const _iv_t<float, 8>& offsets)
{
#ifndef HAS___AVX2__
	/* no intrinsic available */
	_s_t<float, 8> tmp[8];
	_iv_st<float, 8> tmpi[8];

	iv_store<float, 8>(offsets, tmpi);
	tmp[0] = base[tmpi[0]];
	tmp[1] = base[tmpi[1]];
	tmp[2] = base[tmpi[2]];
	tmp[3] = base[tmpi[3]];
	tmp[4] = base[tmpi[4]];
	tmp[5] = base[tmpi[5]];
	tmp[6] = base[tmpi[6]];
	tmp[7] = base[tmpi[7]];

	return v_load<float, 8>(tmp);
#else
	return _mm256_i32gather_ps(base, offsets, 1);
#endif
}

/* ************************************************************************** */

/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<float, 8>()
{
	return 8;
}

#endif /* __AVX__ */

/** 
 * *****************************************************************************
 * ***************************** double, width = 1 *****************************
 * *****************************************************************************
 */

/* Constructors */

template<>
FORCEINLINE
_v_t<double, 1>
v_init<double, 1>()
{
	return ((_v_t<double, 1>) 0);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_init<double, 1>(
	const _s_t<double, 1>& a)
{
	return ((_v_t<double, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_init<double, 1>()
{
	return ((_iv_t<double, 1>) 0);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_init<double, 1>(
	const _iv_st<double, 1>& a)
{
	return ((_iv_t<double, 1>) a);
} 

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_sequence<double, 1>(
	const _iv_st<double, 1>& start)
{
	return start;
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<double, 1>
v_add<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	return (a + b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_sub<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	return (a - b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_mult<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	return (a * b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_abs<double, 1>(
	const _v_t<double, 1>& a)
{
	return std::abs(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_min<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	return std::min(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_max<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	return std::max(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_add<double, 1>(
	const _iv_t<double, 1>& a, 
	const _iv_t<double, 1>& b)
{
	return (a + b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_sub<double, 1>(
	const _iv_t<double, 1>& a,
	const _iv_t<double, 1>& b)
{
	return (a - b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_mult<double, 1>(
	const _iv_t<double, 1>& a,
	const _iv_t<double, 1>& b)
{
	return (a * b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_abs<double, 1>(
	const _iv_t<double, 1>& a)
{
	return std::abs(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_min<double, 1>(
	const _iv_t<double, 1>& a, 
	const _iv_t<double, 1>& b)
{
	return std::min(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_max<double, 1>(
	const _iv_t<double, 1>& a, 
	const _iv_t<double, 1>& b)
{
	return std::max(a, b);
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<double, 1>
iv_reinterpret_v<double, 1>(
	const _iv_t<double, 1>& a)
{
	_v_t<double, 1> aa;
	std::memcpy(&aa, &a, sizeof(_iv_t<double, 1>));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
v_reinterpret_iv<double, 1>(
	const _v_t<double, 1>& a)
{
	_iv_t<double, 1> aa;
	std::memcpy(&aa, &a, sizeof(_v_t<double, 1>));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
iv_convert_v<double, 1>(
	const _iv_t<double, 1>& a)
{
	return ((_v_t<double, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
v_convert_iv<double, 1>(
	const _v_t<double, 1>& a)
{
	return ((_iv_t<double, 1>) a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_eq<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	uint64_t result = ((a == b) ? 0xFFFFFFFFFFFFFFFF : 0x0);
	_v_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint64_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_not<double, 1>(
	const _v_t<double, 1>& a)
{
	const uint64_t result = ~reinterpret_cast<const uint64_t&>(a);
	_v_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint64_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_and<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	const uint64_t ia = reinterpret_cast<const uint64_t&>(a);
	const uint64_t ib = reinterpret_cast<const uint64_t&>(b);

	uint64_t result = ia & ib;
	_v_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint64_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_or<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	const uint64_t ia = reinterpret_cast<const uint64_t&>(a);
	const uint64_t ib = reinterpret_cast<const uint64_t&>(b);

	uint64_t result = ia | ib;
	_v_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(uint64_t));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_le<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b)
{
	uint64_t result = ((a <= b) ? 0xFFFFFFFFFFFFFFFF : 0);
	_v_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(result));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_le<double, 1>(
	const _iv_t<double, 1>& a,
	const _iv_t<double, 1>& b)
{
	uint64_t result = ((a <= b) ? 0xFFFFFFFFFFFFFFFF : 0);
	_iv_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(result));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_eq<double, 1>(
	const _iv_t<double, 1>& a,
	const _iv_t<double, 1>& b)
{
	uint64_t result = ((a == b) ? 0xFFFFFFFFFFFFFFFF : 0);
	_iv_t<double, 1> aa;
	std::memcpy(&aa, &result, sizeof(result));

	return aa;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_blend<double, 1>(
	const _v_t<double, 1>& a, 
	const _v_t<double, 1>& b, 
	const _v_t<double, 1>& mask)
{
	const uint64_t imask = reinterpret_cast<const uint64_t&>(mask);

	return (imask == 0xFFFFFFFFFFFFFFFF ? b : a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_blend<double, 1>(
	const _iv_t<double, 1>& a, 
	const _iv_t<double, 1>& b, 
	const _iv_t<double, 1>& mask)
{
	const uint64_t imask = reinterpret_cast<const uint64_t&>(mask);

	return (imask == 0xFFFFFFFFFFFFFFFF ? b : a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<double, 1>
v_extract<double, 1>(
	const _v_t<double, 1>& a,
	const int8_t imm)
{
	return a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<double, 1>
iv_extract<double, 1>(
	const _iv_t<double, 1>& a,
	const int8_t imm)
{
	return a;
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<double, 1>
v_load<double, 1>(
	const _s_t<double, 1>* ptr)
{
	return *ptr;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 1>
iv_load<double, 1>(
	const _iv_st<double, 1>* ptr)
{
	return *ptr;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<double, 1>(
	const _v_t<double, 1>& a,
	_s_t<double, 1>* ptr)
{
	*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<double, 1>(
	const _v_t<double, 1>& a,
	const _iv_t<double, 1>& mask,
	_s_t<double, 1> * ptr)
{
	if(mask)
		*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<double, 1>(
	const _iv_t<double, 1>& a,
	_iv_st<double, 1>* ptr)
{
	*ptr = a;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 1>
v_gather<double, 1>(
	const _s_t<double, 1> * base,
	const _iv_t<double, 1>& offsets)
{
	_iv_st<double, 1> tmpi[1];
	iv_store<double, 1>(offsets, tmpi);

	return base[tmpi[0]];
}

/* ************************************************************************** */

/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<double, 1>()
{
	return 1;
}

/** 
 * *****************************************************************************
 * **************************** double, width = 2 ******************************
 * *****************************************************************************
 */
#if defined(__SSE4_2__)

/* Constructor */

template<>
FORCEINLINE
_v_t<double, 2>
v_init<double, 2>()
{
	return _mm_setzero_pd();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_init<double, 2>(
	const _s_t<double, 2>& a)
{
	return _mm_set1_pd(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_init<double, 2>()
{
	return _mm_setzero_si128();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_init<double, 2>(
	const _iv_st<double, 2>& a)
{
	return _mm_set1_epi64x(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_sequence<double, 2>(
	const _iv_st<double, 2>& start)
{
	return _mm_set_epi64x(start + 1, start);
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<double, 2>
v_add<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_add_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_sub<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_sub_pd(a, b);
}

/* ************************************************************************** */
	
template<>
FORCEINLINE
_v_t<double, 2>
v_mult<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_mul_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_abs<double, 2>(
	const _v_t<double, 2>& a)
{
	/* for double, just delete first bit to set positive sign */
	_iv_t<double, 2> imask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF);
	_v_t<double, 2> mask = _mm_castsi128_pd(imask);

	return _mm_and_pd(a, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_min<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_min_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_max<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_max_pd(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_add<double, 2>(
	const _iv_t<double, 2>& a, 
	const _iv_t<double, 2>& b)
{
	return _mm_add_epi64(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_sub<double, 2>(
	const _iv_t<double, 2>& a, 
	const _iv_t<double, 2>& b)
{
	return _mm_sub_epi64(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_mult<double, 2>(
	const _iv_t<double, 2>& a,
	const _iv_t<double, 2>& b)
{
	/* WARNING: ignores upper 32bits of each operand! */
	return _mm_mul_epi32(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_abs<double, 2>(
	const _iv_t<double, 2>& a)
{
	__m128i minus_a = _mm_sub_epi64(_mm_setzero_si128(), a);

	__m128i mask = _mm_cmpgt_epi64(a, minus_a);
	__m128i result = _mm_blendv_epi8(minus_a, a, mask);

	return result;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_min<double, 2>(
	const _iv_t<double, 2>& a, 
	const _iv_t<double, 2>& b)
{
	__m128i mask = _mm_cmpgt_epi64(a, b);
	__m128i result = _mm_blendv_epi8(a, b, mask);

	return result;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_max<double, 2>(
	const _iv_t<double, 2>& a, 
	const _iv_t<double, 2>& b)
{
	__m128i mask = _mm_cmpgt_epi64(a, b);
	__m128i result = _mm_blendv_epi8(b, a, mask);

	return result;
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<double, 2>
iv_reinterpret_v<double, 2>(
	const _iv_t<double, 2>& a)
{
	return _mm_castsi128_pd(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
v_reinterpret_iv<double, 2>(
	const _v_t<double, 2>& a)
{
	return _mm_castpd_si128(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
iv_convert_v<double, 2>(
	const _iv_t<double, 2>& a)
{
	/** 
	 * there are only intrinsics for converting the lower two 32-bit integers
	 * to doubles, hence first compress 64 bit integers to 32bit by discarding
	 * the leading 32bits
	 */

	_iv_t<double, 2> aa = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 3, 2, 0)); 
	return _mm_cvtepi32_pd(aa);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
v_convert_iv<double, 2>(
	const _v_t<double, 2>& a)
{
	/**
	 * there are only intrinsics for converting doubles to 32bit inetegers,
	 * hence do that and reshuffle the register, then mask out upper 32bit
	 * per integer
	 */
	_iv_t<double, 2> aa = _mm_cvtpd_epi32(a);
	aa = _mm_shuffle_epi32(aa, _MM_SHUFFLE(3, 1, 2, 0));

	_iv_t<double, 2> mask = _mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF);
	
	return _mm_and_si128(aa, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_eq<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_cmpeq_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_not<double, 2>(
	const _v_t<double, 2>& a)
{
	return _mm_andnot_pd(a, 
		_mm_castsi128_pd(_mm_set1_epi64x(0xFFFFFFFFFFFFFFFF)));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_and<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_and_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_or<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_or_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_le<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b)
{
	return _mm_cmple_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_le<double, 2>(
	const _iv_t<double, 2>& a,
	const _iv_t<double, 2>& b)
{
	return v_reinterpret_iv<double, 2>(v_not<double, 2>(
		iv_reinterpret_v<double, 2>(_mm_cmpgt_epi64(a, b))));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_eq<double, 2>(
	const _iv_t<double, 2>& a,
	const _iv_t<double, 2>& b)
{
	return _mm_cmpeq_epi64(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double,2>
v_blend<double, 2>(
	const _v_t<double, 2>& a, 
	const _v_t<double, 2>& b, 
	const _v_t<double, 2>& mask)
{
	return _mm_blendv_pd(a, b, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_blend<double, 2>(
	const _iv_t<double, 2>& a, 
	const _iv_t<double, 2>& b, 
	const _iv_t<double, 2>& mask)
{
	return _mm_blendv_epi8(a, b, mask); 
}

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<double, 2>
v_extract<double, 2>(
	const _v_t<double, 2>& a,
	const int8_t imm)
{
	_iv_t<double, 2> tmp = v_reinterpret_iv<double, 2>(a);

	_iv_st<double, 2> b;
	switch(imm)
	{
		case 0:
			b = _mm_extract_epi64(tmp, 0);
			return iv_reinterpret_v<double, 1>(b);
		case 1:
			b = _mm_extract_epi64(tmp, 1);
			return iv_reinterpret_v<double, 1>(b);
		default:
			return ((_s_t<double, 2>) 0);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<double, 2>
iv_extract<double, 2>(
	const _iv_t<double, 2>& a,
	const int8_t imm)
{
	switch(imm)
	{
		case 0:
			return _mm_extract_epi64(a, 0);
		case 1:
			return _mm_extract_epi64(a, 1);
		default:
			return ((_iv_st<double, 2>) 0);
	}
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<double, 2>
v_load<double, 2>(
	const _s_t<double, 2>* ptr)
{
	return (!((unsigned long) ptr & v_get_mask<double, 2>()) ? 
		_mm_load_pd(ptr) : _mm_loadu_pd(ptr));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 2>
iv_load<double, 2>(
	const _iv_st<double, 2>* ptr)
{
	return (!((unsigned long) ptr & iv_get_mask<double, 2>()) ? 
		_mm_load_si128((__m128i *) ptr) : _mm_loadu_si128((__m128i *) ptr));
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<double, 2>(
	const _v_t<double, 2>& a,
	_s_t<double, 2>* ptr)
{
	if(!((unsigned long) ptr & v_get_mask<double, 2>()))
		_mm_store_pd(ptr, a);
	else
		_mm_storeu_pd(ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<double, 2>(
	const _iv_t<double, 2>& a,
	_iv_st<double, 2>* ptr)
{
	if(!((unsigned long) ptr & iv_get_mask<double, 2>()))
		_mm_store_si128((__m128i *) ptr, a);
	else
		_mm_storeu_si128((__m128i *) ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<double, 2>(
	const _v_t<double, 2>& a,
	const _iv_t<double, 2>& mask,
	_s_t<double, 2> * ptr)
{
	union {
		int64_t i;
		__m64 v;
	} a1, m1, a2, m2;

	if(!((unsigned long) ptr & v_get_mask<double, 2>()))
	{
		/* split and store both parts */
		const __m128i aa = v_reinterpret_iv<double, 2>(a);
		a1.i = _mm_extract_epi64(aa, 0);
		a2.i = _mm_extract_epi64(aa, 1);
	   	m1.i = _mm_extract_epi64(mask, 0);
		m2.i = _mm_extract_epi64(mask, 1);
		_mm_maskmove_si64(a1.v, m1.v, (char *) ptr);
		_mm_maskmove_si64(a2.v, m2.v, (char *) ptr + 8);
	}
	else
	{
		_mm_maskmoveu_si128(v_reinterpret_iv<double, 2>(a), mask, (char *) ptr);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 2>
v_gather<double, 2>(
	const _s_t<double, 2> * base,
	const _iv_t<double, 2>& offsets)
{
#ifndef HAS___AVX2__
	/* no intrinsic available */
	_s_t<double, 2> tmp[4];
	_iv_st<double, 2> tmpi[4];

	iv_store<double, 2>(offsets, tmpi);
	tmp[0] = base[tmpi[0]];
	tmp[1] = base[tmpi[1]];

	return v_load<double, 2>(tmp);
#else
	return _mm_i64gather_pd(base, offsets, 1);
#endif
}

/* ************************************************************************** */

/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<double, 2>()
{
	return 2;
}

#endif /* __SSE4_2__ */

/**
 * *****************************************************************************
 * ******************************* double, width = 4 ***************************
 * *****************************************************************************
 */
#if defined(__AVX__)

/* Constructor */

template<>
FORCEINLINE
_v_t<double, 4>
v_init<double, 4>()
{
	return _mm256_setzero_pd();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_init<double, 4>(
	const _s_t<double, 4>& a)
{
	return _mm256_set1_pd(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_init<double, 4>()
{
	return _mm256_setzero_si256();
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_init<double, 4>(
	const _iv_st<double, 4>& a)
{
	return _mm256_set1_epi64x(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_sequence<double, 4>(
	const _iv_st<double, 4>& start)
{
	return _mm256_set_epi64x(start + 3, start + 2, start + 1, start);
}

/* ************************************************************************** */

/* Real arithmetique */

template<>
FORCEINLINE
_v_t<double, 4>
v_add<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_add_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_sub<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_sub_pd(a, b);
}

/* ************************************************************************** */
	
template<>
FORCEINLINE
_v_t<double, 4>
v_mult<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_mul_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_abs<double, 4>(
	const _v_t<double, 4>& a)
{
	/* for double, just delete first bit to set positive sign */
	_iv_t<double, 4> imask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
	_v_t<double, 4> mask = _mm256_castsi256_pd(imask);

	return _mm256_and_pd(a, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_min<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_min_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_max<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_max_pd(a, b);
}

/* ************************************************************************** */

/* Integer arithmetique */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_add<double, 4>(
	const _iv_t<double, 4>& a, 
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_add_epi64(a1, b1);
	a2 = _mm_add_epi64(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_add_epi64(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_sub<double, 4>(
	const _iv_t<double, 4>& a, 
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_sub_epi64(a1, b1);
	a2 = _mm_sub_epi64(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_sub_epi64(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_mult<double, 4>(
	const _iv_t<double, 4>& a,
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_mul_epi32(a1, b1);
	a2 = _mm_mul_epi32(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_mul_epi32(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_abs<double, 4>(
	const _iv_t<double, 4>& a)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i minus_a1 = _mm_sub_epi64(_mm_setzero_si128(), a1);
	__m128i minus_a2 = _mm_sub_epi64(_mm_setzero_si128(), a2);

	__m128i mask_1 = _mm_cmpgt_epi64(a1, minus_a1);
	__m128i result_1 = _mm_blendv_epi8(minus_a1, a1, mask_1);

	__m128i mask_2 = _mm_cmpgt_epi64(a2, minus_a2);
	__m128i result_2 = _mm_blendv_epi8(minus_a2, a2, mask_2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, result_1, 0);
	result = _mm256_insertf128_si256(result, result_2, 1);

	return result;
#else
	__m256i minus_a = _mm256_sub_epi64(_mm256_setzero_si256(), a);
	__m256i mask = _mm256_cmpgt_epi64(a, minus_a);

	return _mm256_blendv_epi8(minus_a, a, mask);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_min<double, 4>(
	const _iv_t<double, 4>& a, 
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	__m128i mask_1 = _mm_cmpgt_epi64(a1, b1);
	__m128i result_1 = _mm_blendv_epi8(a1, b1, mask_1);

	__m128i mask_2 = _mm_cmpgt_epi64(a2, b2);
	__m128i result_2 = _mm_blendv_epi8(a2, b2, mask_2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, result_1, 0);
	result = _mm256_insertf128_si256(result, result_2, 1);

	return result;
#else
	__m256i mask = _mm256_cmpgt_epi64(a, b);

	return _mm256_blendv_epi8(a, b, mask);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_max<double, 4>(
	const _iv_t<double, 4>& a, 
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	__m128i mask_1 = _mm_cmpgt_epi64(a1, b1);
	__m128i result_1 = _mm_blendv_epi8(b1, a1, mask_1);

	__m128i mask_2 = _mm_cmpgt_epi64(a2, b2);
	__m128i result_2 = _mm_blendv_epi8(b2, a2, mask_2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, result_1, 0);
	result = _mm256_insertf128_si256(result, result_2, 1);

	return result;
#else
	__m256i mask = _mm256_cmplt_epi64(a, b);

	return _mm256_blendv_epi8(a, b, mask);
#endif
}

/* ************************************************************************** */

/* Logical */

template<>
FORCEINLINE
_v_t<double, 4>
iv_reinterpret_v<double, 4>(
	const _iv_t<double, 4>& a)
{
	return _mm256_castsi256_pd(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
v_reinterpret_iv<double, 4>(
	const _v_t<double, 4>& a)
{
	return _mm256_castpd_si256(a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
iv_convert_v<double, 4>(
	const _iv_t<double, 4>& a)
{
	/** 
	 * there are only intrinsics for converting the lower four 32-bit integers
	 * to doubles, hence first compress 64 bit integers to 32bit by discarding
	 * the leading 32bits,
	 */

	_iv_t<double, 2> a1 = _mm256_extractf128_si256(a, 0);
	_iv_t<double, 2> a2 = _mm256_extractf128_si256(a, 1);

	a1 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(1, 3, 2, 0));
	a2 = _mm_shuffle_epi32(a2, _MM_SHUFFLE(1, 3, 2, 0));

	_v_t<double, 2> result1 = _mm_cvtepi32_pd(a1);
	_v_t<double, 2> result2 = _mm_cvtepi32_pd(a2);

	_v_t<double, 4> result = _mm256_setzero_pd();
	result = _mm256_insertf128_pd(result, result1, 0);
	result = _mm256_insertf128_pd(result, result2, 1);

	return result;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
v_convert_iv<double, 4>(
	const _v_t<double, 4>& a)
{
	/**
	 * there are only intrinsics for converting doubles to 32bit inetegers,
	 * hence do that and reshuffle the register, then mask out upper 32bit
	 * per integer
	 */

	_v_t<double, 2> a1 = _mm256_extractf128_pd(a, 0);
	_v_t<double, 2> a2 = _mm256_extractf128_pd(a, 1);

	_iv_t<double, 2> aa1 = _mm_cvtpd_epi32(a1);
	_iv_t<double, 2> aa2 = _mm_cvtpd_epi32(a2);

	aa1 = _mm_shuffle_epi32(aa1, _MM_SHUFFLE(3, 1, 2, 0));
	aa2 = _mm_shuffle_epi32(aa2, _MM_SHUFFLE(3, 1, 2, 0));

	_iv_t<double, 2> mask = _mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF);
	aa1 = _mm_and_si128(aa1, mask);
	aa2 = _mm_and_si128(aa2, mask);

	_iv_t<double, 4> result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, aa1, 0);
	result = _mm256_insertf128_si256(result, aa2, 1);

	return result;
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_eq<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_not<double, 4>(
	const _v_t<double, 4>& a)
{
	return _mm256_andnot_pd(a, 
		_mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_and<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_and_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_or<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_or_pd(a, b);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_le<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b)
{
	return _mm256_cmp_pd(a, b, _CMP_LE_OS);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_le<double, 4>(
	const _iv_t<double, 4>& a,
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = v_reinterpret_iv<double, 2>(v_not<double, 2>(
		iv_reinterpret_v<double, 2>(_mm_cmpgt_epi64(a1, b1))));
	a2 = v_reinterpret_iv<double, 2>(v_not<double, 2>(
		iv_reinterpret_v<double, 2>(_mm_cmpgt_epi64(a2, b2))));

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return v_reinterpret_iv<double, 4>(v_not<double, 4>(
		iv_reinterpret_v<double, 4>(_mm256_cmpgt_epi64(a, b))));
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_eq<double, 4>(
	const _iv_t<double, 4>& a,
	const _iv_t<double, 4>& b)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	a1 = _mm_cmpeq_epi64(a1, b1);
	a2 = _mm_cmpeq_epi64(a2, b2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_cmpeq_epi64(a, b);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_blend<double, 4>(
	const _v_t<double, 4>& a, 
	const _v_t<double, 4>& b, 
	const _v_t<double, 4>& mask)
{
	return _mm256_blendv_pd(a, b, mask);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_blend<double, 4>(
	const _iv_t<double, 4>& a, 
	const _iv_t<double, 4>& b, 
	const _iv_t<double, 4>& mask)
{
#ifndef HAS___AVX2__
	__m128i a1 = _mm256_extractf128_si256(a, 0);
	__m128i a2 = _mm256_extractf128_si256(a, 1);

	__m128i b1 = _mm256_extractf128_si256(b, 0);
	__m128i b2 = _mm256_extractf128_si256(b, 1);

	__m128i mask1 = _mm256_extractf128_si256(mask, 0);
	__m128i mask2 = _mm256_extractf128_si256(mask, 1);

	a1 = _mm_blendv_epi8(a1, b1, mask1);
	a2 = _mm_blendv_epi8(a2, b2, mask2);

	__m256i result = _mm256_setzero_si256();
	result = _mm256_insertf128_si256(result, a1, 0);
	result = _mm256_insertf128_si256(result, a2, 1);

	return result;
#else
	return _mm256_blendv_epi8(a, b, mask);
#endif
}

/* ************************************************************************** */

template<>
FORCEINLINE
_s_t<double, 4>
v_extract<double, 4>(
	const _v_t<double, 4>& a,
	const int8_t imm)
{
	_iv_t<double, 4> tmp = v_reinterpret_iv<double, 4>(a);

	_iv_st<double, 2> b;
	switch(imm)
	{
		case 0:
			b = _mm256_extract_epi64(tmp, 0);
			return iv_reinterpret_v<double, 1>(b);
		case 1:
			b = _mm256_extract_epi64(tmp, 1);
			return iv_reinterpret_v<double, 1>(b);
		case 2:
			b = _mm256_extract_epi64(tmp, 2);
			return iv_reinterpret_v<double, 1>(b);
		case 3:
			b = _mm256_extract_epi64(tmp, 3);
			return iv_reinterpret_v<double, 1>(b);
		default:
			return ((_iv_st<double, 4>) 0);
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_st<double, 4>
iv_extract<double, 4>(
	const _iv_t<double, 4>& a,
	const int8_t imm)
{
	switch(imm)
	{
		case 0:
			return _mm256_extract_epi64(a, 0);
		case 1:
			return _mm256_extract_epi64(a, 1);
		case 2:
			return _mm256_extract_epi64(a, 2);
		case 3:
			return _mm256_extract_epi64(a, 3);
		default:
			return ((_iv_st<double, 4>) 0);
	}
}

/* ************************************************************************** */

/* Memory */

template<>
FORCEINLINE
_v_t<double, 4>
v_load<double, 4>(
	const _s_t<double, 4>* ptr)
{
	return (!((unsigned long) ptr & v_get_mask<double, 4>()) ? 
		_mm256_load_pd(ptr) : _mm256_loadu_pd(ptr));
}

/* ************************************************************************** */

template<>
FORCEINLINE
_iv_t<double, 4>
iv_load<double, 4>(
	const _iv_st<double, 4>* ptr)
{
	return (!((unsigned long) ptr & iv_get_mask<double, 4>()) ? 
		_mm256_load_si256((__m256i *) ptr) : _mm256_loadu_si256((__m256i *) 
		ptr));  
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_store<double, 4>(
	const _v_t<double, 4>& a,
	_s_t<double, 4>* ptr)
{
	if(!((unsigned long) ptr & v_get_mask<double, 4>()))
		_mm256_store_pd(ptr, a);
	else
		_mm256_storeu_pd(ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
v_masked_store<double, 4>(
	const _v_t<double, 4>& a,
	const _iv_t<double, 4>& mask,
	_s_t<double, 4> * ptr)
{
	if(!((unsigned long) ptr & v_get_mask<double, 4>()))
	{
		_mm256_maskstore_pd(ptr, mask, a);
	}
	else
	{
		const _iv_t<double, 4> ia = v_reinterpret_iv<double, 4>(a);

		/* split and store both parts */
		const __m128i a1 = _mm256_extractf128_si256(ia, 0);
		const __m128i a2 = _mm256_extractf128_si256(ia, 1);

		const __m128i mask1 = _mm256_extractf128_si256(mask, 0);
		const __m128i mask2 = _mm256_extractf128_si256(mask, 1);

		_mm_maskmoveu_si128(a1, mask1, (char *) ptr);
		_mm_maskmoveu_si128(a2, mask2, (char *) (ptr + 2));
	}
}

/* ************************************************************************** */

template<>
FORCEINLINE
void
iv_store<double, 4>(
	const _iv_t<double, 4>& a,
	_iv_st<double, 4>* ptr)
{
	if(!((unsigned long) ptr & iv_get_mask<double, 4>()))
		_mm256_store_si256((__m256i *) ptr, a);
	else
		_mm256_storeu_si256((__m256i *) ptr, a);
}

/* ************************************************************************** */

template<>
FORCEINLINE
_v_t<double, 4>
v_gather<double, 4>(
	const _s_t<double, 4> * base,
	const _iv_t<double, 4>& offsets)
{
#ifndef HAS___AVX2__
	/* no intrinsic available */
	_s_t<double, 4> tmp[4];
	_iv_st<double, 4> tmpi[4];

	iv_store<double, 4>(offsets, tmpi);
	tmp[0] = base[tmpi[0]];
	tmp[1] = base[tmpi[1]];
	tmp[2] = base[tmpi[2]];
	tmp[3] = base[tmpi[3]];

	return v_load<double, 4>(tmp);
#else
	return _mm256_i64gather_pd(base, offsets, 1);
#endif
}

/* ************************************************************************** */

/* Miscellaneous */

template<>
FORCEINLINE
uint_t
v_alignment<double, 4>()
{
	return 4;
}

#endif /* __AVX__ */

NS_MAPMAP_END 
