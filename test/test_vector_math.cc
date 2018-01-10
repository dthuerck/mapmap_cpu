/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <utility>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include "header/defines.h"
#include "header/vector_types.h"
#include "header/vector_math.h"

NS_MAPMAP_BEGIN

template <typename T, uint_t N>
class TestTuple {
public:
    typedef T Type;
    static const uint_t Value = N;
};

template<typename T>
class mapMAPTestVectorMath : public ::testing::Test
{
public:
    /* rename template parameters for convenience */
    typedef typename T::Type COSTTYPE;
    static const uint_t SIMDWIDTH = T::Value;

public:
    mapMAPTestVectorMath()
    {

    }

    ~mapMAPTestVectorMath()
    {

    }

    void
    SetUp()
    {
        fill_input();
    }

    void
    TearDown()
    {

    }

    void
    fill_input()
    {
        /* fill in data: [0 1 2 3 ...] */
        for(uint_t i = 0; i < SIMDWIDTH; ++i)
        {
            m_v_in[i] = (_s_t<COSTTYPE, SIMDWIDTH>) i;
            m_iv_in[i] = (_iv_st<COSTTYPE, SIMDWIDTH>) i;
        }

        /* fill in mask: [0xFF, 0x0, 0xFF, 0x0, ...] */
        for(uint_t i = 0; i < SIMDWIDTH; ++i)
            m_iv_mask[i] = (i % 2 == 0) ? 
                (_iv_st<COSTTYPE, SIMDWIDTH>) -1 : 
                0x0;
    }

protected:
    /* test input/output */
    _s_t<COSTTYPE, SIMDWIDTH> m_v_in[T::Value];
    _s_t<COSTTYPE, SIMDWIDTH> m_v_out[T::Value];
    _iv_st<COSTTYPE, SIMDWIDTH> m_iv_in[T::Value];
    _iv_st<COSTTYPE, SIMDWIDTH> m_iv_out[T::Value];
    _iv_st<COSTTYPE, SIMDWIDTH> m_iv_mask[T::Value];

    /* golden data */
    _s_t<COSTTYPE, SIMDWIDTH> m_v_ref[T::Value];
    _iv_st<COSTTYPE, SIMDWIDTH> m_iv_ref[T::Value];
};
TYPED_TEST_CASE_P(mapMAPTestVectorMath);

TYPED_TEST_P(mapMAPTestVectorMath, TestMemory)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* commonly used variables */
    _v_t<COSTTYPE, SIMDWIDTH> utest;
    _iv_t<COSTTYPE, SIMDWIDTH> iutest;
    _iv_t<COSTTYPE, SIMDWIDTH> imask;

    /* load and save real data */
    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_in[i], this->m_v_out[i]);

    /* load and save integer data */
    iutest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    iv_store<COSTTYPE, SIMDWIDTH>(iutest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_in[i], this->m_iv_out[i]);

    /* masked save */
    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    imask = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_mask);

    std::fill(this->m_v_out, this->m_v_out + SIMDWIDTH, 0);
    v_masked_store<COSTTYPE, SIMDWIDTH>(utest, imask, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ((i % 2 == 0) ? this->m_v_in[i] : 0, this->m_v_out[i]);
}

TYPED_TEST_P(mapMAPTestVectorMath, TestInit)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* commonly used variables */
    _v_t<COSTTYPE, SIMDWIDTH> utest;
    _iv_t<COSTTYPE, SIMDWIDTH> iutest;

    /* test v_init */
    utest = v_init<COSTTYPE, SIMDWIDTH>();
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ((_s_t<COSTTYPE, SIMDWIDTH>) 0, this->m_v_out[i]);

    /* test v_init(a) */
    utest = v_init<COSTTYPE, SIMDWIDTH>
        ((_s_t<COSTTYPE, SIMDWIDTH>) 1337.0);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ((_s_t<COSTTYPE, SIMDWIDTH>) 1337.0, this->m_v_out[i]);

    /* test iv_init */
    iutest = iv_init<COSTTYPE, SIMDWIDTH>();
    iv_store<COSTTYPE, SIMDWIDTH>(iutest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ((_iv_st<COSTTYPE, SIMDWIDTH>) 0, this->m_iv_out[i]);

    /* test iv_init(9)a) */
    iutest = iv_init<COSTTYPE, SIMDWIDTH>
        ((_iv_st<COSTTYPE, SIMDWIDTH>) 1337);
    iv_store<COSTTYPE, SIMDWIDTH>(iutest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ((_iv_st<COSTTYPE, SIMDWIDTH>) 1337, this->m_iv_out[i]);
}

TYPED_TEST_P(mapMAPTestVectorMath, TestRealArithmetique)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* commonly used variables and second operands */
    _v_t<COSTTYPE, SIMDWIDTH> utest;
    _v_t<COSTTYPE, SIMDWIDTH> usecond;
    _s_t<COSTTYPE, SIMDWIDTH> m_second[SIMDWIDTH];

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        m_second[i] = SIMDWIDTH - i;

    /* test v_add */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = this->m_v_in[i] + m_second[i];

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = v_add<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);

    /* test v_sub */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = this->m_v_in[i] - m_second[i];

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = v_sub<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);

    /* test v_mult */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = this->m_v_in[i] * m_second[i];

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = v_mult<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);

    /* test v_abs */
    this->fill_input();
    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_in[i] *= (i % 2 == 0 ? 1 : -1);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = std::abs(this->m_v_in[i]);

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    utest = v_abs<COSTTYPE, SIMDWIDTH>(utest);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);

    /* test v_min */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = std::min(this->m_v_in[i], m_second[i]);

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = v_min<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);

    /* test v_max */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_v_ref[i] = std::max(this->m_v_in[i], m_second[i]);

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = v_max<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(utest, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_v_ref[i], this->m_v_out[i]);
}

TYPED_TEST_P(mapMAPTestVectorMath, TestIntegerArithmetique)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* commonly used variables and second operands */
    _iv_t<COSTTYPE, SIMDWIDTH> utest;
    _iv_t<COSTTYPE, SIMDWIDTH> usecond;
    _iv_st<COSTTYPE, SIMDWIDTH> m_second[SIMDWIDTH];

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        m_second[i] = (_s_t<COSTTYPE, SIMDWIDTH>) (SIMDWIDTH - i);

    /* test iv_add */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = this->m_iv_in[i] + m_second[i];

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    usecond = iv_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = iv_add<COSTTYPE, SIMDWIDTH>(utest, usecond);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);

    /* test iv_sub */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = this->m_iv_in[i] - m_second[i];

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    usecond = iv_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = iv_sub<COSTTYPE, SIMDWIDTH>(utest, usecond);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);

    /* test iv_mult */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = this->m_iv_in[i] * m_second[i];

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    usecond = iv_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = iv_mult<COSTTYPE, SIMDWIDTH>(utest, usecond);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);

    /* test iv_abs */
    this->fill_input();
    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_in[i] *= (i % 2 == 0 ? 1 : -1);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = std::abs(this->m_iv_in[i]);

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    utest = iv_abs<COSTTYPE, SIMDWIDTH>(utest);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);

    /* test iv_min */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = std::min(this->m_iv_in[i], m_second[i]);

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    usecond = iv_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = iv_min<COSTTYPE, SIMDWIDTH>(utest, usecond);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);

    /* test iv_max */
    this->fill_input();
    
    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        this->m_iv_ref[i] = std::max(this->m_iv_in[i], m_second[i]);

    utest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    usecond = iv_load<COSTTYPE, SIMDWIDTH>(m_second);
    utest = iv_max<COSTTYPE, SIMDWIDTH>(utest, usecond);
    iv_store<COSTTYPE, SIMDWIDTH>(utest, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_ref[i], this->m_iv_out[i]);
}

TYPED_TEST_P(mapMAPTestVectorMath, TestLogical)
{
    typedef typename TypeParam::Type COSTTYPE;
    static const uint_t SIMDWIDTH = TypeParam::Value;

    /* common variables and other operands */
    _v_t<COSTTYPE, SIMDWIDTH> utest;
    _iv_t<COSTTYPE, SIMDWIDTH> iutest;
    _v_t<COSTTYPE, SIMDWIDTH> usecond;
    _iv_t<COSTTYPE, SIMDWIDTH> iusecond;
    _v_t<COSTTYPE, SIMDWIDTH> mask;
    _iv_t<COSTTYPE, SIMDWIDTH> imask;
    _v_t<COSTTYPE, SIMDWIDTH> result;
    _iv_t<COSTTYPE, SIMDWIDTH> iresult;

    _s_t<COSTTYPE, SIMDWIDTH> v_second[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> v_i_second[SIMDWIDTH];

    _s_t<COSTTYPE, SIMDWIDTH> v_mask[SIMDWIDTH];
    _iv_st<COSTTYPE, SIMDWIDTH> v_i_mask[SIMDWIDTH];

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        v_second[i] = (_s_t<COSTTYPE, SIMDWIDTH>) (SIMDWIDTH - i);
        v_i_second[i] = (_iv_st<COSTTYPE, SIMDWIDTH>) (SIMDWIDTH - i);
    }

    /* test iv_convert_v */
    this->fill_input();

    iutest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    result = iv_convert_v<COSTTYPE, SIMDWIDTH>(iutest);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
            std::isnan(this->m_v_out[i])) ||
            this->m_v_out[i] == this->m_v_in[i]);

    /* test v_convert_iv */
    this->fill_input();

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    iresult = v_convert_iv<COSTTYPE, SIMDWIDTH>(utest);
    iv_store<COSTTYPE, SIMDWIDTH>(iresult, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_EQ(this->m_iv_out[i], this->m_iv_in[i]);

    /* test v_eq */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _iv_ust<COSTTYPE, SIMDWIDTH> res = (this->m_v_in[i] ==
            v_second[i]) ? ~0x0 : 0x0;
        std::memcpy(&this->m_v_ref[i], &res,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(v_second);
    result = v_eq<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    /* note: 0xFFFF... is -nan and nan == nan is false, thus the checks */
    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
            std::isnan(this->m_v_out[i])) ||
            this->m_v_ref[i] == this->m_v_out[i]);

    /* test v_not */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _s_t<COSTTYPE, SIMDWIDTH> val = this->m_v_in[i];

        /* cast to unsigned int and negate all bits */
        _iv_ust<COSTTYPE, SIMDWIDTH> tmp;
        std::memcpy(&tmp, &val, sizeof(_s_t<COSTTYPE, SIMDWIDTH>));
        tmp = ~tmp;
        std::memcpy(&this->m_v_ref[i], &tmp,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    result = v_not<COSTTYPE, SIMDWIDTH>(utest);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
            std::isnan(this->m_v_out[i])) ||
            this->m_v_ref[i] == this->m_v_out[i]);

    /* test v_and */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _s_t<COSTTYPE, SIMDWIDTH> val = this->m_v_in[i];
        _s_t<COSTTYPE, SIMDWIDTH> second = v_second[i];

        _iv_st<COSTTYPE, SIMDWIDTH> ival;
        _iv_st<COSTTYPE, SIMDWIDTH> isecond;

        std::memcpy(&ival, &val, sizeof(_s_t<COSTTYPE, SIMDWIDTH>));
        std::memcpy(&isecond, &second, sizeof(_s_t<COSTTYPE, SIMDWIDTH>));

        _iv_ust<COSTTYPE, SIMDWIDTH> tmp = ival & isecond;
        std::memcpy(&this->m_v_ref[i], &tmp,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(v_second);
    result = v_and<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
            std::isnan(this->m_v_out[i])) ||
            this->m_v_ref[i] == this->m_v_out[i]);

    /* test v_le */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _iv_ust<COSTTYPE, SIMDWIDTH> res = (this->m_v_in[i] <=
            v_second[i]) ? ~0x0 : 0x0;
        std::memcpy(&this->m_v_ref[i], &res,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    utest =  v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(v_second);
    result = v_le<COSTTYPE, SIMDWIDTH>(utest, usecond);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
            std::isnan(this->m_v_out[i])) ||
            this->m_v_ref[i] == this->m_v_out[i]);

    /* test iv_le */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _iv_ust<COSTTYPE, SIMDWIDTH> res = (this->m_iv_in[i] <=
            v_i_second[i]) ? ~0x0 : 0x0;
        std::memcpy(&this->m_iv_ref[i], &res,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    iutest =  iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    iusecond = iv_load<COSTTYPE, SIMDWIDTH>(v_i_second);
    iresult = iv_le<COSTTYPE, SIMDWIDTH>(iutest, iusecond);
    iv_store<COSTTYPE, SIMDWIDTH>(iresult, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE(this->m_iv_ref[i] == this->m_iv_out[i]);

    /* test iv_eq */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _iv_ust<COSTTYPE, SIMDWIDTH> res = (this->m_iv_in[i] ==
            v_i_second[i]) ? ~0x0 : 0x0;
        std::memcpy(&this->m_iv_ref[i], &res,
            sizeof(_iv_ust<COSTTYPE, SIMDWIDTH>));
    }

    iutest =  iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    iusecond = iv_load<COSTTYPE, SIMDWIDTH>(v_i_second);
    iresult = iv_eq<COSTTYPE, SIMDWIDTH>(iutest, iusecond);
    iv_store<COSTTYPE, SIMDWIDTH>(iresult, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE(this->m_iv_ref[i] == this->m_iv_out[i]);

    /* test v_blend */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        _iv_st<COSTTYPE, SIMDWIDTH> tmp = (i % 2 == 1 ? ~0x0 : 0x0);
        std::memcpy(&v_mask[i], &tmp, sizeof(_iv_st<COSTTYPE, SIMDWIDTH>));

        this->m_v_ref[i] = (tmp == 0x0 ? this->m_v_in[i] : v_second[i]);
    }

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);
    usecond = v_load<COSTTYPE, SIMDWIDTH>(v_second);
    mask = v_load<COSTTYPE, SIMDWIDTH>(v_mask);
    result = v_blend<COSTTYPE, SIMDWIDTH>(utest, usecond, mask);
    v_store<COSTTYPE, SIMDWIDTH>(result, this->m_v_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_v_ref[i]) &&
                std::isnan(this->m_v_out[i])) ||
                this->m_v_ref[i] == this->m_v_out[i]);

    /* test iv_blend */
    this->fill_input();

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        v_i_mask[i] = (_iv_st<COSTTYPE, SIMDWIDTH>) (i % 2 == 1 ? ~0x0 : 0x0);

        this->m_iv_ref[i] = (v_i_mask == 0x0 ?
            this->m_iv_in[i] : v_i_second[i]);
    }

    iutest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);
    iusecond = iv_load<COSTTYPE, SIMDWIDTH>(v_i_second);
    imask = iv_load<COSTTYPE, SIMDWIDTH>(v_i_mask);
    iresult = iv_blend<COSTTYPE, SIMDWIDTH>(iutest, iusecond, imask);
    iv_store<COSTTYPE, SIMDWIDTH>(iresult, this->m_iv_out);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
        ASSERT_TRUE((std::isnan(this->m_iv_ref[i]) &&
                std::isnan(this->m_v_out[i])) ||
                this->m_v_ref[i] == this->m_v_out[i]);

    /* test iv_extract */
    this->fill_input();

    iutest = iv_load<COSTTYPE, SIMDWIDTH>(this->m_iv_in);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        const int8_t imm = (int8_t) i;
        _iv_st<COSTTYPE, SIMDWIDTH> ex = iv_extract<COSTTYPE, SIMDWIDTH>(
            iutest, imm);
        ASSERT_TRUE((std::isnan(ex) &&
                    std::isnan(this->m_iv_in[i])) ||
                    ex == this->m_iv_in[i]);
    }

    /* test v_extract */
    this->fill_input();

    utest = v_load<COSTTYPE, SIMDWIDTH>(this->m_v_in);

    for(uint_t i = 0; i < SIMDWIDTH; ++i)
    {
        const int8_t imm = (int8_t) i;
        _s_t<COSTTYPE, SIMDWIDTH> ex = v_extract<COSTTYPE, SIMDWIDTH>(
            utest, imm);
        ASSERT_TRUE((std::isnan(ex) &&
                    std::isnan(this->m_v_in[i])) ||
                    ex == this->m_v_in[i]);
    }
}

/* register test cases */
REGISTER_TYPED_TEST_CASE_P(mapMAPTestVectorMath,
    TestMemory,
    TestInit,
    TestRealArithmetique,
    TestIntegerArithmetique,
    TestLogical);

/* instantiate tests */
typedef ::testing::Types<
    #if defined(__SSE4_2__)
    TestTuple<float, 4>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<float, 8>,
    #endif /* __AVX__ */
    #if defined(__SSE4_2__)
    TestTuple<double, 2>,
    #endif /* __SSE4_2__ */
    #if defined(__AVX__)
    TestTuple<double, 4>,
    #endif /* __AVX__ */
    TestTuple<float, 1>,
    TestTuple<double, 1>
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(VectorMathTest, mapMAPTestVectorMath,
    TestTupleInstances);

NS_MAPMAP_END

