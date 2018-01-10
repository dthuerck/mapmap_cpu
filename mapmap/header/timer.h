/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_TIMER_H_
#define __MAPMAP_TIMER_H_

#include "header/defines.h"

NS_MAPMAP_BEGIN

using t_point = std::chrono::time_point<std::chrono::system_clock>;
struct _timer
{
    _timer();
    ~_timer();

    void start(const std::string& s);
    void stop(const std::string& s);
    double get_ms(const std::string& s);

    std::map<std::string, t_point> m_t_start;
    std::map<std::string, t_point> m_t_end;
};

extern _timer * __T;

#define START_TIMER(str) (__T->start(str));
#define STOP_TIMER(str) (__T->stop(str));
#define PRINT_TIMER(str) (printf("(Timing) %s: %f ms\n", str, __T->get_ms(str)));
#define GET_TIMER(str) (__T->get_ms(str));

NS_MAPMAP_END

/* include timer implementation */
#include "source/timer.impl.h"

#endif /* __MAPMAP_TIMER_H_ */