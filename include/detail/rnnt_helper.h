#pragma once

#include <limits>
#include <algorithm>
#include <cmath>

#include "hostdevice.h"

namespace rnnt_helper {

static const float threshold = 1e-1;

template<typename T>
inline HOSTDEVICE T neg_inf() { return -T(INFINITY); }

template<typename T>
inline HOSTDEVICE T log_sum_exp(T a, T b) {
    if (a == neg_inf<T>()) return b;
    if (b == neg_inf<T>()) return a;
    if (a > b)
        return log1p(exp(b-a)) + a;
    else
        return log1p(exp(a-b)) + b;
}

inline int div_up(int x, int y) {
    return (x + y - 1) / y;
}

template <typename Arg, typename Res = Arg> struct maximum {
    HOSTDEVICE
    Res operator()(const Arg& x, const Arg& y) const {
        return x < y ? y : x;
    }
};

template <typename Arg, typename Res = Arg> struct add {
    HOSTDEVICE
    Res operator()(const Arg& x, const Arg& y) const {
        return x + y;
    }
};

template <typename Arg, typename Res = Arg> struct identity {
    HOSTDEVICE Res operator()(const Arg& x) const {return Res(x);}
};

template <typename Arg, typename Res = Arg> struct negate {
    HOSTDEVICE Res operator()(const Arg& x) const {return Res(-x);}
};

template <typename Arg, typename Res = Arg> struct exponential {
    HOSTDEVICE Res operator()(const Arg& x) const {return std::exp(x);}
};

template<typename Arg1, typename Arg2 = Arg1, typename Res=Arg1>
struct log_plus {
    typedef Res result_type;
    HOSTDEVICE
    Res operator()(const Arg1& p1, const Arg2& p2) {
        if (p1 == neg_inf<Arg1>())
            return p2;
        if (p2 == neg_inf<Arg2>())
            return p1;
        Res result = log1p(exp(-fabs(p1 - p2))) + maximum<Res>()(p1, p2);
        return result;
    }
};

}