//
// Created by chenkaihuang on 6/7/22.
//



#ifndef CONV_CLUSTERING_UTILS_H
#define CONV_CLUSTERING_UTILS_H


#include <utility>
#include <iterator>
#include <algorithm>
#include <time.h>
//#include <sys/time.h>
#include <chrono>
/** Function operator.
    Returns true if t1.first &lt; t2.first (i.e., increasing). */
template < class S, class T >
class mFirstLess_2 {
public:
    /// Compare function
    inline bool operator()(const std::pair< S, T > &t1,
                           const std::pair< S, T > &t2) const
    {
        return t1.first < t2.first;
    }
};
//-----------------------------------------------------------------------------
/** Function operator.
    Returns true if t1.first &gt; t2.first (i.e, decreasing). */
template < class S, class T >
class mFirstGreater_2 {
public:
    /// Compare function
    inline bool operator()(const std::pair< S, T > &t1,
                           const std::pair< S, T > &t2) const
    {
        return t1.first > t2.first;
    }
};


template < class T >
inline void
vMemcpy(const T *from, const int size, T *to)
{
    if (size == 0 || from == to)
        return;

    for (int n = static_cast<int>(size>>3); n > 0; --n, from += 8, to += 8) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
        to[3] = from[3];
        to[4] = from[4];
        to[5] = from[5];
        to[6] = from[6];
        to[7] = from[7];
    }
    switch (size % 8) {
        case 7:
            to[6] = from[6];
        case 6:
            to[5] = from[5];
        case 5:
            to[4] = from[4];
        case 4:
            to[3] = from[3];
        case 3:
            to[2] = from[2];
        case 2:
            to[1] = from[1];
        case 1:
            to[0] = from[0];
        case 0:
            break;
    }
}

template < class S, class T, class mCompare >
/// sort from sfirst to slast, t will also change at same order.
void mSort(S *sfirst, S *slast, T *tfirst, const mCompare &pc)
{
    const size_t len = std::distance(sfirst, slast);
    if (len <= 1)
        return;

    typedef std::pair< S, T > ST_pair;
    ST_pair *x = static_cast< ST_pair * >(::operator new(len * sizeof(ST_pair)));
#ifdef ZEROFAULT
    // Can show RUI errors on some systems due to copy of ST_pair with gaps.
  // E.g., <int, double> has 4 byte alignment gap on Solaris/SUNWspro.
  memset(x, 0, (len * sizeof(ST_pair)));
#endif

    size_t i = 0;
    S *scurrent = sfirst;
    T *tcurrent = tfirst;
    while (scurrent != slast) {
        new (x + i++) ST_pair(*scurrent++, *tcurrent++);
    }

    std::sort(x, x + len, pc);

    scurrent = sfirst;
    tcurrent = tfirst;
    for (i = 0; i < len; ++i) {
        *scurrent++ = x[i].first;
        *tcurrent++ = x[i].second;
    }

    ::operator delete(x);
}

template < class S, class T >
void mSort(S *sfirst, S *slast, T *tfirst)
{
    mSort(sfirst, slast, tfirst, mFirstLess_2< S, T >());
}

inline
std::chrono::steady_clock::time_point time_now()
{
//    timeval res{};
//    gettimeofday(&res, NULL);
    using namespace std::chrono;
    return steady_clock::now();
}

inline
double time_since(std::chrono::steady_clock::time_point clock_begin) {
    using namespace std::chrono;
    steady_clock::time_point clock_end = steady_clock::now();
    steady_clock::duration time_span = clock_end - clock_begin;
    return double(time_span.count()) * steady_clock::period::num / steady_clock::period::den;
}

#endif //CONV_CLUSTERING_UTILS_H
