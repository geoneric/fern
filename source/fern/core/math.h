#pragma once


namespace fern {

template<
    typename T>
inline bool is_equal(
    T const value1,
    T const value2)
{
    return value1 == value2;
}


template<>
inline bool is_equal(
    float const value1,
    float const value2)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return value1 == value2;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}


template<>
inline bool is_equal(
    double const value1,
    double const value2)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return value1 == value2;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}


template<
    typename T>
inline bool is_not_equal(
    T const value1,
    T const value2)
{
    return value1 != value2;
}


template<>
inline bool is_not_equal(
    float const value1,
    float const value2)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return value1 != value2;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}


template<>
inline bool is_not_equal(
    double const value1,
    double const value2)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return value1 != value2;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

}  // namespace fern
