#pragma once


namespace fern {

template<
    typename T>
bool is_equal(
    T const value1,
    T const value2)
{
    return value1 == value2;
}


template<>
bool is_equal(
    float const value1,
    float const value2)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return value1 == value2;
#pragma GCC diagnostic pop
}


template<>
bool is_equal(
    double const value1,
    double const value2)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return value1 == value2;
#pragma GCC diagnostic pop
}


template<
    typename T>
bool is_not_equal(
    T const value1,
    T const value2)
{
    return value1 != value2;
}


template<>
bool is_not_equal(
    float const value1,
    float const value2)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return value1 != value2;
#pragma GCC diagnostic pop
}


template<>
bool is_not_equal(
    double const value1,
    double const value2)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return value1 != value2;
#pragma GCC diagnostic pop
}

}  // namespace fern
