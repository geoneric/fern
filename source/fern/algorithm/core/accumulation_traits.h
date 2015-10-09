// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace algorithm {

template<
    typename T>
struct AccumulationTraits;


template<>
struct AccumulationTraits<
    bool>
{
    using Type = uint64_t;
    static Type const zero = 0;
};


template<>
struct AccumulationTraits<
    int32_t>
{
    using Type = int64_t;
    static Type const zero = 0;
};


template<>
struct AccumulationTraits<
    float>
{
    using Type = double;
    static Type constexpr zero = 0.0;
};


template<>
struct AccumulationTraits<
    double>
{
    using Type = double;
    static Type constexpr zero = 0.0;
};


template<
    typename T>
using accumulate_type = typename AccumulationTraits<T>::Type;

} // namespace algorithm
} // namespace fern
