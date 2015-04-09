// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>


namespace fern {
namespace algorithm {

template<
    typename T>
inline constexpr int sign(
    T x,
    std::false_type /* is_signed */)
{
    static_assert(std::is_unsigned<T>::value, "T assumed to be unsigned");
    return 1;
}


template<
    typename T>
inline constexpr int sign(
    T x,
    std::true_type /* is_signed */)
{
    static_assert(std::is_signed<T>::value, "T assumed to be signed");
    return x >= 0 ? 1 : -1;
}


template<
    typename T>
inline constexpr int sign(
    T x)
{
    static_assert(std::is_arithmetic<T>::value, "T assumed to be arithmetic");
    return sign(x, std::is_signed<T>());
}

} // namespace algorithm
} // namespace fern
