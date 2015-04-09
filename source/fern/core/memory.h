// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <type_traits>


namespace std {

//! Create and return a std::unique_ptr instance.
/*!
  This function template must be removed once the compiler supports this out
  of the box.
*/
template<
    class T,
    class ...Args>
inline typename std::enable_if<
    !std::is_array<T>::value,
    std::unique_ptr<T>>::type
make_unique(
    Args&& ...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


template<
    class T>
inline typename std::enable_if<
    std::is_array<T>::value,
    std::unique_ptr<T>>::type
make_unique(
    std::size_t n)
{
    using RT = typename std::remove_extent<T>::type;
    return std::unique_ptr<T>(new RT[n]);
}

} // namespace std
