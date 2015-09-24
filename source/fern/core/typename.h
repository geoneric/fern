// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#if !defined(_MSC_VER)
#include <cxxabi.h>  // abi::__cxa_demangle
#endif
#include "fern/core/type_traits.h"


namespace fern {
namespace detail {

std::string demangle(
    std::string const& name)
{
#if defined(_MSC_VER)
    // TODO
    return name;
#else
    int status;
    char* buffer;
    buffer = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
    assert(status == 0);
    std::string real_name(buffer);
    free(buffer);
    return real_name;
#endif
}


template<
    class T>
std::string demangled_type_name()
{
    return demangle(typeid(T).name());
}


template<
    class T >
std::string typename_(
    std::true_type /* builtin */)
{
    return TypeTraits<T>::name;

}


template<
    class T >
std::string typename_(
    std::false_type /* builtin */)
{
    return demangled_type_name<T>();
}

} // namespace detail


template<
    class T>
std::string typename_()
{
    return detail::typename_<T>(
        typename std::integral_constant<bool, TypeTraits<T>::builtin>());
}

} // namespace fern
