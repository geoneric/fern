#pragma once
#include <cassert>
#include <cxxabi.h>
#include "fern/core/type_traits.h"


namespace fern {
namespace detail {

std::string demangle(
    std::string const& name)
{
    int status;
    char* buffer;
    buffer = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
    assert(status == 0);
    std::string real_name(buffer);
    free(buffer);
    return real_name;
}


template<
    class T>
String demangled_type_name()
{
    return demangle(typeid(T).name());
}


template<
    class T >
String typename_(
    std::true_type /* builtin */)
{
    return TypeTraits<T>::name;

}


template<
    class T >
String typename_(
    std::false_type /* builtin */)
{
    return demangled_type_name<T>();
}

} // namespace detail


template<
    class T>
String typename_()
{
    return detail::typename_<T>(typename std::integral_constant<bool, TypeTraits<T>::builtin>());
}

} // namespace fern
