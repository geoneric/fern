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
