#pragma once
#include <algorithm>


namespace fern {

template<
    class Container>
void sort(
    Container& container)
{
    std::sort(container.begin(), container.end());
}


template<
    class Container,
    class Predicate>
void sort(
    Container& container,
    Predicate predicate)
{
    std::sort(container.begin(), container.end(), predicate);
}

} // namespace fern
