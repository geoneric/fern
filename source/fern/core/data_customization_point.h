#pragma once
#include <cstddef>


namespace fern {

// Declarations of functions that are used in the implementation of operations.
// These are not defined. For each collection type they need to be implemented.
// See also vector_traits.h, array_traits.h, ...

template<
    class T>
size_t             size                (T const& collection);

// Convert 2D indices to linear index.
template<
    class T>
size_t             index               (T const& collection,
                                        size_t index1,
                                        size_t index2);

// For constant only.
template<
    class T>
T const&           get                 (T const&);

// For constant only.
template<
    class T>
T&                 get                 (T&);

template<
    class T>
T const&           get                 (T const& collection,
                                        size_t index);

template<
    class T>
T&                 get                 (T& collection,
                                        size_t index);

} // namespace fern
