#pragma once
#include "fern/core/argument_traits.h"


namespace fern {

template<
    class T>
typename ArgumentTraits<T>::const_reference
                   get                 (T const& array,
                                        size_t index1,
                                        size_t index2);

template<
    class T>
typename ArgumentTraits<T>::reference
                   get                 (T const& array,
                                        size_t index1,
                                        size_t index2);

} // namespace fern
