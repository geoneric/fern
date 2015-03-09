#pragma once
#include "fern/core/data_traits.h"


namespace fern {

template<
    class T>
typename DataTraits<T>::const_reference
                   get                 (T const& array,
                                        size_t index);

template<
    class T>
typename DataTraits<T>::reference
                   get                 (T const& array,
                                        size_t index);

} // namespace fern
