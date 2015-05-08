// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include "fern/core/data_type_traits.h"


namespace fern {

template<
    class T>
struct DataTypeTraits<
    std::vector<T>>
{

    using argument_category = array_1d_tag;

    template<
        class U>
    struct Clone
    {
        using type = std::vector<U>;
    };

    // Don't use vector's typedefs. Doing it like this will make it impossible
    // to use vector<bool>, which is Good. vector<bool> is nasty since it
    // doesn't store bools. Using it works out bad in combination with threads.

    // typename std::vector<T>::value_type;
    using value_type = T;

    // typename std::vector<T>::reference;
    using reference = T&;

    // typename std::vector<T>::const_reference;
    using const_reference = T const&;

    static bool const is_masking = false;

    static size_t const rank = 1u;

};

} // namespace fern
