#pragma once
#include "fern/core/argument_categories.h"


namespace fern {

/// // Argument categories. Used in tag dispatching.
/// struct constant_tag {};
/// struct collection_tag {};
/// struct array_1d_tag: collection_tag {};
/// struct array_2d_tag: collection_tag {};
/// struct array_3d_tag: collection_tag {};


template<
    class T>
struct ArgumentTraits
{

    //! By default, we grab T's value type. Specialize if needed.
    using value_type = typename T::value_type;

    //! By default, we grab T's reference type. Specialize if needed.
    using reference = typename T::reference;

    //! By default, we grab T's const_reference type. Specialize if needed.
    using const_reference = typename T::const_reference;

};

} // namespace fern
