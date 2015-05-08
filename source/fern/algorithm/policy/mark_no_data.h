// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include "fern/core/data_type_traits.h"
#include "fern/core/type_traits.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Output no-data policy class that marks no-data using the
                default, type-dependent, no-data value.
    @tparam     Collection Collection receiving the no-data values.

    This class keeps a reference to a collection; it doesn't copy the
    collection. So, copy construction and copy assignment are not supported.
*/
template<
    typename Collection>
class MarkNoData {

private:

    using value_type = fern::value_type<Collection>;

public:

    void           mark_as_no_data     ();

    void           mark_as_no_data     (size_t index);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   MarkNoData          (Collection& collection);

                   ~MarkNoData         ()=default;

protected:

                   MarkNoData          ()=delete;

                   MarkNoData          (MarkNoData const&)=delete;

                   MarkNoData          (MarkNoData&&)=default;

    MarkNoData&    operator=           (MarkNoData const&)=delete;

    MarkNoData&    operator=           (MarkNoData&&)=default;

private:

    Collection&    _collection;

};


template<
    typename Collection>
inline MarkNoData<Collection>::MarkNoData(
    Collection& collection)

    : _collection(collection)

{
}


template<
    typename Collection>
inline void MarkNoData<Collection>::mark_as_no_data()
{
    // In case of a compile error, make sure that get is overloaded for
    // Collection. This is not the case for regular constants. You may need to
    // pick a type like MaskedConstant, which supports masking.
    set_no_data(_collection);
}


template<
    typename Collection>
inline void MarkNoData<Collection>::mark_as_no_data(
    size_t index)
{
    set_no_data(get(_collection, index));
}


template<
    typename Collection>
inline void MarkNoData<Collection>::mark_as_no_data(
    size_t index1,
    size_t index2)
{
    set_no_data(get(_collection, index1, index2));
}


template<
    typename Collection>
inline void MarkNoData<Collection>::mark_as_no_data(
    size_t index1,
    size_t index2,
    size_t index3)
{
    set_no_data(get(_collection, index1, index2, index3));
}

} // namespace algorithm
} // namespace fern
