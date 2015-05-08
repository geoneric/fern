// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_type_traits.h"
#include "fern/core/type_traits.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Input no-data policy class that detect no-data by the
                default, type-dependent, no-data value.
    @tparam     Collection Collection containing the values.

    This class keeps a reference to a collection; it doesn't copy the
    collection.
*/
template<
    typename Collection>
class DetectNoData
{

private:

    using value_type = fern::value_type<Collection>;

public:

    bool           is_no_data          () const;

    bool           is_no_data          (size_t index) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2,
                                        size_t index3) const;

                   DetectNoData        (DetectNoData const&)=default;

                   DetectNoData        (Collection const& collection);

                   ~DetectNoData       ()=default;

    DetectNoData&  operator=           (DetectNoData const&)=default;

protected:

                   DetectNoData        ()=delete;

                   DetectNoData        (DetectNoData&& other)=default;

    DetectNoData&  operator=           (DetectNoData&&)=default;

private:

    Collection const&    _collection;

};


template<
    typename Collection>
inline DetectNoData<Collection>::DetectNoData(
    Collection const& collection)

    : _collection(collection)

{
}


template<
    typename Collection>
inline bool DetectNoData<Collection>::is_no_data() const
{
    return fern::is_no_data(get(_collection));
}


template<
    typename Collection>
inline bool DetectNoData<Collection>::is_no_data(
    size_t index) const
{
    return fern::is_no_data(get(_collection, index));
}


template<
    typename Collection>
inline bool DetectNoData<Collection>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return fern::is_no_data(get(_collection, index1, index2));
}


template<
    typename Collection>
inline bool DetectNoData<Collection>::is_no_data(
    size_t index1,
    size_t index2,
    size_t index3) const
{
    return fern::is_no_data(get(_collection, index1, index2, index3));
}

} // namespace algorithm
} // namespace fern
