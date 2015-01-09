#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/input_no_data_policies.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Input no-data policy class that detect no-data by the
                default, type-dependent, no-data value.
    @tparam     Collection Collection containing the values.

    This class keeps a reference to a collection; it doesn't copy the
    collection. So, copy construction and copy assignment are not supported.
*/
template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
class DetectNoData:
    public InputNoDataPolicies<ArgumentNoDataPolicies...>
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

                   DetectNoData        (Collection const& collection,
                                        ArgumentNoDataPolicies&&... policies);

    virtual        ~DetectNoData       ()=default;

                   DetectNoData        (DetectNoData&& other)=default;

protected:

                   DetectNoData        ()=delete;

                   DetectNoData        (DetectNoData const&)=delete;

    DetectNoData&  operator=           (DetectNoData const&)=delete;

    DetectNoData&  operator=           (DetectNoData&&)=default;

private:

    Collection const&    _collection;

};


template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
inline DetectNoData<Collection, ArgumentNoDataPolicies...>::
        DetectNoData(
    Collection const& collection,
    ArgumentNoDataPolicies&&... policies)

    : InputNoDataPolicies<ArgumentNoDataPolicies...>(
          std::forward<ArgumentNoDataPolicies>(policies)...),
      _collection(collection)

{
}


template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
inline bool DetectNoData<Collection, ArgumentNoDataPolicies...>
        ::is_no_data() const
{
    return fern::is_no_data(get(_collection));
}


template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
inline bool DetectNoData<Collection, ArgumentNoDataPolicies...>::is_no_data(
    size_t index) const
{
    return fern::is_no_data(get(_collection, index));
}


template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
inline bool DetectNoData<Collection, ArgumentNoDataPolicies...>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return fern::is_no_data(get(_collection, index1, index2));
}


template<
    typename Collection,
    typename... ArgumentNoDataPolicies>
inline bool DetectNoData<Collection, ArgumentNoDataPolicies...>::is_no_data(
    size_t index1,
    size_t index2,
    size_t index3) const
{
    return fern::is_no_data(get(_collection, index1, index2, index3));
}

} // namespace algorithm
} // namespace fern
