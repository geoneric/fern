#pragma once
#include <cstddef>
#include "fern/algorithm/policy/input_no_data_policies.h"


namespace fern {
namespace algorithm {

template<
    class... ArgumentNoDataPolicies>
class SkipNoData:
    public InputNoDataPolicies<ArgumentNoDataPolicies...>
{

public:

    static constexpr bool is_no_data   ();

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   SkipNoData          (ArgumentNoDataPolicies&&... policies);

                   SkipNoData          (SkipNoData const&)=delete;

                   SkipNoData          (SkipNoData&&)=default;

    SkipNoData&
                   operator=           (SkipNoData const&)=delete;

    SkipNoData&    operator=           (SkipNoData&&)=default;

                   ~SkipNoData         ()=default;

    /// template<
    ///     size_t index>
    /// SkipNoData const&
    ///                get                 () const;

};


template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data()
{
    return false;
}


template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index */)
{
    return false;
}


template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
    return false;
}


template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
    return false;
}


/// template<
///     size_t index>
/// inline SkipNoData const& SkipNoData::get() const
/// {
///     return *this;
/// }


template<
    class... ArgumentNoDataPolicies>
inline SkipNoData<ArgumentNoDataPolicies...>::SkipNoData(
    ArgumentNoDataPolicies&&... policies)

    : InputNoDataPolicies<ArgumentNoDataPolicies...>(
          std::forward<ArgumentNoDataPolicies>(policies)...)

{
}

} // namespace algorithm
} // namespace fern
