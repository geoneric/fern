#pragma once
#include <cstddef>


class SkipNoData
{

public:

    static constexpr bool is_no_data   ();

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

                   SkipNoData          ()=default;

                   SkipNoData          (SkipNoData const&)=delete;

                   SkipNoData          (SkipNoData&&)=default;

                   ~SkipNoData         ()=default;

   SkipNoData&     operator=           (SkipNoData const&)=delete;

   SkipNoData&     operator=           (SkipNoData&&)=delete;

};


inline constexpr bool SkipNoData::is_no_data()
{
    return false;
}


inline constexpr bool SkipNoData::is_no_data(
    size_t /* index */)
{
    return false;
}


inline constexpr bool SkipNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
    return false;
}
