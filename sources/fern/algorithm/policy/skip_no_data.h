#pragma once


namespace fern {

class SkipNoData {

public:

    static constexpr bool is_no_data   ();

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   SkipNoData          ()=default;

                   SkipNoData          (SkipNoData const&)=default;

                   SkipNoData          (SkipNoData&&)=default;

    SkipNoData&
                   operator=           (SkipNoData const&)=default;

    SkipNoData&    operator=           (SkipNoData&&)=default;

    virtual        ~SkipNoData         ()=default;

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


inline constexpr bool SkipNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
    return false;
}

} // namespace fern