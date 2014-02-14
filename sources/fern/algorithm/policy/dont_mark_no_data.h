#pragma once


namespace fern {

class DontMarkNoData {

public:

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2,
                                        size_t index3);

    static void    mark_as_no_data     (size_t index);

    static void    mark_as_no_data     (size_t index1,
                                        size_t index2);

    static void    mark_as_no_data     (size_t index1,
                                        size_t index2,
                                        size_t index3);

protected:

                   DontMarkNoData      ()=default;

                   DontMarkNoData      (DontMarkNoData&&)=delete;

    DontMarkNoData&
                   operator=           (DontMarkNoData&&)=delete;

                   DontMarkNoData      (DontMarkNoData const&)=delete;

    DontMarkNoData&
                   operator=           (DontMarkNoData const&)=delete;

                   ~DontMarkNoData     ()=default;

private:

};


inline constexpr bool DontMarkNoData::is_no_data(
    size_t /* index */)
{
    return false;
}


inline constexpr bool DontMarkNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
    return false;
}


inline constexpr bool DontMarkNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
    return false;
}


inline void DontMarkNoData::mark_as_no_data(
    size_t /* index */)
{
}


inline void DontMarkNoData::mark_as_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
}


inline void DontMarkNoData::mark_as_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
}

} // namespace fern
