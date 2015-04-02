#pragma once
#include <cstddef>


class DontMarkNoData {

public:

    static void    mark_as_no_data     ();

    static void    mark_as_no_data     (size_t index);

    static void    mark_as_no_data     (size_t index1,
                                        size_t index2);

                   DontMarkNoData      ()=default;

                   DontMarkNoData      (DontMarkNoData const&)=delete;

                   DontMarkNoData      (DontMarkNoData&&)=default;

                   ~DontMarkNoData     ()=default;

    DontMarkNoData&
                   operator=           (DontMarkNoData&&)=default;

    DontMarkNoData&
                   operator=           (DontMarkNoData const&)=delete;

};


inline void DontMarkNoData::mark_as_no_data()
{
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
