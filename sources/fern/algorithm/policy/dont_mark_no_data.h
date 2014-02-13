#pragma once


namespace fern {

class DontMarkNoData {

public:

    static constexpr bool is_no_data   (size_t index);

    static void    mark_as_no_data     (size_t index);

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


inline void DontMarkNoData::mark_as_no_data(
    size_t /* index */)
{
}

} // namespace fern
