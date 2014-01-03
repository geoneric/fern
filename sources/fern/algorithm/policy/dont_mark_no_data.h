#pragma once


namespace fern {

class DontMarkNoData {

public:

    static void    mark                (size_t index);

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


inline void DontMarkNoData::mark(
    size_t /* index */)
{
}

} // namespace fern
