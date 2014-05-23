#pragma once
#include <cstddef>


namespace fern {

template<
    class Collection>
class CollectNoDataIndices {

public:

    void           mark                (size_t index);

    Collection const&
                   indices             () const;

protected:

                   CollectNoDataIndices()=default;

                   CollectNoDataIndices(CollectNoDataIndices&&)=delete;

    CollectNoDataIndices&
                   operator=           (CollectNoDataIndices&&)=delete;

                   CollectNoDataIndices(CollectNoDataIndices const&)=delete;

    CollectNoDataIndices&
                   operator=           (CollectNoDataIndices const&)=delete;

                   ~CollectNoDataIndices()=default;

private:

    Collection     _indices;

};


template<
    class Collection>
inline void CollectNoDataIndices<Collection>::mark(
    size_t index)
{
    _indices.push_back(index);
}


template<
    class Collection>
inline Collection const& CollectNoDataIndices<Collection>::indices() const
{
    return _indices;
}

} // namespace fern
