// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>


namespace fern {
namespace algorithm {

template<
    typename Collection>
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
    typename Collection>
inline void CollectNoDataIndices<Collection>::mark(
    size_t index)
{
    _indices.push_back(index);
}


template<
    typename Collection>
inline Collection const& CollectNoDataIndices<Collection>::indices() const
{
    return _indices;
}

} // namespace algorithm
} // namespace fern
