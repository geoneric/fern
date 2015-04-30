// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <map>
#include <memory>
#include "fern/language/feature/core/domain.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    typename Geometry_>
class SpatialDomain:
    private std::map<size_t, Geometry_>,
    public Domain
{

public:

    using Geometry = Geometry_;

    using GID = size_t;

    using const_iterator = typename std::map<size_t, Geometry_>::const_iterator;

                   SpatialDomain       ()=default;

                   SpatialDomain       (SpatialDomain const&)=delete;

    SpatialDomain& operator=           (SpatialDomain const&)=delete;

                   SpatialDomain       (SpatialDomain&&)=delete;

    SpatialDomain& operator=           (SpatialDomain&&)=delete;

                   ~SpatialDomain      ()=default;

    const_iterator cbegin              () const;

    const_iterator begin               () const;

    const_iterator cend                () const;

    const_iterator end                 () const;

    // std::vector<GID> const& gids       () const;

    GID            add                 (Geometry_ const& geometry);

    Geometry_ const& geometry          (GID const& gid);

    bool           empty               () const;

    size_t         size                () const;

private:

    // std::map<GID, Geometry_> _geometries;

};


// template<
//     typename Geometry_>
// inline auto SpatialDomain<Geometry_>::gids() const -> std::vector<GID> const&
// {
//     return _gids;
// }


template<
    typename Geometry_>
inline auto SpatialDomain<Geometry_>::cbegin() const -> const_iterator
{
    return std::map<GID, Geometry_>::cbegin();
}


template<
    typename Geometry_>
inline auto SpatialDomain<Geometry_>::begin() const -> const_iterator
{
    return std::map<GID, Geometry_>::begin();
}


template<
    typename Geometry_>
inline auto SpatialDomain<Geometry_>::cend() const -> const_iterator
{
    return std::map<GID, Geometry_>::cend();
}


template<
    typename Geometry_>
inline auto SpatialDomain<Geometry_>::end() const -> const_iterator
{
    return std::map<GID, Geometry_>::end();
}


template<
    typename Geometry_>
inline auto SpatialDomain<Geometry_>::add(
    Geometry_ const& geometry) -> GID
{
    GID gid = size();
    this->insert(std::make_pair(gid, geometry));
    return gid;
}


template<
    typename Geometry_>
inline Geometry_ const& SpatialDomain<Geometry_>::geometry(
    GID const& gid)
{
    assert(this->find(gid) != this->end());
    return this->at(gid);
}


template<
    typename Geometry_>
inline bool SpatialDomain<Geometry_>::empty() const
{
    return std::map<GID, Geometry_>::empty();
}


template<
    typename Geometry_>
inline size_t SpatialDomain<Geometry_>::size() const
{
    return std::map<GID, Geometry_>::size();
}

} // namespace fern
