// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/language/feature/core/attribute.h"
#include "fern/language/feature/core/geometry_values.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    typename Domain,
    typename Value>
class SpatialAttribute:
    public Attribute
{

public:

    LOKI_DEFINE_CONST_VISITABLE()

    using GID = typename Domain::GID;

                   SpatialAttribute    ();

                   SpatialAttribute    (SpatialAttribute const&)=delete;

    SpatialAttribute& operator=        (SpatialAttribute const&)=delete;

                   SpatialAttribute    (SpatialAttribute&&)=delete;

    SpatialAttribute& operator=        (SpatialAttribute&&)=delete;

                   ~SpatialAttribute   ()=default;

    GID            add                 (typename Domain::Geometry const&
                                            geometry,
                                        Value const& value);

    bool           empty               () const;

    size_t         size                () const;

    Domain const&  domain              () const;

    GeometryValues<typename Domain::GID, Value> const& values() const;

private:

    std::unique_ptr<Domain> _domain;

    std::unique_ptr<GeometryValues<typename Domain::GID, Value>> _values;

};


template<
    typename Domain,
    typename Value>
inline SpatialAttribute<Domain, Value>::SpatialAttribute()

    : _domain(std::make_unique<Domain>()),
      _values(std::make_unique<GeometryValues<typename Domain::GID, Value>>())

{
}


template<
    typename Domain,
    typename Value>
inline auto SpatialAttribute<Domain, Value>::add(
    typename Domain::Geometry const& geometry,
    Value const& value) -> GID
{
    assert((domain().empty() == values().empty()) || values().empty());

    GID gid = _domain->add(geometry);
    _values->add(gid, value);

    return gid;
}


template<
    typename Domain,
    typename Value>
inline Domain const&  SpatialAttribute<Domain, Value>::domain() const
{
    return *_domain;
}


template<
    typename Domain,
    typename Value>
inline GeometryValues<typename Domain::GID, Value> const&
    SpatialAttribute<Domain, Value>::values() const
{
    return *_values;
}


template<
    typename Domain,
    typename Value>
bool SpatialAttribute<Domain, Value>::empty() const
{
    assert((domain().empty() == values().empty()) || values().empty());

    return domain().empty();
}


template<
    typename Domain,
    typename Value>
size_t SpatialAttribute<Domain, Value>::size() const
{
    assert((domain().size() == values().size()) || values().empty());

    return domain().size();
}

} // namespace language
} // namespace fern
