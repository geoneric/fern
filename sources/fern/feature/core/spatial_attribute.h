#pragma once
#include <memory>
#include "fern/core/memory.h"
#include "fern/feature/core/attribute.h"
#include "fern/feature/core/geometry_values.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Domain,
    class Value>
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
    class Domain,
    class Value>
inline SpatialAttribute<Domain, Value>::SpatialAttribute()

    : _domain(std::make_unique<Domain>()),
      _values(std::make_unique<GeometryValues<typename Domain::GID, Value>>())

{
}


template<
    class Domain,
    class Value>
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
    class Domain,
    class Value>
inline Domain const&  SpatialAttribute<Domain, Value>::domain() const
{
    return *_domain;
}


template<
    class Domain,
    class Value>
inline GeometryValues<typename Domain::GID, Value> const&
    SpatialAttribute<Domain, Value>::values() const
{
    return *_values;
}


template<
    class Domain,
    class Value>
bool SpatialAttribute<Domain, Value>::empty() const
{
    assert((domain().empty() == values().empty()) || values().empty());

    return domain().empty();
}


template<
    class Domain,
    class Value>
size_t SpatialAttribute<Domain, Value>::size() const
{
    assert((domain().size() == values().size()) || values().empty());

    return domain().size();
}

} // namespace fern
