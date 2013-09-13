#pragma once

#include <memory>
#include "geoneric/io/gdal/attribute.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Domain,
    class Values>
class SpatialAttribute:
    public Attribute
{

public:

                   SpatialAttribute    ()=default;

                   SpatialAttribute    (SpatialAttribute const&)=delete;

    SpatialAttribute& operator=        (SpatialAttribute const&)=delete;

                   SpatialAttribute    (SpatialAttribute&&)=delete;

    SpatialAttribute& operator=        (SpatialAttribute&&)=delete;

                   ~SpatialAttribute   ()=default;

    void           add                 (typename Domain::Geometry const& geometry,
                                        typename Values::Value const& value);

private:

    std::unique_ptr<Domain> _domain;

    std::unique_ptr<Values> _values;

};


template<
    class Domain,
    class Values>
void SpatialAttribute<Domain, Values>::add(
    typename Domain::Geometry const& geometry,
    typename Values::Value const& value)
{
    typename Domain::GID gid = _domain->add(geometry);
    _values->add(gid, value);
}

} // namespace geoneric
