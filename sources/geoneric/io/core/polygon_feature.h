#pragma once
#include "geoneric/io/core/feature.h"
#include "geoneric/io/core/polygon_ptr.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PolygonFeature:
    public Feature
{

    friend class PolygonFeatureTest;

public:

                   PolygonFeature      (String const& name,
                                        PolygonDomainPtr const& domain);

                   ~PolygonFeature     ();

    PolygonDomain const& domain          () const;

    PolygonAttributes const& attributes  () const;

private:

    PolygonDomainPtr _domain;

    PolygonAttributes _attributes;

};

} // namespace geoneric
