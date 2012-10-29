#pragma once
#include "Ranally/IO/Feature.h"
#include "Ranally/IO/PolygonPtr.h"


namespace ranally {

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

} // namespace ranally
