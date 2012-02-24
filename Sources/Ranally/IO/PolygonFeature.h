#ifndef INCLUDED_RANALLY_POLYGONFEATURE
#define INCLUDED_RANALLY_POLYGONFEATURE

#include <vector>
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

                   PolygonFeature      (PolygonDomainPtr const& domain);

                   ~PolygonFeature     ();

  PolygonDomain const& domain          () const;

private:

  PolygonDomainPtr _domain;

  // TODO Store attribute by name?
  std::vector<PolygonAttributePtr> _attributes;

};

} // namespace ranally

#endif
