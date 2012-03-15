#ifndef INCLUDED_RANALLY_POINTFEATURE
#define INCLUDED_RANALLY_POINTFEATURE

#include "Ranally/IO/Feature.h"
#include "Ranally/IO/PointPtr.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PointFeature:
  public Feature
{

  friend class PointFeatureTest;

public:

                   PointFeature        (UnicodeString const& name,
                                        PointDomainPtr const& domain);

                   ~PointFeature       ();

  PointDomain const& domain            () const;

  PointAttributes const& attributes    () const;

private:

  PointDomainPtr   _domain;

  PointAttributes  _attributes;

};

} // namespace ranally

#endif
