#ifndef INCLUDED_RANALLY_POLYGONATTRIBUTE
#define INCLUDED_RANALLY_POLYGONATTRIBUTE

#include "Ranally/IO/Attribute.h"
#include "Ranally/IO/PolygonPtr.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PolygonAttribute:
  public Attribute
{

  friend class PolygonAttributeTest;

public:

                   PolygonAttribute    (String const& name,
                                        PolygonDomainPtr const& domain);

                   ~PolygonAttribute   ();

  PolygonDomainPtr const& domain       () const;

  PolygonFeaturePtr const& feature     () const;

  std::vector<PolygonValuePtr> values  () const;

private:

  PolygonDomainPtr _domain;

  PolygonFeaturePtr _feature;

  std::vector<PolygonValuePtr> _values;

};

} // namespace ranally

#endif
