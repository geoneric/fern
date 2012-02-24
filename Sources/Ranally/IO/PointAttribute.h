#ifndef INCLUDED_RANALLY_POINTATTRIBUTE
#define INCLUDED_RANALLY_POINTATTRIBUTE

#include <vector>
#include "Ranally/IO/Attribute.h"
#include "Ranally/IO/PointPtr.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PointAttribute:
  public Attribute
{

  friend class PointAttributeTest;

public:

                   PointAttribute      (PointDomainPtr const& domain);

                   ~PointAttribute     ();

private:

  PointDomainPtr   _domain;

  PointFeaturePtr  _feature;

  std::vector<PointValuePtr> _values;

};

} // namespace ranally

#endif
