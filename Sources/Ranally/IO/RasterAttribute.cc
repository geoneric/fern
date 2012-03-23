#include "Ranally/IO/RasterAttribute.h"



namespace ranally {

RasterAttribute::RasterAttribute(
  UnicodeString const& name)

  : Attribute(name),
    _value()

{
}



RasterAttribute::~RasterAttribute()
{
}

} // namespace ranally

