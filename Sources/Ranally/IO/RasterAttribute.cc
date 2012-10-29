#include "Ranally/IO/RasterAttribute.h"


namespace ranally {

RasterAttribute::RasterAttribute(
    String const& name)

    : Attribute(name),
      _value()

{
}


RasterAttribute::~RasterAttribute()
{
}

} // namespace ranally
