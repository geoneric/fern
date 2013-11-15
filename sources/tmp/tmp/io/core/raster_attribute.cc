#include "fern/io/core/raster_attribute.h"


namespace fern {

RasterAttribute::RasterAttribute(
    String const& name)

    : Attribute(name),
      _value()

{
}


RasterAttribute::~RasterAttribute()
{
}

} // namespace fern
