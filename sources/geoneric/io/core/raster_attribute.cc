#include "geoneric/io/raster_attribute.h"


namespace geoneric {

RasterAttribute::RasterAttribute(
    String const& name)

    : Attribute(name),
      _value()

{
}


RasterAttribute::~RasterAttribute()
{
}

} // namespace geoneric
