#pragma once
#include "geoneric/io/core/value.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  A raster value is a collection of values, arranged in a grid. No information
  about where these values are positioned in space or time is stored here.
  For that, query the Domain that is associated with this value in the
  enclosing RasterAttribute.

  \sa        .
*/
class RasterValue:
    public Value
{

    friend class RasterValueTest;

public:

                   RasterValue         ();

                  ~RasterValue         ();

private:

};

} // namespace geoneric
