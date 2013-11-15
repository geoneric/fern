#pragma once
#include "fern/io/core/value.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PolygonValue:
    public Value
{

    friend class PolygonValueTest;

public:

                   PolygonValue        ();

                   ~PolygonValue       ();

private:

    // Store values per polygon feature id.

    // This class must provide clients with value type information and a
    // template function to get at the collection with correctly typed values.
    // This class itself must not be a template.
    // Values should probably be stored in arrays, one array per PolygonValue
    // instance. A second array should contain the feature-id's and should be
    // shared between multiple PolygonValue instances. This array may be passed
    // on from the PolygonDomain(?).

};

} // namespace fern
