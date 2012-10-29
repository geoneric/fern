#pragma once
#include "Ranally/IO/Value.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PointValue:
  public Value
{

  friend class PointValueTest;

public:

                   PointValue          ();

                   ~PointValue         ();

private:

  // Store values per point feature id.

};

} // namespace ranally
