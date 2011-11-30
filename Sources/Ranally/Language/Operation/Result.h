#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATION_RESULT
#define INCLUDED_RANALLY_LANGUAGE_OPERATION_RESULT

#include "Ranally/Language/Operation/Parameter.h"



namespace ranally {
namespace language {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Result:
  public Parameter
{

  friend class ResultTest;

public:

                   Result              ();

                   ~Result             ();

protected:

private:

};

} // namespace operation
} // namespace language
} // namespace ranally

#endif
