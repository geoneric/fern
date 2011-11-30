#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATION_REQUIREMENTS
#define INCLUDED_RANALLY_LANGUAGE_OPERATION_REQUIREMENTS

#include <vector>
#include <boost/noncopyable.hpp>
#include "Ranally/Language/Operation/Argument.h"
#include "Ranally/Language/Operation/Result.h"



namespace ranally {
namespace language {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Requirements:
  private boost::noncopyable
{

  friend class RequirementsTest;

public:

                   Requirements        ();

                   ~Requirements       ();

  std::vector<Argument> const& arguments();

  std::vector<Result> const& results   ();

private:

  std::vector<Argument> _arguments;

  std::vector<Result> _results;

};

} // namespace operation
} // namespace language
} // namespace ranally

#endif
