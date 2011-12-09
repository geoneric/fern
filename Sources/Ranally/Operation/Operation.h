#ifndef INCLUDED_RANALLY_OPERATION_OPERATION
#define INCLUDED_RANALLY_OPERATION_OPERATION

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>



namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Operation:
  private boost::noncopyable
{

  friend class OperationTest;

public:

                   Operation           (UnicodeString const& name);

                   ~Operation          ();

  UnicodeString const& name            () const;

private:

  UnicodeString    _name;

};



typedef boost::shared_ptr<Operation> OperationPtr;

} // namespace operation
} // namespace ranally

#endif
