#ifndef INCLUDED_RANALLY_OPERATION_RESULT
#define INCLUDED_RANALLY_OPERATION_RESULT

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"



namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Result:
  private boost::noncopyable
{

  friend class ResultTest;

public:

                   Result              (UnicodeString const& name,
                                        UnicodeString const& description,
                                        DataType const& dataType,
                                        ValueType const& valueType);

                   ~Result             ();

private:

  UnicodeString    _name;

  UnicodeString    _description;

  DataType         _dataType;

  ValueType        _valueType;

};

} // namespace operation
} // namespace ranally

#endif
