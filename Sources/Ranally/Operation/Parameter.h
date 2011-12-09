#ifndef INCLUDED_RANALLY_OPERATION_PARAMETER
#define INCLUDED_RANALLY_OPERATION_PARAMETER

#include <vector>
#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include <boost/range/algorithm/copy.hpp>
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"



namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Parameter:
  private boost::noncopyable
{

  friend class ParameterTest;

public:

  template<class Range>
                   Parameter           (UnicodeString const& name,
                                        UnicodeString const& description,
                                        Range const& dataTypes,
                                        Range const& valueTypes);

                   ~Parameter          ();

private:

  UnicodeString    _name;

  UnicodeString    _description;

  std::vector<DataType> _dataTypes;

  std::vector<ValueType> _valueTypes;

};



template<
  class Range>
inline Parameter::Parameter(
  UnicodeString const& name,
  UnicodeString const& description,
  Range const& dataTypes,
  Range const& valueTypes)

  : _name(name),
    _description(description)

{
  boost::range::copy(dataTypes, _dataTypes);
  boost::range::copy(valueTypes, _valueTypes);

  assert(!_name.isEmpty());
  assert(!_description.isEmpty());
  assert(!_dataTypes.empty());
  assert(!_valueTypes.empty());
}

} // namespace operation
} // namespace ranally

#endif
