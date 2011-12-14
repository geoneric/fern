#ifndef INCLUDED_RANALLY_OPERATION_PARAMETER
#define INCLUDED_RANALLY_OPERATION_PARAMETER

#include <vector>
#include <unicode/unistr.h>
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
class Parameter
{

  friend class ParameterTest;

public:

  template<
    class DataTypes,
    class ValueTypes>
                   Parameter           (UnicodeString const& name,
                                        UnicodeString const& description,
                                        DataTypes const& dataTypes,
                                        ValueTypes const& valueTypes);

                   Parameter           (Parameter const& other);

  Parameter&       operator=           (Parameter const& other);

                   ~Parameter          ();

  UnicodeString const& name            () const;

  UnicodeString const& description     () const;

  std::vector<DataType> const& dataTypes() const;

  std::vector<ValueType> const& valueTypes() const;

private:

  UnicodeString    _name;

  UnicodeString    _description;

  std::vector<DataType> _dataTypes;

  std::vector<ValueType> _valueTypes;

};



template<
  class DataTypes,
  class ValueTypes>
inline Parameter::Parameter(
  UnicodeString const& name,
  UnicodeString const& description,
  DataTypes const& dataTypes,
  ValueTypes const& valueTypes)

  : _name(name),
    _description(description)

{
  _dataTypes.resize(boost::size(dataTypes));
  boost::range::copy(dataTypes, boost::begin(_dataTypes));
  _valueTypes.resize(boost::size(valueTypes));
  boost::range::copy(valueTypes, boost::begin(_valueTypes));

  assert(!_name.isEmpty());
  assert(!_description.isEmpty());
  assert(!_dataTypes.empty());
  assert(!_valueTypes.empty());
}

} // namespace operation
} // namespace ranally

#endif
