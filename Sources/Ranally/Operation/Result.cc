#include "Ranally/Operation/Result.h"
#include <cassert>



namespace ranally {
namespace operation {

Result::Result(
  UnicodeString const& name,
  UnicodeString const& description,
  DataType const& dataType,
  ValueType const& valueType)

  : _name(name),
    _description(description),
    _dataType(dataType),
    _valueType(valueType)

{
  assert(!_name.isEmpty());
  assert(!_description.isEmpty());
}



Result::~Result()
{
}

} // namespace operation
} // namespace ranally

