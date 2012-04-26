#include "Ranally/IO/DataSet.h"



namespace ranally {
namespace io {

DataSet::DataSet(
  UnicodeString const& name)

  : _name(name)

{
}



DataSet::~DataSet()
{
}



UnicodeString const& DataSet::name() const
{
  return _name;
}

} // namespace io
} // namespace ranally

