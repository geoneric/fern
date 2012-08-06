#include "Ranally/IO/DataSet.h"



namespace ranally {
namespace io {

DataSet::DataSet(
  String const& name)

  : _name(name)

{
}



DataSet::~DataSet()
{
}



String const& DataSet::name() const
{
  return _name;
}

} // namespace io
} // namespace ranally

