#include "Ranally/IO/HDF5DataSet.h"



namespace ranally {
namespace io {

HDF5DataSet::HDF5DataSet(
  UnicodeString const& name)

  : DataSet(name)

{
}



HDF5DataSet::~HDF5DataSet()
{
}

} // namespace io
} // namespace ranally

