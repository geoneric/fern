#include "Ranally/IO/HDF5DataSet.h"
#include <H5Cpp.h>



namespace ranally {
namespace io {

HDF5DataSet::HDF5DataSet(
  UnicodeString const& name,
  H5::H5File* file)

  : DataSet(name),
    _file(file)

{
  assert(_file);
}



HDF5DataSet::~HDF5DataSet()
{
}

} // namespace io
} // namespace ranally

