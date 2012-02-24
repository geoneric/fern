#include "Ranally/IO/HDF5DataSetDriver.h"
#include <cassert>
#include <boost/filesystem.hpp>
#include <H5Cpp.h>
#include "Ranally/Util/String.h"
#include "Ranally/IO/HDF5DataSet.h"



namespace ranally {
namespace io {

HDF5DataSetDriver::HDF5DataSetDriver()

  : DataSetDriver()

{
}



HDF5DataSetDriver::~HDF5DataSetDriver()
{
}



bool HDF5DataSetDriver::exists(
  UnicodeString const& name) const
{
  bool result = false;

  try {
    result = H5::H5File::isHdf5(ranally::util::encodeInUTF8(name).c_str());
  }
  catch(H5::FileIException const&) {
    result = false;
  }

  return result;
}



HDF5DataSet* HDF5DataSetDriver::create(
  UnicodeString const& name) const
{
  HDF5DataSet* result = 0;

  try {
    H5::H5File file(ranally::util::encodeInUTF8(name).c_str(), H5F_ACC_TRUNC);

    result = new HDF5DataSet(name);
  }
  catch(H5::FileIException const&) {
    // TODO Raise exception.
    assert(result == 0);
    result = 0;
  }

  return result;
}



void HDF5DataSetDriver::remove(
  UnicodeString const& name) const
{
  if(exists(name)) {
    try {
      boost::filesystem::remove(ranally::util::encodeInUTF8(name).c_str());
    }
    catch(...) {
      // TODO Raise exception.
      throw;
    }
  }
}



HDF5DataSet* HDF5DataSetDriver::open(
  UnicodeString const& name) const
{
  return new HDF5DataSet(name);
}

} // namespace io
} // namespace ranally

