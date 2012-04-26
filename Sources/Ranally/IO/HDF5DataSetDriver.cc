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
    unsigned int accessMode = H5F_ACC_TRUNC; // | H5F_ACC_RDWR?
    H5::FileCreatPropList creationProperties = H5::FileCreatPropList::DEFAULT;
    H5::FileAccPropList accessProperties = H5::FileAccPropList::DEFAULT;
    H5::H5File* file = new H5::H5File(
      ranally::util::encodeInUTF8(name).c_str(), accessMode,
      creationProperties, accessProperties);
    file->flush(H5F_SCOPE_GLOBAL);
    result = new HDF5DataSet(name, file);
  }
  catch(H5::FileIException const& exception) {
    // TODO Raise exception.
    exception.printError(stderr);
    throw std::string("cannot create hdf5 dataset");
  }

  assert(exists(name));
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
      throw std::string("cannot remove hdf5 dataset");
    }
  }
}



HDF5DataSet* HDF5DataSetDriver::open(
  UnicodeString const& name) const
{
  HDF5DataSet* result = 0;

  try {
    H5::H5File* file = new H5::H5File(
      ranally::util::encodeInUTF8(name).c_str(), H5F_ACC_RDONLY);

    result = new HDF5DataSet(name, file);
  }
  catch(H5::FileIException const&) {
    // TODO Raise exception.
    throw std::string("cannot open hdf5 file");
  }

  assert(result);
  return result;
}

} // namespace io
} // namespace ranally

