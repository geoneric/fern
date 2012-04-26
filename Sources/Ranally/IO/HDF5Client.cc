#include "Ranally/IO/HDF5Client.h"
#include <H5Cpp.h>



namespace ranally {
namespace io {

HDF5Client::HDF5Client()
{
  // H5open();
  H5::Exception::dontPrint();
}



HDF5Client::~HDF5Client()
{
  // This prints a lot of error messages on std stream.
  // H5close();
}

} // namespace io
} // namespace ranally

