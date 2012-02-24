#include "Ranally/IO/HDF5Client.h"
#include <H5Cpp.h>



namespace ranally {
namespace io {

HDF5Client::HDF5Client()
{
  H5::Exception::dontPrint();
}



HDF5Client::~HDF5Client()
{
}

} // namespace io
} // namespace ranally

