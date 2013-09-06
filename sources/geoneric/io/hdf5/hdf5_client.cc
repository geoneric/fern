#include "geoneric/io/hdf5/hdf5_client.h"
#include <cpp/H5Cpp.h>


namespace geoneric {

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

} // namespace geoneric
