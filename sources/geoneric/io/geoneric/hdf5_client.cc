#include "geoneric/io/geoneric/hdf5_client.h"
#include <cpp/H5Cpp.h>
#include "geoneric/io/drivers.h"


namespace geoneric {

size_t HDF5Client::_count = 0u;


HDF5Client::HDF5Client()
{
    ++_count;

    if(_count == 1u) {
        H5::Exception::dontPrint();
    }
}


HDF5Client::~HDF5Client()
{
    assert(_count > 0u);
    --_count;
}

} // namespace geoneric
