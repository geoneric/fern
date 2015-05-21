// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_client.h"
#include <hdf5.h>
#include "fern/language/io/drivers.h"


namespace fern {
namespace language {

size_t HDF5Client::_count = 0u;


HDF5Client::HDF5Client()
{
    ++_count;

    if(_count == 1u) {
        H5open();
        H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

        // H5::Exception::dontPrint();
        // H5::H5Library::open();
    }
}


HDF5Client::~HDF5Client()
{
    assert(_count > 0u);
    --_count;

    if(_count == 0) {
        // H5Eset_auto(H5E_DEFAULT, H5Eprint, nullptr);
        H5close();

        // TODO For some reason closing the library and opening it again
        //      is not supported... So we just not close it for now. *&@#($*&!
        // H5::H5Library::close();
    }
}

} // namespace language
} // namespace fern
