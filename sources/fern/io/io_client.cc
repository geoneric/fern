#include "fern/io/io_client.h"


namespace fern {

IOClient::IOClient()

    : HDF5Client(),
      FernClient(),
      GDALClient()

{
}


IOClient::~IOClient()
{
}

} // namespace fern
