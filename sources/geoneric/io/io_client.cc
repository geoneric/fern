#include "geoneric/io/io_client.h"


namespace geoneric {

IOClient::IOClient()

    : GDALClient(),
      HDF5Client(),
      GeonericClient()

{
}


IOClient::~IOClient()
{
}

} // namespace geoneric
