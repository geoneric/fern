#include "geoneric/io/io_client.h"


namespace geoneric {

IOClient::IOClient()

    : HDF5Client(),
      GeonericClient(),
      GDALClient()

{
}


IOClient::~IOClient()
{
}

} // namespace geoneric
