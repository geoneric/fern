#include "fern/io/ogr/ogr_client.h"
#include "cpl_error.h"
#include "ogr_api.h"


namespace fern {

OGRClient::OGRClient()
{
    OGRRegisterAll();
    CPLSetErrorHandler(CPLQuietErrorHandler);
}


OGRClient::~OGRClient()
{
    OGRCleanupAll();
}

} // namespace fern
