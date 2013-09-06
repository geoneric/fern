#include "geoneric/io/ogr/ogr_client.h"
#include "cpl_error.h"
#include "ogr_api.h"


namespace geoneric {

OGRClient::OGRClient()
{
    OGRRegisterAll();
    CPLSetErrorHandler(CPLQuietErrorHandler);
}


OGRClient::~OGRClient()
{
    OGRCleanupAll();
}

} // namespace geoneric
