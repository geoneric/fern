#include "Ranally/IO/ogr_client.h"
#include "cpl_error.h"
#include "ogr_api.h"


namespace ranally {

OGRClient::OGRClient()
{
    OGRRegisterAll();
    CPLSetErrorHandler(CPLQuietErrorHandler);
}


OGRClient::~OGRClient()
{
    OGRCleanupAll();
}

} // namespace ranally
