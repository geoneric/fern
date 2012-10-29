#include "Ranally/IO/OGRClient.h"
#include "cpl_error.h"
#include "ogr_api.h"


namespace ranally {
namespace io {

OGRClient::OGRClient()
{
    OGRRegisterAll();
    CPLSetErrorHandler(CPLQuietErrorHandler);
}


OGRClient::~OGRClient()
{
    OGRCleanupAll();
}

} // namespace io
} // namespace ranally
