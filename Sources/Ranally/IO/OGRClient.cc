#include "Ranally/IO/OGRClient.h"
#include "ogr_api.h"



namespace ranally {
namespace io {

OGRClient::OGRClient()
{
  OGRRegisterAll();
}



OGRClient::~OGRClient()
{
  OGRCleanupAll();
}

} // namespace io
} // namespace ranally

