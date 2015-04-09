// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
