// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstdlib>


namespace fern {
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      Class for making sure GDAL is initialized before use.

    Upon creation this class loads all GDAL drivers and upon destuction
    it unloads them again, removing them from memory.
*/
class GDALClient
{

public:

                   GDALClient          ();

                   GDALClient          (GDALClient const&)=delete;

                   GDALClient          (GDALClient&&)=delete;

    virtual        ~GDALClient         ();

    GDALClient&    operator=           (GDALClient const&)=delete;

    GDALClient&    operator=           (GDALClient&&)=delete;

private:

    //! Number of times an instance is created.
    static size_t  _count;

    void           register_all_drivers();

    void           deregister_all_drivers();

};

} // namespace gdal
} // namespace io
} // namespace fern
