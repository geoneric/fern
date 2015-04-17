// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/core/file.h"
#include <boost/filesystem.hpp>


namespace fern {
namespace io {

/*!
    @ingroup    fern_io_core_group
    @brief      Return whether @a pathname exists and is accessible.
*/
bool file_exists(
    std::string const& pathname)
{
    return boost::filesystem::exists(pathname);
}

} // namespace io
} // namespace fern
