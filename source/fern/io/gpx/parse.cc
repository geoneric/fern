// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gpx/parse.h"
#include <fern/io/xml/parse.h>


namespace fern {
namespace io {
namespace gpx_1_0 {

/*!
    @ingroup    fern_io_gpx_group
    @brief      Parse GPX document pointed to by @a pathname and return
                the contents in a tree-like datastructure.
    @sa         xml::parse
*/
std::unique_ptr<::gpx_1_0::gpx> parse(
    std::string const& pathname)
{
    auto parse = [](std::istream& stream) { return ::gpx_1_0::gpx_(stream); };
    return xml::parse(parse, pathname);
}

} // namespace gpx_1_0


namespace gpx_1_1 {

/*!
    @ingroup    fern_io_gpx_group
    @brief      Parse GPX document pointed to by @a pathname and return
                the contents in a tree-like datastructure.
    @sa         xml::parse
*/
std::unique_ptr<::gpx_1_1::gpxType> parse(
    std::string const& pathname)
{
    auto parse = [](std::istream& stream) { return ::gpx_1_1::gpx(stream); };
    return xml::parse(parse, pathname);
}

} // namespace gpx_1_1
} // namespace io
} // namespace fern
