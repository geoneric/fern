// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gdal/gdal_value_type_traits.h"


namespace fern {
namespace io {
namespace gdal {

std::string const GDALValueTypeTraits<GDT_Unknown>::name = "GDT_Unknown";
std::string const GDALValueTypeTraits<GDT_CInt16>::name = "GDT_CInt16";
std::string const GDALValueTypeTraits<GDT_CInt32>::name = "GDT_CInt32";
std::string const GDALValueTypeTraits<GDT_CFloat32>::name = "GDT_CFloat32";
std::string const GDALValueTypeTraits<GDT_CFloat64>::name = "GDT_CFloat64";
std::string const GDALValueTypeTraits<GDT_TypeCount>::name = "GDT_TypeCount";

} // namespace gdal
} // namespace io
} // namespace fern
