// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gdal/gdal_data_type_traits.h"


namespace fern {

String const GDALDataTypeTraits<GDT_Unknown>::name = "GDT_Unknown";
String const GDALDataTypeTraits<GDT_CInt16>::name = "GDT_CInt16";
String const GDALDataTypeTraits<GDT_CInt32>::name = "GDT_CInt32";
String const GDALDataTypeTraits<GDT_CFloat32>::name = "GDT_CFloat32";
String const GDALDataTypeTraits<GDT_CFloat64>::name = "GDT_CFloat64";
String const GDALDataTypeTraits<GDT_TypeCount>::name = "GDT_TypeCount";

} // namespace fern
