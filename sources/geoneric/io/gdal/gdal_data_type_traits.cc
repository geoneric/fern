#include "geoneric/io/gdal/gdal_data_type_traits.h"


namespace geoneric {

String const GDALDataTypeTraits<GDT_Unknown>::name = "GDT_Unknown";
String const GDALDataTypeTraits<GDT_CInt16>::name = "GDT_CInt16";
String const GDALDataTypeTraits<GDT_CInt32>::name = "GDT_CInt32";
String const GDALDataTypeTraits<GDT_CFloat32>::name = "GDT_CFloat32";
String const GDALDataTypeTraits<GDT_CFloat64>::name = "GDT_CFloat64";
String const GDALDataTypeTraits<GDT_TypeCount>::name = "GDT_TypeCount";

} // namespace geoneric
