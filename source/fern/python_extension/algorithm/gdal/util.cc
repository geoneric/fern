#include "fern/python_extension/algorithm/gdal/util.h"
#include <map>
#include "fern/python_extension/core/swig.h"


namespace fern {
namespace python {

String to_string(
    GDALDataType const& data_type)
{
    static std::map<GDALDataType, String> gdal_type_names {
        { GDT_Byte    , "GDT_Byte"     },
        { GDT_UInt16  , "GDT_UInt16"   },
        { GDT_Int16   , "GDT_Int16"    },
        { GDT_UInt32  , "GDT_UInt32"   },
        { GDT_Int32   , "GDT_Int32"    },
        { GDT_Float32 , "GDT_Float32"  },
        { GDT_Float64 , "GDT_Float64"  },
        { GDT_CInt16  , "GDT_CInt16"   },
        { GDT_CInt32  , "GDT_CInt32"   },
        { GDT_CFloat32, "GDT_CFloat32" },
        { GDT_CFloat64, "GDT_CFloat64" }
    };

    assert(gdal_type_names.find(data_type) != gdal_type_names.end());
    return gdal_type_names[data_type];
}


bool is_gdal_raster_band(
    PyObject* object)
{
    return swig_object(object, "_p_GDALRasterBandShadow") != nullptr;
}


GDALRasterBand* gdal_raster_band(
    PyObject* object)
{
    assert(object);
    return static_cast<GDALRasterBand*>(swig_object(object)->ptr);
}

} // namespace python
} // namespace fern
