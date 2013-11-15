#pragma once
#include <gdal_priv.h>
#include "fern/core/string.h"


namespace fern {

template<
    GDALDataType data_type>
struct GDALDataTypeTraits
{
};


template<>
struct GDALDataTypeTraits<GDT_Byte>
{
    static String const name;
    typedef uint8_t type;
};


template<>
struct GDALDataTypeTraits<GDT_UInt16>
{
    static String const name;
    typedef uint16_t type;
};


template<>
struct GDALDataTypeTraits<GDT_Int16>
{
    static String const name;
    typedef int16_t type;
};


template<>
struct GDALDataTypeTraits<GDT_UInt32>
{
    static String const name;
    typedef uint32_t type;
};


template<>
struct GDALDataTypeTraits<GDT_Int32>
{
    static String const name;
    typedef int32_t type;
};


template<>
struct GDALDataTypeTraits<GDT_Float32>
{
    static String const name;
    typedef float type;
};


template<>
struct GDALDataTypeTraits<GDT_Float64>
{
    static String const name;
    typedef double type;
};


template<>
struct GDALDataTypeTraits<GDT_CInt16>
{
    static String const name;
};


template<>
struct GDALDataTypeTraits<GDT_CInt32>
{
    static String const name;
};


template<>
struct GDALDataTypeTraits<GDT_CFloat32>
{
    static String const name;
};


template<>
struct GDALDataTypeTraits<GDT_CFloat64>
{
    static String const name;
};


template<>
struct GDALDataTypeTraits<GDT_TypeCount>
{
    static String const name;
};


template<>
struct GDALDataTypeTraits<GDT_Unknown>
{
    static String const name;
};

} // namespace fern
