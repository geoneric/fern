// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
    using type = uint8_t;
};


template<>
struct GDALDataTypeTraits<GDT_UInt16>
{
    static String const name;
    using type = uint16_t;
};


template<>
struct GDALDataTypeTraits<GDT_Int16>
{
    static String const name;
    using type = int16_t;
};


template<>
struct GDALDataTypeTraits<GDT_UInt32>
{
    static String const name;
    using type = uint32_t;
};


template<>
struct GDALDataTypeTraits<GDT_Int32>
{
    static String const name;
    using type = int32_t;
};


template<>
struct GDALDataTypeTraits<GDT_Float32>
{
    static String const name;
    using type = float;
};


template<>
struct GDALDataTypeTraits<GDT_Float64>
{
    static String const name;
    using type = double;
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
