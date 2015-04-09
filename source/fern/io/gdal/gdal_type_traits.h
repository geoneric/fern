// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {

template<
    class T>
struct GDALTypeTraits
{
};


template<>
struct GDALTypeTraits<uint8_t>
{
    static GDALDataType const data_type = GDT_Byte;
};


template<>
struct GDALTypeTraits<int8_t>
{
    // int8 is not supported by GDAL...
    static GDALDataType const data_type = GDT_Int16;
};


template<>
struct GDALTypeTraits<uint16_t>
{
    static GDALDataType const data_type = GDT_UInt16;
};


template<>
struct GDALTypeTraits<int16_t>
{
    static GDALDataType const data_type = GDT_Int16;
};


template<>
struct GDALTypeTraits<uint32_t>
{
    static GDALDataType const data_type = GDT_UInt32;
};


template<>
struct GDALTypeTraits<int32_t>
{
    static GDALDataType const data_type = GDT_Int32;
};


template<>
struct GDALTypeTraits<uint64_t>
{
    // uint64 is not supported by GDAL...
    static GDALDataType const data_type = GDT_Float64;
};


template<>
struct GDALTypeTraits<int64_t>
{
    // int64 is not supported by GDAL...
    static GDALDataType const data_type = GDT_Float64;
};


template<>
struct GDALTypeTraits<float>
{
    static GDALDataType const data_type = GDT_Float32;
};


template<>
struct GDALTypeTraits<double>
{
    static GDALDataType const data_type = GDT_Float64;
};

} // namespace fern
