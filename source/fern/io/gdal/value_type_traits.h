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
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      C++ value type traits.

    - GDALDataType gdal_value_type: GDAL Datatype corresponding to C++ value
      type @a T, but only in case there is an exact match.
*/
template<
    class T>
struct ValueTypeTraits
{
};


template<>
struct ValueTypeTraits<uint8_t>
{
    static GDALDataType const gdal_value_type = GDT_Byte;
};


template<>
struct ValueTypeTraits<int8_t>
{
    // int8 is not supported by GDAL...
    // static GDALDataType const gdal_value_type = GDT_Int16;
};


template<>
struct ValueTypeTraits<uint16_t>
{
    static GDALDataType const gdal_value_type = GDT_UInt16;
};


template<>
struct ValueTypeTraits<int16_t>
{
    static GDALDataType const gdal_value_type = GDT_Int16;
};


template<>
struct ValueTypeTraits<uint32_t>
{
    static GDALDataType const gdal_value_type = GDT_UInt32;
};


template<>
struct ValueTypeTraits<int32_t>
{
    static GDALDataType const gdal_value_type = GDT_Int32;
};


template<>
struct ValueTypeTraits<uint64_t>
{
    // uint64 is not supported by GDAL...
    // static GDALDataType const gdal_value_type = GDT_Float64;
};


template<>
struct ValueTypeTraits<int64_t>
{
    // int64 is not supported by GDAL...
    // static GDALDataType const gdal_value_type = GDT_Float64;
};


template<>
struct ValueTypeTraits<float>
{
    static GDALDataType const gdal_value_type = GDT_Float32;
};


template<>
struct ValueTypeTraits<double>
{
    static GDALDataType const gdal_value_type = GDT_Float64;
};

} // namespace gdal
} // namespace io
} // namespace fern
