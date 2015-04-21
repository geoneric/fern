// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <complex>
#include <string>
#include <gdal_priv.h>


namespace fern {
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      GDAL value type traits.

    GDAL data types are what we call value types in Fern. So, although the
    template parameter is of type GDALDataType, we call it a value type.

    - std::string name: @a value_type as string.
    - type: C++ value type corresponding to GDAL @a value_type, but only in
          case there is an exact match.
*/
template<
    GDALDataType value_type>
struct GDALValueTypeTraits
{
};


template<>
struct GDALValueTypeTraits<GDT_Byte>
{
    static std::string const name;
    using type = uint8_t;
};


template<>
struct GDALValueTypeTraits<GDT_UInt16>
{
    static std::string const name;
    using type = uint16_t;
};


template<>
struct GDALValueTypeTraits<GDT_Int16>
{
    static std::string const name;
    using type = int16_t;
};


template<>
struct GDALValueTypeTraits<GDT_UInt32>
{
    static std::string const name;
    using type = uint32_t;
};


template<>
struct GDALValueTypeTraits<GDT_Int32>
{
    static std::string const name;
    using type = int32_t;
};


template<>
struct GDALValueTypeTraits<GDT_Float32>
{
    static std::string const name;
    using type = float;
};


template<>
struct GDALValueTypeTraits<GDT_Float64>
{
    static std::string const name;
    using type = double;
};


template<>
struct GDALValueTypeTraits<GDT_CInt16>
{
    static std::string const name;
};


template<>
struct GDALValueTypeTraits<GDT_CInt32>
{
    static std::string const name;
};


template<>
struct GDALValueTypeTraits<GDT_CFloat32>
{
    static std::string const name;
    using type = std::complex<float>;
};


template<>
struct GDALValueTypeTraits<GDT_CFloat64>
{
    static std::string const name;
    using type = std::complex<double>;
};


template<>
struct GDALValueTypeTraits<GDT_TypeCount>
{
    static std::string const name;
};


template<>
struct GDALValueTypeTraits<GDT_Unknown>
{
    static std::string const name;
};

} // namespace gdal
} // namespace io
} // namespace fern
