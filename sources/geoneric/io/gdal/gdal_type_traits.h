#pragma once


namespace geoneric {

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
struct GDALTypeTraits<float>
{
    static GDALDataType const data_type = GDT_Float32;
};


template<>
struct GDALTypeTraits<double>
{
    static GDALDataType const data_type = GDT_Float64;
};

} // namespace geoneric
