// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/fern_dataset.h"
#include <iostream>
#include "fern/core/data_name.h"
#include "fern/core/io_error.h"
#include "fern/core/type_traits.h"
#include "fern/core/value_type_traits.h"
#include "fern/language/feature/visitor/attribute_type_visitor.h"
#include "fern/language/io/fern/hdf5_type_class_traits.h"
#include "fern/language/io/fern/hdf5_type_traits.h"
#include "fern/language/io/fern/utils.h"


namespace fern {
namespace language {
namespace {

} // Anonymous namespace


FernDataset::FernDataset(
    std::string const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file()

{
    DataName data_name(name);
    _file = open_file(data_name.database_pathname(), open_mode);
}


FernDataset::FernDataset(
    std::unique_ptr<HDF5File>& file,
    std::string const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file(std::move(file))

{
}


FernDataset::~FernDataset()
{
}


HDF5Dataset FernDataset::dataset(
    Path const& path) const
{
    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, path));
    }

    return HDF5Dataset{_file->open_dataset(path.generic_string())};
}


// TODO exception
#define UNSUPPORTED_TYPE_CLASS_CASE(                                           \
        type_class)                                                            \
    case type_class: {                                                         \
        std::cout << HDF5TypeClassTraits<type_class>::name << std::endl;       \
        assert(false);                                                         \
        break;                                                                 \
    }

ValueType FernDataset::value_type(
    HDF5Dataset const& dataset) const
{
    ValueType result;
    H5T_class_t const type_class = dataset.type_class();
    hid_t const type = dataset.type();

    switch(type_class) {
        case H5T_INTEGER: {
            H5T_sign_t const sign = H5Tget_sign(type);
            size_t const int_size = H5Tget_size(type);
            assert(sign == H5T_SGN_NONE || sign == H5T_SGN_2);

            if(sign == H5T_SGN_NONE) {
                // Unsigned.
                if(int_size == 1u) {
                    result = VT_UINT8;
                }
                else if(int_size == 2u) {
                    result = VT_UINT16;
                }
                else if(int_size == 4u) {
                    result = VT_UINT32;
                }
                else if(int_size == 8u) {
                    result = VT_UINT64;
                }
                else {
                    assert(false);
                }
            }
            else {
                // Signed.
                if(int_size == 1u) {
                    result = VT_INT8;
                }
                else if(int_size == 2u) {
                    result = VT_INT16;
                }
                else if(int_size == 4u) {
                    result = VT_INT32;
                }
                else if(int_size == 8u) {
                    result = VT_INT64;
                }
                else {
                    // TODO Exception.
                    assert(false);
                }
            }

            break;
        }
        case H5T_FLOAT: {
            size_t const float_size = H5Tget_size(type);

            if(float_size == 4u) {
                result = VT_FLOAT32;
            }
            else if(float_size == 8u) {
                result = VT_FLOAT64;
            }
            else {
                // TODO Exception.
                assert(false);
            }

            break;
        }
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_TIME)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_STRING)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_NO_CLASS)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_BITFIELD)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_OPAQUE)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_COMPOUND)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_REFERENCE)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_ENUM)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_VLEN)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_ARRAY)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_NCLASSES)
    }

    return result;
}

#undef UNSUPPORTED_TYPE_CLASS_CASE


size_t FernDataset::nr_features() const
{
    return nr_features("/");
}


size_t FernDataset::nr_features(
        Path const& path) const
{
    assert(contains_feature(path));
    HDF5Group group(_file->open_group(path.generic_string()));

    return group.nr_groups();
}


std::vector<std::string> FernDataset::feature_names() const
{
    Path path("/");
    assert(contains_feature(path));
    HDF5Group group(_file->open_group(path.generic_string()));

    return group.group_names();
}


size_t FernDataset::nr_attributes(
        Path const& path) const
{
    assert(contains_feature(path));
    HDF5Group group(_file->open_group(path.generic_string()));

    return group.nr_datasets();
}


bool FernDataset::contains_feature(
    Path const& path) const
{
    return contains_feature(path.names());
}


bool FernDataset::contains_feature(
    std::vector<std::string> const& names) const
{
    std::string pathname;
    bool result = true;

    for(auto const& name: names) {
        pathname += std::string("/") + name;

        if(!contains_feature_by_name(pathname)) {
            result = false;
        }
    }

    return result;
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   It is assumed that the parent feature of \a pathname exists.
  \sa        .
*/
bool FernDataset::contains_feature_by_name(
    std::string const& pathname) const
{
    return _file->is_group(pathname);
}


bool FernDataset::contains_attribute(
    Path const& path) const
{
    return path.parent_path().generic_string().empty()
        ? contains_attribute_by_name(path.generic_string())
        : contains_feature(path.parent_path()) &&
              contains_attribute_by_name(path.generic_string());
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   It is assumed that the parent feature of \a pathname exists.
  \sa        .
*/
bool FernDataset::contains_attribute_by_name(
    std::string const& pathname) const
{
    return _file->is_dataset(pathname);
}


template<
    class T>
ExpressionType FernDataset::expression_type_numeric_attribute(
    HDF5Dataset const& dataset) const
{
    ExpressionType result;

    switch(dataset.extent_type()) {
        case H5S_SCALAR: {
            result = ExpressionType(DataTypes::CONSTANT,
                TypeTraits<T>::value_types);
            break;
        }
        case H5S_SIMPLE: {
            // TODO Implement.
            assert(false);
            break;
        }
        case H5S_NO_CLASS: {
            // TODO Exception.
            assert(false);
            break;
        }
        case H5S_NULL: {
            // TODO Exception.
            assert(false);
            break;
        }
    }

    return result;
}


// TODO exception
#define UNSUPPORTED_TYPE_CLASS_CASE(                                           \
        type_class)                                                            \
    case type_class: {                                                         \
        std::cout << HDF5TypeClassTraits<type_class>::name << std::endl;       \
        assert(false);                                                         \
        break;                                                                 \
    }

#define NUMBER_CASE(                                                           \
        type)                                                                  \
    result = expression_type_numeric_attribute<type>(dataset);

ExpressionType FernDataset::expression_type(
    Path const& path) const
{
    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, path));
    }

    ExpressionType result;
    HDF5Dataset const dataset{_file->open_dataset(path.generic_string())};
    H5T_class_t const type_class = dataset.type_class();
    hid_t const type = dataset.type();

    switch(type_class) {
        case H5T_INTEGER: {
            H5T_sign_t const sign = H5Tget_sign(type);
            size_t const int_size = H5Tget_size(type);
            assert(sign == H5T_SGN_NONE || sign == H5T_SGN_2);

            if(sign == H5T_SGN_NONE) {
                // Unsigned.
                if(int_size == 1u) {
                    NUMBER_CASE(uint8_t);
                }
                else if(int_size == 2u) {
                    NUMBER_CASE(uint16_t);
                }
                else if(int_size == 4u) {
                    NUMBER_CASE(uint32_t);
                }
                else if(int_size == 8u) {
                    NUMBER_CASE(uint64_t);
                }
                else {
                    assert(false);
                }
            }
            else {
                // Signed.
                if(int_size == 1u) {
                    NUMBER_CASE(int8_t);
                }
                else if(int_size == 2u) {
                    NUMBER_CASE(int16_t);
                }
                else if(int_size == 4u) {
                    NUMBER_CASE(int32_t);
                }
                else if(int_size == 8u) {
                    NUMBER_CASE(int64_t);
                }
                else {
                    // TODO Exception.
                    assert(false);
                }
            }

            break;
        }
        case H5T_FLOAT: {
            size_t const float_size = H5Tget_size(type);

            if(float_size == 4u) {
                NUMBER_CASE(float);
            }
            else if(float_size == 8u) {
                NUMBER_CASE(double);
            }
            else {
                // TODO Exception.
                assert(false);
            }

            break;
        }
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_TIME)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_STRING)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_NO_CLASS)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_BITFIELD)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_OPAQUE)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_COMPOUND)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_REFERENCE)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_ENUM)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_VLEN)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_ARRAY)
        UNSUPPORTED_TYPE_CLASS_CASE(H5T_NCLASSES)
    }

    return result;
}

#undef UNSUPPORTED_TYPE_CLASS_CASE


std::shared_ptr<Feature> FernDataset::open_feature(
    Path const& /* path */) const
{
    assert(false);
    std::shared_ptr<Feature> result;
    return result;
}


template<
    class T>
std::shared_ptr<Attribute> FernDataset::open_attribute(
    HDF5Dataset const& dataset) const
{
    std::shared_ptr<Attribute> result;

    switch(dataset.extent_type()) {
        case H5S_SCALAR: {
            // This attribute contains a single constant value. We might as
            // well read it.
            result = read_constant_attribute<T>(dataset);
            break;
        }
        case H5S_SIMPLE: {
            // TODO Implement.
            assert(false);
            break;
        }
        case H5S_NO_CLASS: {
            // TODO Exception.
            assert(false);
            break;
        }
        case H5S_NULL: {
            // TODO Exception.
            assert(false);
            break;
        }
    }

    return result;
}


#define OPEN_CASE(                                                             \
        value_type)                                                            \
    case value_type: {                                                         \
        result = open_attribute<ValueTypeTraits<value_type>::type>(dataset);   \
        break;                                                                 \
    }

std::shared_ptr<Attribute> FernDataset::open_attribute(
    Path const& path) const
{
    // Open dataset.
    // Determine value type.
    // Open data.

    auto const dataset = this->dataset(path);
    ValueType value_type = this->value_type(dataset);

    std::shared_ptr<Attribute> result;
    switch(value_type) {
        OPEN_CASE(VT_UINT8);
        OPEN_CASE(VT_UINT16);
        OPEN_CASE(VT_UINT32);
        OPEN_CASE(VT_UINT64);
        OPEN_CASE(VT_INT8);
        OPEN_CASE(VT_INT16);
        OPEN_CASE(VT_INT32);
        OPEN_CASE(VT_INT64);
        OPEN_CASE(VT_FLOAT32);
        OPEN_CASE(VT_FLOAT64);
        case VT_BOOL:
        case VT_CHAR:
        case VT_STRING: {
            // TODO
            assert(false);
            break;
        }
    }
    assert(result);
    return result;
}

#undef OPEN_CASE


std::shared_ptr<Feature> FernDataset::read_feature(
    Path const& path) const
{
    if(!contains_feature_by_name(path.generic_string())) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_FEATURE, path));
    }

    std::shared_ptr<Feature> result;

    assert(false);

    return result;
}


template<
    class T>
std::shared_ptr<Attribute> FernDataset::read_constant_attribute(
    HDF5Dataset const& dataset) const
{
    T value;
    dataset.read(value);
    return std::shared_ptr<Attribute>(std::make_shared<ConstantAttribute<T>>(
        value));
}


template<
    class T>
std::shared_ptr<Attribute> FernDataset::read_attribute(
    HDF5Dataset const& dataset) const
{
    std::shared_ptr<Attribute> result;

    switch(dataset.extent_type()) {
        case H5S_SCALAR: {
            result = read_constant_attribute<T>(dataset);
            break;
        }
        case H5S_SIMPLE: {
            // TODO Implement.
            assert(false);
            break;
        }
        case H5S_NO_CLASS: {
            // TODO Exception.
            assert(false);
            break;
        }
        case H5S_NULL: {
            // TODO Exception.
            assert(false);
            break;
        }
    }

    return result;
}


#define READ_CASE(                                                             \
        value_type)                                                            \
    case value_type: {                                                         \
        result = read_attribute<ValueTypeTraits<value_type>::type>(dataset);   \
        break;                                                                 \
    }

std::shared_ptr<Attribute> FernDataset::read_attribute(
    Path const& path) const
{
    // Open dataset.
    // Determine value type.
    // Read data.

    auto const dataset = this->dataset(path);
    ValueType value_type = this->value_type(dataset);

    std::shared_ptr<Attribute> result;
    switch(value_type) {
        READ_CASE(VT_UINT8);
        READ_CASE(VT_UINT16);
        READ_CASE(VT_UINT32);
        READ_CASE(VT_UINT64);
        READ_CASE(VT_INT8);
        READ_CASE(VT_INT16);
        READ_CASE(VT_INT32);
        READ_CASE(VT_INT64);
        READ_CASE(VT_FLOAT32);
        READ_CASE(VT_FLOAT64);
        case VT_BOOL:
        case VT_CHAR:
        case VT_STRING: {
            // TODO
            assert(false);
            break;
        }
    }
    assert(result);
    return result;
}

#undef READ_CASE


void FernDataset::add_feature(
    Path const& path)
{
    assert(!contains_feature(path));
    assert(!contains_attribute(path));
    add_feature(path.names());
}


void FernDataset::add_feature(
    std::vector<std::string> const& names)
{
    // TODO Create groups, when necessary. At least the last group does not
    //      exist.
    assert(!names.empty());
    std::string feature_pathname;

    for(auto const& name: names) {
        feature_pathname += std::string("/") + name;

        if(!contains_feature_by_name(feature_pathname)) {
            _file->create_group(feature_pathname);
        }
    }
}


HDF5Group FernDataset::group(
        Path const& path) const
{
    return HDF5Group{_file->open_group(path.generic_string())};
}


template<
    class T>
void FernDataset::write_attribute(
    ConstantAttribute<T> const& constant,
    Path const& path)
{
    assert(open_mode() == OpenMode::OVERWRITE ||
        open_mode() == OpenMode::UPDATE);
    assert(_file);

    Path feature_path(path.parent_path());
    assert(!feature_path.generic_string().empty());

    if(!contains_feature(feature_path)) {
        add_feature(feature_path);
    }

    HDF5Group group(this->group(feature_path));
    HDF5Dataset dataset;
    std::string attribute_name(path.filename().generic_string());

    if(!contains_attribute_by_name(path.generic_string())) {
        // Create new dataset for a scalar value. Destination value has native
        // type. When reading on a different platform, this may require a
        // conversion to another endianess.
        dataset = HDF5Dataset{group.create_dataset<T>(attribute_name)};
    }
    else {
        // Open the existing dataset and overwrite the contents.
        dataset = HDF5Dataset{group.open_dataset(attribute_name)};
#ifndef NDEBUG
        assert(dataset.space_is_simple());
        assert(dataset.extent_type() == H5S_SCALAR);

        H5T_class_t const type_class = dataset.type_class();
        assert(type_class == HDF5TypeTraits<T>::type_class);
        // assert(dataset.type() == HDF5TypeTraits<T>::data_type);

        // switch(type_class) {
        //     case H5T_INTEGER: {
        //         assert(dataset.getIntType() == HDF5TypeTraits<T>::data_type);
        //         break;
        //     }
        //     case H5T_FLOAT: {
        //         assert(dataset.getFloatType() == HDF5TypeTraits<T>::data_type);
        //         break;
        //     }
        //     default: {
        //         assert(false);
        //         break;
        //     }
        // }
#endif
    }

    // Write scalar value to dataset. Source value has native type.
    ConstantValue<T> const& constant_value(constant.values());
    dataset.write(constant_value.value());
    _file->flush();
}


#define WRITE_ATTRIBUTE_CASE(                                                  \
        value_type)                                                            \
case value_type: {                                                             \
    write_attribute(                                                           \
        dynamic_cast<ConstantAttribute<                                        \
            ValueTypeTraits<value_type>::type> const&>(attribute), path);      \
    break;                                                                     \
}

void FernDataset::write_attribute(
    Attribute const& attribute,
    Path const& path)
{
    // TODO we have a _file. How to write a constant attribute? Just a single
    //      value, global to the feature.

    AttributeTypeVisitor visitor;
    attribute.Accept(visitor);

    switch(visitor.data_type()) {
        case DT_CONSTANT: {
            switch(visitor.value_type()) {
                WRITE_ATTRIBUTE_CASE(VT_UINT8)
                WRITE_ATTRIBUTE_CASE(VT_UINT16)
                WRITE_ATTRIBUTE_CASE(VT_UINT32)
                WRITE_ATTRIBUTE_CASE(VT_UINT64)
                WRITE_ATTRIBUTE_CASE(VT_INT8)
                WRITE_ATTRIBUTE_CASE(VT_INT16)
                WRITE_ATTRIBUTE_CASE(VT_INT32)
                WRITE_ATTRIBUTE_CASE(VT_INT64)
                WRITE_ATTRIBUTE_CASE(VT_FLOAT32)
                WRITE_ATTRIBUTE_CASE(VT_FLOAT64)
                case VT_BOOL:
                case VT_CHAR:
                case VT_STRING: {
                    assert(false); // Value type not supported by driver.
                    break;
                }
            }
            break;
        }
        case DT_STATIC_FIELD: {
            assert(false); // Data type not supported by driver.
            break;
        }
    }
}

#undef WRITE_ATTRIBUTE_CASE

} // namespace language
} // namespace fern
