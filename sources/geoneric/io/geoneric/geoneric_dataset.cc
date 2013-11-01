#include "geoneric/io/geoneric/geoneric_dataset.h"
#include "geoneric/core/data_name.h"
#include "geoneric/core/io_error.h"
#include "geoneric/core/type_traits.h"
#include "geoneric/core/value_type_traits.h"
#include "geoneric/feature/visitor/attribute_type_visitor.h"
#include "geoneric/io/geoneric/hdf5_type_class_traits.h"
#include "geoneric/io/geoneric/hdf5_type_traits.h"
#include "geoneric/io/geoneric/utils.h"


namespace geoneric {

GeonericDataset::GeonericDataset(
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file()

{
    DataName data_name(name);
    _file = open_file(data_name.database_pathname(), open_mode);
}


GeonericDataset::GeonericDataset(
    std::shared_ptr<H5::H5File> const& file,
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _file(file)

{
}


GeonericDataset::~GeonericDataset()
{
}


size_t GeonericDataset::nr_features() const
{
    return nr_features("/");
}


size_t GeonericDataset::nr_features(
        Path const& path) const
{
    assert(contains_feature(path));
    H5::Group group(_file->openGroup(String(path).encode_in_utf8()));

    size_t result = 0;
    H5G_obj_t type;

    for(hsize_t i = 0; i < group.getNumObjs(); ++i) {
        type = group.getObjTypeByIdx(i);
        if(type == H5G_GROUP) {
            ++result;
        }
    }

    return result;
}


size_t GeonericDataset::nr_attributes(
        Path const& path) const
{
    assert(contains_feature(path));
    H5::Group group(_file->openGroup(String(path).encode_in_utf8()));

    size_t result = 0;
    H5G_obj_t type;

    for(hsize_t i = 0; i < group.getNumObjs(); ++i) {
        type = group.getObjTypeByIdx(i);
        if(type == H5G_DATASET) {
            ++result;
        }
    }

    return result;
}


bool GeonericDataset::contains_feature(
    Path const& path) const
{
    return contains_feature(path.names());
}


bool GeonericDataset::contains_feature(
    std::vector<String> const& names) const
{
    String pathname;
    bool result = true;

    for(auto const& name: names) {
        pathname += "/" + name;

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
bool GeonericDataset::contains_feature_by_name(
    String const& pathname) const
{
    bool result = false;

    if(H5Lexists(_file->getLocId(), pathname.encode_in_utf8().c_str(),
            H5P_DEFAULT) == TRUE) {
        H5G_stat_t status;
        _file->getObjinfo(pathname.encode_in_utf8(), status);
        result = status.type == H5G_GROUP;
    }

    return result;
}


bool GeonericDataset::contains_attribute(
    Path const& path) const
{
    return String(path.parent_path()).is_empty()
        ? contains_attribute_by_name(path)
        : contains_feature(path.parent_path()) &&
              contains_attribute_by_name(path);
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
bool GeonericDataset::contains_attribute_by_name(
    String const& pathname) const
{
    bool result = false;

    if(H5Lexists(_file->getLocId(), pathname.encode_in_utf8().c_str(),
            H5P_DEFAULT) == TRUE) {
        H5G_stat_t status;
        _file->getObjinfo(pathname.encode_in_utf8(), status);
        result = status.type == H5G_DATASET;
    }

    return result;
}


template<
    class T>
ExpressionType GeonericDataset::expression_type_numeric_attribute(
    H5::DataSet const& dataset) const
{
    ExpressionType result;

    H5::DataSpace data_space = dataset.getSpace();
    assert(data_space.isSimple());

    switch(data_space.getSimpleExtentType()) {
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

ExpressionType GeonericDataset::expression_type(
    Path const& path) const
{
    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, path));
    }

    ExpressionType result;
    H5::DataSet const dataset = _file->openDataSet(
        String(path).encode_in_utf8());
    H5T_class_t const type_class = dataset.getTypeClass();

    switch(type_class) {
        case H5T_INTEGER: {
            H5::IntType const int_type = dataset.getIntType();
            assert(int_type.getSign() == H5T_SGN_NONE ||
                int_type.getSign() == H5T_SGN_2);
            size_t const int_size = int_type.getSize();

            if(int_type.getSign() == H5T_SGN_NONE) {
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
            H5::FloatType const float_type = dataset.getFloatType();
            size_t const float_size = float_type.getSize();

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
#undef NUMBER_CASE


std::shared_ptr<Feature> GeonericDataset::read_feature(
    Path const& path) const
{
    if(!contains_feature_by_name(path)) {
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
std::shared_ptr<Attribute> GeonericDataset::read_constant_attribute(
    Path const& /* path */,
    H5::DataSet const& dataset) const
{
    T value;
    dataset.read(&value, HDF5TypeTraits<T>::data_type);
    return std::shared_ptr<Attribute>(new ConstantAttribute<T>(value));
}


template<
    class T>
std::shared_ptr<Attribute> GeonericDataset::read_numeric_attribute(
    Path const& path,
    H5::DataSet const& dataset) const
{
    std::shared_ptr<Attribute> result;

    H5::DataSpace data_space = dataset.getSpace();
    assert(data_space.isSimple());

    switch(data_space.getSimpleExtentType()) {
        case H5S_SCALAR: {
            result = read_constant_attribute<T>(path, dataset);
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

#define NUMBER_CASE( \
        type) \
    result = read_numeric_attribute<type>(path, dataset);

std::shared_ptr<Attribute> GeonericDataset::read_attribute(
    Path const& path) const
{
    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, path));
    }

    std::shared_ptr<Attribute> result;

    // Open group that contains the attribute.
    // Get properties of attribute.
    // Determine what kind of attribute to create and create it.
    // Read attribute and return result.

    // std::shared_ptr<H5::Group> group(this->group(path.parent_path()));

    // H5G_stat_t status;
    // group->getObjinfo(pathname.encode_in_utf8(), status);
    // result = status.type == H5G_GROUP;

    H5::DataSet const dataset = _file->openDataSet(
        String(path).encode_in_utf8());
    H5T_class_t const type_class = dataset.getTypeClass();

    switch(type_class) {
        case H5T_INTEGER: {
            H5::IntType const int_type = dataset.getIntType();
            assert(int_type.getSign() == H5T_SGN_NONE ||
                int_type.getSign() == H5T_SGN_2);
            size_t const int_size = int_type.getSize();

            if(int_type.getSign() == H5T_SGN_NONE) {
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
            H5::FloatType const float_type = dataset.getFloatType();
            size_t const float_size = float_type.getSize();

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
#undef NUMBER_CASE


void GeonericDataset::add_feature(
    Path const& path)
{
    assert(!contains_feature(path));
    assert(!contains_attribute(path));
    add_feature(path.names());
}


void GeonericDataset::add_feature(
    std::vector<String> const& names)
{
    // TODO Create groups, when necessary. At least the last group does not
    //      exist.
    assert(!names.empty());
    String feature_pathname;

    for(auto const& name: names) {
        feature_pathname += "/" + name;

        if(!contains_feature_by_name(feature_pathname)) {
            _file->createGroup(feature_pathname.encode_in_utf8());
        }
    }
}


std::shared_ptr<H5::Group> GeonericDataset::group(
        Path const& path) const
{
    return std::make_shared<H5::Group>(
        _file->openGroup(String(path).encode_in_utf8()));
}


template<
    class T>
void GeonericDataset::write_attribute(
    ConstantAttribute<T> const& constant,
    Path const& path)
{
    assert(open_mode() == OpenMode::OVERWRITE ||
        open_mode() == OpenMode::UPDATE);
    assert(_file);

    Path feature_path(path.parent_path());
    assert(!String(feature_path).is_empty());

    if(!contains_feature(feature_path)) {
        add_feature(feature_path);
    }

    std::shared_ptr<H5::Group> group(this->group(feature_path));
    H5::DataSet dataset;
    String attribute_name(path.filename());

    if(!contains_attribute_by_name(path)) {
        // Create new dataset for a scalar value. Destination value has native
        // type. When reading on a different platform, this may require a
        // conversion to another endianess.
        H5::DataSpace data_space;
        dataset = group->createDataSet(
            attribute_name.encode_in_utf8(), HDF5TypeTraits<T>::data_type,
            data_space);
    }
    else {
        // Open the existing dataset and overwrite the contents.
        dataset = group->openDataSet(attribute_name.encode_in_utf8());
#ifndef NDEBUG
        H5::DataSpace const data_space = dataset.getSpace();
        assert(data_space.isSimple());
        assert(data_space.getSimpleExtentType() == H5S_SCALAR);
        H5T_class_t const type_class = dataset.getTypeClass();
        assert(type_class == HDF5TypeTraits<T>::type_class);

        switch(type_class) {
            case H5T_INTEGER: {
                assert(dataset.getIntType() == HDF5TypeTraits<T>::data_type);
                break;
            }
            case H5T_FLOAT: {
                assert(dataset.getFloatType() == HDF5TypeTraits<T>::data_type);
                break;
            }
            default: {
                assert(false);
                break;
            }
        }
#endif
    }

    // Write scalar value to dataset. Source value has native type.
    ConstantValue<T> const& constant_value(constant.values());
    dataset.write(&constant_value.value(), HDF5TypeTraits<T>::data_type);
}


#define WRITE_ATTRIBUTE_CASE(                                                  \
        value_type)                                                            \
case value_type: {                                                             \
    write_attribute(                                                           \
        dynamic_cast<ConstantAttribute<                                        \
            ValueTypeTraits<value_type>::type> const&>(attribute), path);      \
    break;                                                                     \
}

void GeonericDataset::write_attribute(
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

} // namespace geoneric
