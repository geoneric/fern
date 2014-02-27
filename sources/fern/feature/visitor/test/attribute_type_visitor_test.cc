#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/feature/core/constant_attribute.h"
#include "fern/feature/visitor/attribute_type_visitor.h"


#define CALL_FUNCTION_TEMPLATE_1(                                              \
        function_template,                                                     \
        argument)                                                              \
    function_template<bool>(argument);                                         \
    function_template<int8_t>(argument);                                       \
    function_template<int16_t>(argument);                                      \
    function_template<int32_t>(argument);                                      \
    function_template<int64_t>(argument);                                      \
    function_template<uint8_t>(argument);                                      \
    function_template<uint16_t>(argument);                                     \
    function_template<uint32_t>(argument);                                     \
    function_template<uint64_t>(argument);                                     \
    function_template<float>(argument);                                        \
    function_template<double>(argument);                                       \
    function_template<fern::String>(argument);


template<
    class T>
void test_constant_attribute(
    fern::AttributeTypeVisitor& visitor)
{
    fern::ConstantAttribute<T> attribute;
    attribute.Accept(visitor);
    BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
    BOOST_CHECK_EQUAL(visitor.value_type(),
        fern::TypeTraits<T>::value_type);
}


BOOST_AUTO_TEST_SUITE(attribute_type_visitor)

BOOST_AUTO_TEST_CASE(attribute_type_visitor)
{
    fern::AttributeTypeVisitor visitor;

    // Check constant attributes.
    CALL_FUNCTION_TEMPLATE_1(test_constant_attribute, visitor);

    // Make sure visiting using a pointer to the base class works too.
    {
        fern::ConstantAttribute<int32_t> int32_attribute;
        fern::Attribute *attribute = &int32_attribute;
        attribute->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
        BOOST_CHECK_EQUAL(visitor.value_type(), fern::VT_INT32);
    }

    // Visit a constant string attribute.
    {
        fern::ConstantAttribute<fern::String> string_attribute;
        fern::Attribute *attribute = &string_attribute;
        attribute->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
        BOOST_CHECK_EQUAL(visitor.value_type(), fern::VT_STRING);
    }
}

BOOST_AUTO_TEST_SUITE_END()
