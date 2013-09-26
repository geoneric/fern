#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
#include "geoneric/core/type_traits.h"
#include "geoneric/feature/core/constant_attribute.h"
#include "geoneric/feature/visitor/attribute_type_visitor.h"


#define CALL_FUNCTION_TEMPLATE_1(                                              \
        function_template,                                                     \
        argument)                                                              \
    function_template<int8_t>(argument);                                       \
    function_template<int16_t>(argument);                                      \
    function_template<int32_t>(argument);                                      \
    function_template<int64_t>(argument);                                      \
    function_template<uint8_t>(argument);                                      \
    function_template<uint16_t>(argument);                                     \
    function_template<uint32_t>(argument);                                     \
    function_template<uint64_t>(argument);                                     \
    function_template<float>(argument);                                        \
    function_template<double>(argument);


template<
    class T>
void test_constant_scalar_attribute(
    geoneric::AttributeTypeVisitor& visitor)
{
    geoneric::ConstantAttribute<T> attribute;
    attribute.Accept(visitor);
    BOOST_CHECK_EQUAL(visitor.data_type(), geoneric::DT_SCALAR);
    BOOST_CHECK_EQUAL(visitor.value_type(),
        geoneric::TypeTraits<T>::value_type);
}


BOOST_AUTO_TEST_SUITE(attribute_type_visitor)

BOOST_AUTO_TEST_CASE(attribute_type_visitor)
{
    geoneric::AttributeTypeVisitor visitor;

    // Check constant scalar attributes.
    CALL_FUNCTION_TEMPLATE_1(test_constant_scalar_attribute, visitor);

    // Make sure visiting using a pointer to the base class works too.
    {
        geoneric::ConstantAttribute<int32_t> int32_attribute;
        geoneric::Attribute *attribute = &int32_attribute;
        attribute->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.data_type(), geoneric::DT_SCALAR);
        BOOST_CHECK_EQUAL(visitor.value_type(), geoneric::VT_INT32);
    }
}

BOOST_AUTO_TEST_SUITE_END()
