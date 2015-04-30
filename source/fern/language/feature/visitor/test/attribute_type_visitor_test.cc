// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/language/feature/core/constant_attribute.h"
#include "fern/language/feature/visitor/attribute_type_visitor.h"


namespace fl = fern::language;


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
    fl::AttributeTypeVisitor& visitor)
{
    fl::ConstantAttribute<T> attribute;
    attribute.Accept(visitor);
    BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
    BOOST_CHECK_EQUAL(visitor.value_type(),
        fern::TypeTraits<T>::value_type);
}


BOOST_AUTO_TEST_SUITE(attribute_type_visitor)

BOOST_AUTO_TEST_CASE(attribute_type_visitor)
{
    fl::AttributeTypeVisitor visitor;

    // Check constant attributes.
    CALL_FUNCTION_TEMPLATE_1(test_constant_attribute, visitor);

    // Make sure visiting using a pointer to the base class works too.
    {
        fl::ConstantAttribute<int32_t> int32_attribute;
        fl::Attribute *attribute = &int32_attribute;
        attribute->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
        BOOST_CHECK_EQUAL(visitor.value_type(), fern::VT_INT32);
    }

    // Visit a constant string attribute.
    {
        fl::ConstantAttribute<fern::String> string_attribute;
        fl::Attribute *attribute = &string_attribute;
        attribute->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.data_type(), fern::DT_CONSTANT);
        BOOST_CHECK_EQUAL(visitor.value_type(), fern::VT_STRING);
    }
}

BOOST_AUTO_TEST_SUITE_END()
