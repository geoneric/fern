#include "fern/feature/visitor/attribute_type_visitor.h"
#include "fern/core/type_traits.h"


namespace fern {

AttributeTypeVisitor::AttributeTypeVisitor()

    : AttributeVisitor(),
      _data_type(DT_LAST_DATA_TYPE),
      _value_type(VT_LAST_VALUE_TYPE)

{
}


void AttributeTypeVisitor::Visit(
    Attribute const& /* attribute */)
{
    // We end up here when an attribute type is visited that isn't supported.
    assert(false);
}


#define VISIT_CONSTANT_ATTRIBUTE(                                              \
    type)                                                                      \
void AttributeTypeVisitor::Visit(                                              \
    ConstantAttribute<type> const& /* attribute */)                            \
{                                                                              \
    _data_type = DT_CONSTANT;                                                  \
    _value_type = TypeTraits<type>::value_type;                                \
}

VISIT_ATTRIBUTES(VISIT_CONSTANT_ATTRIBUTE)

#undef VISIT_CONSTANT_ATTRIBUTE


#define VISIT_FIELD_ATTRIBUTE(                                                 \
    type)                                                                      \
void AttributeTypeVisitor::Visit(                                              \
    FieldAttribute<type> const& /* attribute */)                               \
{                                                                              \
    _data_type = DT_STATIC_FIELD;                                              \
    _value_type = TypeTraits<type>::value_type;                                \
}

VISIT_ATTRIBUTES(VISIT_FIELD_ATTRIBUTE)

#undef VISIT_FIELD_ATTRIBUTE


DataType AttributeTypeVisitor::data_type() const
{
    return _data_type;
}


ValueType AttributeTypeVisitor::value_type() const
{
    return _value_type;
}

} // namespace fern
