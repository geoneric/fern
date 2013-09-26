#include "geoneric/feature/visitor/attribute_type_visitor.h"
#include "geoneric/core/type_traits.h"


namespace geoneric {

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
    // In that case, data type and value type are set to special values.
    _data_type = DT_LAST_DATA_TYPE;
    _value_type = VT_LAST_VALUE_TYPE;
}


#define VISIT_SCALAR_ATTRIBUTE(                                                \
    type)                                                                      \
void AttributeTypeVisitor::Visit(                                              \
    ConstantAttribute<type> const& /* attribute */)                            \
{                                                                              \
    _data_type = DT_SCALAR;                                                    \
    _value_type = TypeTraits<type>::value_type;                                \
}

VISIT_SCALAR_ATTRIBUTES(VISIT_SCALAR_ATTRIBUTE)

#undef VISIT_SCALAR_ATTRIBUTE


DataType AttributeTypeVisitor::data_type() const
{
    return _data_type;
}


ValueType AttributeTypeVisitor::value_type() const
{
    return _value_type;
}

} // namespace geoneric
