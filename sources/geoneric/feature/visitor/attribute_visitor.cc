#include "geoneric/feature/visitor/attribute_visitor.h"
#include <cassert>
#include <iostream>


namespace geoneric {

void AttributeVisitor::Visit(
    Attribute const& /* attribute */)
{
    // Unsupported attribute type!
    // The default implementation does nothing.
}


// The default implementation calls Visit(Attribute&).
#define VISIT_CONSTANT_ATTRIBUTE(                                              \
    type)                                                                      \
void AttributeVisitor::Visit(                                                  \
    ConstantAttribute<type> const& attribute)                                  \
{                                                                              \
    Visit(dynamic_cast<Attribute const&>(attribute));                          \
}

VISIT_ATTRIBUTES(VISIT_CONSTANT_ATTRIBUTE)

#undef VISIT_CONSTANT_ATTRIBUTE


// The default implementation calls Visit(Attribute&).
#define VISIT_FIELD_ATTRIBUTE(                                                 \
    type)                                                                      \
void AttributeVisitor::Visit(                                                  \
    FieldAttribute<type> const& attribute)                                     \
{                                                                              \
    Visit(dynamic_cast<Attribute const&>(attribute));                          \
}

VISIT_ATTRIBUTES(VISIT_FIELD_ATTRIBUTE)

#undef VISIT_FIELD_ATTRIBUTE

} // namespace geoneric
