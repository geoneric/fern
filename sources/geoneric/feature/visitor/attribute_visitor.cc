#include "geoneric/feature/visitor/attribute_visitor.h"
#include <cassert>
#include <iostream>
#include "geoneric/feature/core/constant_attribute.h"


namespace geoneric {

void AttributeVisitor::Visit(
    Attribute const& /* attribute */)
{
    // Unsupported attribute type!
    // The default implementation does nothing.
}


// The default implementation calls Visit(Attribute&).
#define VISIT_NUMBER_ATTRIBUTE(                                                \
    type)                                                                      \
void AttributeVisitor::Visit(                                                  \
    ConstantAttribute<type> const& attribute)                                  \
{                                                                              \
    Visit(dynamic_cast<Attribute const&>(attribute));                          \
}

VISIT_NUMBER_ATTRIBUTES(VISIT_NUMBER_ATTRIBUTE)

#undef VISIT_NUMBER_ATTRIBUTE

} // namespace geoneric
