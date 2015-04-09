// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/feature/visitor/attribute_visitor.h"
#include <cassert>
#include <iostream>


namespace fern {

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

} // namespace fern
