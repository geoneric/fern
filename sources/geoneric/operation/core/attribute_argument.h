#pragma once
#include <memory>
#include "geoneric/operation/core/argument.h"


namespace geoneric {

class Attribute;

//! Class for instances holding on to an attribute.
/*!
  An attribute represents a property of a feature. There are differnt kinds
  of attributes. This class holds on to a pointer to a general Attribute
  instance.
*/
class AttributeArgument:
    public Argument
{

public:

                   AttributeArgument      (
                                  std::shared_ptr<Attribute> const& attribute);

                   ~AttributeArgument     ()=default;

                   AttributeArgument      (AttributeArgument&&)=delete;

    AttributeArgument& operator=          (AttributeArgument&&)=delete;

                   AttributeArgument      (AttributeArgument const&)=delete;

    AttributeArgument& operator=          (AttributeArgument const&)=delete;

    std::shared_ptr<Attribute> const& attribute() const;

private:

    std::shared_ptr<Attribute> _attribute;

};

} // namespace geoneric
