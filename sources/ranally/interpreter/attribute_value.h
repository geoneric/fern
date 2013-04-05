#pragma once
#include <memory>
#include "ranally/interpreter/value.h"


namespace ranally {

class Attribute;

namespace interpreter {

//! TODO
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AttributeValue:
    public Value
{

public:

                   AttributeValue      (
                                  std::shared_ptr<Attribute> const& attribute);

                   ~AttributeValue     ()=default;

                   AttributeValue      (AttributeValue&&)=delete;

    AttributeValue& operator=          (AttributeValue&&)=delete;

                   AttributeValue      (AttributeValue const&)=delete;

    AttributeValue& operator=          (AttributeValue const&)=delete;

    std::shared_ptr<Attribute> const& attribute() const;

private:

    std::shared_ptr<Attribute> _attribute;

};

} // namespace interpreter
} // namespace ranally
