#pragma once
#include "geoneric/feature/visitor/attribute_visitor.h"
#include "geoneric/core/data_type.h"
#include "geoneric/core/value_type.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AttributeTypeVisitor:
    public AttributeVisitor
{

public:

                   AttributeTypeVisitor();

                   AttributeTypeVisitor(AttributeTypeVisitor const&)=delete;

    AttributeTypeVisitor& operator=    (AttributeTypeVisitor const&)=delete;

                   AttributeTypeVisitor(AttributeTypeVisitor&&)=delete;

    AttributeTypeVisitor& operator=    (AttributeTypeVisitor&&)=delete;

                   ~AttributeTypeVisitor()=default;

    void           Visit               (ConstantAttribute<int8_t> const& attribute);

    void           Visit               (ConstantAttribute<int16_t> const& attribute);

    void           Visit               (ConstantAttribute<int32_t> const& attribute);

    void           Visit               (ConstantAttribute<int64_t> const& attribute);

    void           Visit               (ConstantAttribute<uint8_t> const& attribute);

    void           Visit               (ConstantAttribute<uint16_t> const& attribute);

    void           Visit               (ConstantAttribute<uint32_t> const& attribute);

    void           Visit               (ConstantAttribute<uint64_t> const& attribute);

    void           Visit               (ConstantAttribute<float> const& attribute);

    void           Visit               (ConstantAttribute<double> const& attribute);

    void           Visit               (ConstantAttribute<String> const& attribute);

    void           Visit               (Attribute const& attribute);

    DataType       data_type           () const;

    ValueType      value_type          () const;

protected:

private:

    DataType       _data_type;

    ValueType      _value_type;

};

} // namespace geoneric
