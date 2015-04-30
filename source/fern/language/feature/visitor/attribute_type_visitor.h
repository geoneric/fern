// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/feature/visitor/attribute_visitor.h"
#include "fern/core/data_type.h"
#include "fern/core/value_type.h"


namespace fern {

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

    void           Visit               (Attribute const& attribute);

    void           Visit               (ConstantAttribute<bool> const& attribute);

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

    virtual void   Visit               (FieldAttribute<bool> const& attribute);

    virtual void   Visit               (FieldAttribute<int8_t> const& attribute);

    virtual void   Visit               (FieldAttribute<int16_t> const& attribute);

    virtual void   Visit               (FieldAttribute<int32_t> const& attribute);

    virtual void   Visit               (FieldAttribute<int64_t> const& attribute);

    virtual void   Visit               (FieldAttribute<uint8_t> const& attribute);

    virtual void   Visit               (FieldAttribute<uint16_t> const& attribute);

    virtual void   Visit               (FieldAttribute<uint32_t> const& attribute);

    virtual void   Visit               (FieldAttribute<uint64_t> const& attribute);

    virtual void   Visit               (FieldAttribute<float> const& attribute);

    virtual void   Visit               (FieldAttribute<double> const& attribute);

    virtual void   Visit               (FieldAttribute<String> const& attribute);

    DataType       data_type           () const;

    ValueType      value_type          () const;

protected:

private:

    DataType       _data_type;

    ValueType      _value_type;

};

} // namespace fern
