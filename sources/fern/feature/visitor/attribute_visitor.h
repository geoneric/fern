#pragma once
#include <cstdint>
#include <loki/Visitor.h>
#include "fern/core/string.h"
#include "fern/feature/core/attributes.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AttributeVisitor:
    public Loki::BaseVisitor,
    public Loki::Visitor<ConstantAttribute<int8_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<int16_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<int32_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<int64_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<uint8_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<uint16_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<uint32_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<uint64_t>, void, true>,
    public Loki::Visitor<ConstantAttribute<float>, void, true>,
    public Loki::Visitor<ConstantAttribute<double>, void, true>,
    public Loki::Visitor<ConstantAttribute<String>, void, true>,
    public Loki::Visitor<FieldAttribute<int8_t>, void, true>,
    public Loki::Visitor<FieldAttribute<int16_t>, void, true>,
    public Loki::Visitor<FieldAttribute<int32_t>, void, true>,
    public Loki::Visitor<FieldAttribute<int64_t>, void, true>,
    public Loki::Visitor<FieldAttribute<uint8_t>, void, true>,
    public Loki::Visitor<FieldAttribute<uint16_t>, void, true>,
    public Loki::Visitor<FieldAttribute<uint32_t>, void, true>,
    public Loki::Visitor<FieldAttribute<uint64_t>, void, true>,
    public Loki::Visitor<FieldAttribute<float>, void, true>,
    public Loki::Visitor<FieldAttribute<double>, void, true>,
    public Loki::Visitor<FieldAttribute<String>, void, true>

{

public:

    virtual void   Visit               (Attribute const& attribute);

    virtual void   Visit               (ConstantAttribute<int8_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<int16_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<int32_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<int64_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<uint8_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<uint16_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<uint32_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<uint64_t> const& attribute);

    virtual void   Visit               (ConstantAttribute<float> const& attribute);

    virtual void   Visit               (ConstantAttribute<double> const& attribute);

    virtual void   Visit               (ConstantAttribute<String> const& attribute);

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

protected:

                   AttributeVisitor    ()=default;

                   AttributeVisitor    (AttributeVisitor const&)=delete;

    AttributeVisitor& operator=        (AttributeVisitor const&)=delete;

                   AttributeVisitor    (AttributeVisitor&&)=delete;

    AttributeVisitor& operator=        (AttributeVisitor&&)=delete;

    virtual        ~AttributeVisitor   ()=default;

private:

};


//! Macro that will call the macro passed in for each numeric value type.
/*!
*/
#define VISIT_ATTRIBUTES(                                                      \
        macro)                                                                 \
    macro(int8_t)                                                              \
    macro(int16_t)                                                             \
    macro(int32_t)                                                             \
    macro(int64_t)                                                             \
    macro(uint8_t)                                                             \
    macro(uint16_t)                                                            \
    macro(uint32_t)                                                            \
    macro(uint64_t)                                                            \
    macro(float)                                                               \
    macro(double)                                                              \
    macro(String)

} // namespace fern
