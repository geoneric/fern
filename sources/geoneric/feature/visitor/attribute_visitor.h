#pragma once
#include <cstdint>
#include <loki/Visitor.h>
#include "geoneric/feature/core/attribute.h"


namespace geoneric {

template<class T>
    class ConstantAttribute;

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
    public Loki::Visitor<ConstantAttribute<double>, void, true>
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
#define VISIT_NUMBER_ATTRIBUTES(                                               \
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
    macro(double)

} // namespace geoneric
