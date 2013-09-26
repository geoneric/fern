#pragma once
#include <loki/Visitor.h>


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Attribute:
    public Loki::BaseVisitable<void, Loki::DefaultCatchAll, true>
{

public:

    LOKI_DEFINE_CONST_VISITABLE()

protected:

                   Attribute           ()=default;

                   Attribute           (Attribute const&)=delete;

    Attribute&     operator=           (Attribute const&)=delete;

                   Attribute           (Attribute&&)=delete;

    Attribute&     operator=           (Attribute&&)=delete;

    virtual        ~Attribute          ()=default;

private:

};

} // namespace geoneric
