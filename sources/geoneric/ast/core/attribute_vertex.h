#pragma once
#include "geoneric/ast/core/name_vertex.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AttributeVertex:
    public ExpressionVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   AttributeVertex     (
                        std::shared_ptr<ExpressionVertex> const& expression,
                        String const& member_name);

                   ~AttributeVertex    ()=default;

                   AttributeVertex     (AttributeVertex&&)=delete;

    AttributeVertex& operator=         (AttributeVertex&&)=delete;

                   AttributeVertex     (AttributeVertex const&)=delete;

    AttributeVertex& operator=         (AttributeVertex const&)=delete;

    String const&  symbol              () const;

    std::shared_ptr<ExpressionVertex> const& expression() const;

    String const&  member_name         () const;

private:

    String const   _symbol;

    ExpressionVertexPtr _expression;

    String         _member_name;

};

} // namespace geoneric
