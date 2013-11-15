#pragma once
#include "fern/ast/visitor/ast_visitor.h"
#include "fern/operation/core/operations.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class BackEnd:
    public AstVisitor
{

public:

                   BackEnd             (BackEnd const&)=delete;

    BackEnd&       operator=           (BackEnd const&)=delete;

                   BackEnd             (BackEnd&&)=delete;

    BackEnd&       operator=           (BackEnd&&)=delete;

    virtual        ~BackEnd            ()=default;

    OperationsPtr const& operations    () const;

protected:

                   BackEnd             (OperationsPtr const& operations);

private:

    OperationsPtr  _operations;

};

} // namespace fern
