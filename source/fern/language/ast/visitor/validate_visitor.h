// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/visitor/ast_visitor.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  This visitor can assume the tree is fully annotated. Any missing
  information must be reported. It means that the information is not
  available. The AnnotateVisitor tries its best to find information but
  won't report errors. That's the task of the ValidateVisitor.

  \todo      In case of subscript vertices, the preferred scope for names is
             the list of attributes of the expression being subscripted.
  \todo      Operation vertices cannot be used on the left side of an
             assignment (defining). Throw an error.
  \todo      Subscript vertices cannot be used on the left side of an
             assignment (defining)(?). Throw an error.
  \todo      Compare each expression's expression_type with the expression's
             requirements. For example, operations can be checked against the
             operation's definition as read from the XML.
             An OperationVertex node is annotated according to the parameter
             types. This may result in a mismatch.

  \sa        AnnotateVisitor
*/
class ValidateVisitor:
    public AstVisitor
{

public:

                   ValidateVisitor     ()=default;

                   ~ValidateVisitor    ()=default;

                   ValidateVisitor     (ValidateVisitor&&)=delete;

    ValidateVisitor& operator=         (ValidateVisitor&&)=delete;

                   ValidateVisitor     (ValidateVisitor const&)=delete;

    ValidateVisitor& operator=         (ValidateVisitor const&)=delete;

private:

    void           Visit               (NameVertex& vertex);

    void           Visit               (OperationVertex& vertex);

};

} // namespace language
} // namespace fern
