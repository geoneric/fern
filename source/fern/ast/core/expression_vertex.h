// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/operation/core/expression_types.h"
#include "fern/ast/core/statement_vertex.h"


namespace fern {

class ExpressionVertex;

using ExpressionVertexPtr = std::shared_ptr<ExpressionVertex>;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  TODO Make sure expressions support multiple results.

  \sa        .
*/
class ExpressionVertex:
    public StatementVertex
{

    friend class ExpressionVertexTest;

public:

    virtual        ~ExpressionVertex   ()=default;

                   ExpressionVertex    (ExpressionVertex&&)=delete;

    ExpressionVertex& operator=        (ExpressionVertex&&)=delete;

                   ExpressionVertex    (ExpressionVertex const&)=delete;

    ExpressionVertex& operator=        (ExpressionVertex const&)=delete;

    String const&  name                () const;

    void           set_expression_types(
                                  ExpressionTypes const& expression_types);

    void           add_result_type     (ExpressionType const& expression_type);

    ExpressionTypes const& expression_types() const;

    void           set_value           (ExpressionVertexPtr const& value);

    ExpressionVertexPtr const& value   () const;

protected:

                   ExpressionVertex    (String const& name);

                   ExpressionVertex    (int line_nr,
                                        int col_id,
                                        String const& name);

private:

    //! Name of the expression, eg: abs, myDog, 5.
    String         _name;

    ExpressionTypes _expression_types;

    //! Value of expression, in case this expression is the target of an assignment.
    ExpressionVertexPtr _value;

};

} // namespace fern
