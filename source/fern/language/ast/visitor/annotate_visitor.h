// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <stack>
#include "fern/core/expression_type.h"
#include "fern/language/core/symbol_table.h"
#include "fern/language/operation/core/operations.h"
#include "fern/language/ast/visitor/ast_visitor.h"



namespace fern {
namespace language {

//! Class for visitors that annotate the syntax tree.
/*!
  For each operation in the tree, the requirements are looked up. Some of these
  are stored in the operation vertex (eg: number of results, number of
  arguments) and others are stored in the expression vertices (eg: data type,
  value type).

  Apart from determining whether each operation exists or not, no validation
  is performed by this visitor. It only annotates the tree. Use a
  ValidateVisitor to perform the actual validation.

  For example, in case of a FunctionCallVertex, it is no problem if the
  operation is not known. Annotation is optional. The ValidateVisitor
  will check if all information required for execution is present.

  \todo      In case of subscript vertices, the names used in the selection
             expression should be first looked up in the list of attributes
             of the expression being subscripted. If not found there, the names
             refer to identifiers in the enclosing scope. For this to work,
             we need to know about the attributes of the main expression.
             if name in list of attribute
               name is feature attribute
             else
               name refers to name defined elsewhere
             fi
  \todo      Set the data types and value types of expressions. Use a stack
             and symbol table to simulate run time behavior and set each
             expression's data and value types, if possible.

  \sa        ValidateVisitor
*/
class AnnotateVisitor:
    public AstVisitor
{

    friend class AnnotateVisitorTest;

public:

                   AnnotateVisitor     (OperationsPtr const& operations);

                   ~AnnotateVisitor    ();

                   AnnotateVisitor     (AnnotateVisitor&&)=delete;

    AnnotateVisitor& operator=         (AnnotateVisitor&&)=delete;

                   AnnotateVisitor     (AnnotateVisitor const&)=delete;

    AnnotateVisitor& operator=         (AnnotateVisitor const&)=delete;

    // void           set_external_symbol_table(
    //                                     SymbolTable<ExpressionType> const&
    //                                         symbol_table);

    void           add_global_symbols  (SymbolTable<ExpressionType> const&
                                            symbol_table);

    std::stack<ExpressionType> const& stack() const;

    // void           reset               ();

    void           clear_stack         ();

private:

    OperationsPtr  _operations;

    std::stack<ExpressionType> _stack;

    SymbolTable<ExpressionType> _symbol_table;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (NameVertex& vertex);

    void           Visit               (NumberVertex<int8_t>& vertex);

    void           Visit               (NumberVertex<int16_t>& vertex);

    void           Visit               (NumberVertex<int32_t>& vertex);

    void           Visit               (NumberVertex<int64_t>& vertex);

    void           Visit               (NumberVertex<uint8_t>& vertex);

    void           Visit               (NumberVertex<uint16_t>& vertex);

    void           Visit               (NumberVertex<uint32_t>& vertex);

    void           Visit               (NumberVertex<uint64_t>& vertex);

    void           Visit               (NumberVertex<float>& vertex);

    void           Visit               (NumberVertex<double>& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (OperationVertex& vertex);

    // void           Visit               (SubscriptVertex& vertex);

    void           Visit               (ModuleVertex& vertex);

};

} // namespace language
} // namespace fern
