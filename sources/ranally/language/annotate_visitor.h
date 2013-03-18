#pragma once
#include "ranally/operation/operations.h"
#include "ranally/language/visitor.h"



namespace ranally {

//! Class for visitors that annotate the syntax tree.
/*!
  For each operation in the tree, the requirements are looked up. Some of these
  are stored in the operation vertex (eg: number of results, number of
  arguments) and others are stored in the expression vertices (eg: data type,
  value type).

  Apart from determining whether each operation exists or not, no validation
  is performed by this visitor. It only annotates the tree. Use a
  ValidateVisitor to perform the actual validation.

  For example, in case of a FunctionVertex, it is no problem if the operation
  is not known. Annotation is optional. The ValidateVisitor will check if all
  information required for execution is present.

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
    public Visitor
{

    friend class AnnotateVisitorTest;

public:

                   AnnotateVisitor     (OperationsPtr const& operations);

                   ~AnnotateVisitor    ()=default;

                   AnnotateVisitor     (AnnotateVisitor&&)=delete;

    AnnotateVisitor& operator=         (AnnotateVisitor&&)=delete;

                   AnnotateVisitor     (AnnotateVisitor const&)=delete;

    AnnotateVisitor& operator=         (AnnotateVisitor const&)=delete;

private:

    // Annotation calculates as many data and value types of expressions as
    // possible.
    //
    // We start with this information:
    // - result types for numbers
    // - result types for operations
    //   - if the result types are not dependent on the data/value types of
    //     the parameters.
    //   - else, if the result types of all parameters are known, the result
    //     type of the operation is calculated.
    // We start with propagating that information across the assignment.
    // We propagate the information from the defined names to the use
    // locations. This may result in more operations knowing data/value
    // types of all parameters.
    // Iterate untill no additional information can be generated anymore.
    enum Mode {
        //! Right sight of assignment statements is visited.
        Using,

        //! Left side of assignment statements is visited.
        Defining
    };

    //! Current mode.
    Mode           _mode;

    OperationsPtr  _operations;

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

    void           Visit               (OperationVertex& vertex);

    void           Visit               (ScriptVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

};

} // namespace ranally
