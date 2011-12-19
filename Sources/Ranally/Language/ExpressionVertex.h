#ifndef INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX

#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"
#include "Ranally/Language/StatementVertex.h"



namespace ranally {
namespace language {

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

  virtual          ~ExpressionVertex   ();

  UnicodeString const& name            () const;

  void             setDataType         (operation::DataType dataType);

  operation::DataType dataType         () const;

  void             setValueType        (operation::ValueType valueType);

  operation::ValueType valueType       () const;

protected:

                   ExpressionVertex    (UnicodeString const& name);

                   ExpressionVertex    (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

private:

  //! Name of the expression, eg: abs, myDog, 5.
  UnicodeString    _name;

  // TODO Use a collection of operation::Result's for this(?). Or a tuple.

  //! Data type of the result(s) of the expression.
  operation::DataType _dataType;

  //! Value type of the result(s) of the expression.
  operation::ValueType _valueType;

};

} // namespace language
} // namespace ranally

#endif
