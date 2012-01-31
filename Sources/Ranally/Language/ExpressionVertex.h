#ifndef INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX

#include <vector>
#include <boost/tuple/tuple.hpp>
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"
#include "Ranally/Language/StatementVertex.h"



namespace ranally {
namespace language {

class ExpressionVertex;

typedef boost::shared_ptr<ExpressionVertex> ExpressionVertexPtr;

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

  typedef boost::tuple<operation::DataType, operation::ValueType> ResultType;

  virtual          ~ExpressionVertex   ();

  UnicodeString const& name            () const;

  // void             setDataType         (operation::DataType dataType);

  // operation::DataType dataType         () const;

  // void             setValueType        (operation::ValueType valueType);

  // operation::ValueType valueType       () const;

  void             setResultTypes      (
                                  std::vector<ResultType> const& resultTypes);

  void             addResultType       (operation::DataType dataType,
                                        operation::ValueType valueType);

  std::vector<ResultType> const& resultTypes() const;

  operation::DataType dataType         (size_t index) const;

  operation::ValueType valueType       (size_t index) const;

  void             setValue            (ExpressionVertexPtr const& value);

  ExpressionVertexPtr const& value     () const;

protected:

                   ExpressionVertex    (UnicodeString const& name);

                   ExpressionVertex    (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

private:

  //! Name of the expression, eg: abs, myDog, 5.
  UnicodeString    _name;

  // //! Data type of the result(s) of the expression.
  // operation::DataType _dataType;

  // //! Value type of the result(s) of the expression.
  // operation::ValueType _valueType;

  // typedef boost::tuple<operation::DataType, operation::ValueType> ResultType;

  std::vector<ResultType> _resultTypes;

  //! Value of expression, in case this expression is the target of an assignment.
  ExpressionVertexPtr _value;

};

} // namespace language
} // namespace ranally

#endif
