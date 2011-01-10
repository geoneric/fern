#ifndef INCLUDED_RANALLY_ASTDOTVISITOR
#define INCLUDED_RANALLY_ASTDOTVISITOR

#include <boost/noncopyable.hpp>

#include "DotVisitor.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AstDotVisitor: public DotVisitor
{

  friend class AstDotVisitorTest;

private:

protected:

public:

                   AstDotVisitor       ();

  /* virtual */    ~AstDotVisitor      ();

  void             Visit               (language::AssignmentVertex& vertex);

  void             Visit               (language::FunctionVertex& vertex);

  void             Visit               (language::IfVertex& vertex);

  void             Visit               (language::NameVertex& vertex);

  void             Visit               (
                                  language::NumberVertex<int8_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<int16_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<int32_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<int64_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<uint8_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<uint16_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<uint32_t>& vertex);

  void             Visit               (
                                  language::NumberVertex<uint64_t>& vertex);

  void             Visit               (language::NumberVertex<float>& vertex);

  void             Visit               (language::NumberVertex<double>& vertex);

  void             Visit               (language::OperatorVertex& vertex);

  void             Visit               (language::ScriptVertex& vertex);

  void             Visit               (language::StringVertex& vertex);

  void             Visit               (language::WhileVertex& vertex);

};

} // namespace ranally

#endif
