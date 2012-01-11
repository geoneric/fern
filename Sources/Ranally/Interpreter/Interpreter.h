#ifndef INCLUDED_RANALLY_INTERPRETER_INTERPRETER
#define INCLUDED_RANALLY_INTERPRETER_INTERPRETER

#include <boost/noncopyable.hpp>
#include "Ranally/Language/ScriptVertex.h"



namespace ranally {
namespace interpreter {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Interpreter:
  private boost::noncopyable
{

  friend class InterpreterTest;

public:

                   Interpreter         ();

                   ~Interpreter        ();

  void             annotate            (language::ScriptVertexPtr const& tree);

  void             validate            (language::ScriptVertexPtr const& tree);

  void             execute             (language::ScriptVertexPtr const& tree);

private:

};

} // namespace interpreter
} // namespace ranally

#endif
