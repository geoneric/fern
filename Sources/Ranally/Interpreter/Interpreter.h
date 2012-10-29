#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/XmlParser.h"


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

    language::ScriptVertexPtr parseString(String const& string);

    void           annotate            (language::ScriptVertexPtr const& tree);

    void           validate            (language::ScriptVertexPtr const& tree);

    void           execute             (language::ScriptVertexPtr const& tree);

private:

    ranally::language::AlgebraParser _algebraParser;

    ranally::language::XmlParser _xmlParser;

};

} // namespace interpreter
} // namespace ranally
