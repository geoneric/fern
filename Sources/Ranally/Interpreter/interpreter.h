#pragma once
#include "Ranally/Language/algebra_parser.h"
#include "Ranally/Language/script_vertex.h"
#include "Ranally/Language/xml_parser.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Interpreter
{

    friend class InterpreterTest;

public:

                   Interpreter         ();

                   ~Interpreter        ()=default;

                   Interpreter         (Interpreter&&)=delete;

    Interpreter&   operator=           (Interpreter&&)=delete;

                   Interpreter         (Interpreter const&)=delete;

    Interpreter&   operator=           (Interpreter const&)=delete;

    ScriptVertexPtr parseString        (String const& string);

    void           annotate            (ScriptVertexPtr const& tree);

    void           validate            (ScriptVertexPtr const& tree);

    void           execute             (ScriptVertexPtr const& tree);

private:

    AlgebraParser  _algebraParser;

    XmlParser      _xmlParser;

};

} // namespace ranally
