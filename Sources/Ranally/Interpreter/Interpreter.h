#pragma once
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/XmlParser.h"


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

                   Interpreter         (Interpreter const&)=delete;

    Interpreter&   operator=           (Interpreter const&)=delete;

                   ~Interpreter        ();

    ScriptVertexPtr parseString        (String const& string);

    void           annotate            (ScriptVertexPtr const& tree);

    void           validate            (ScriptVertexPtr const& tree);

    void           execute             (ScriptVertexPtr const& tree);

private:

    AlgebraParser  _algebraParser;

    XmlParser      _xmlParser;

};

} // namespace ranally
