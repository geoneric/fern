#pragma once
#include "fern/core/string.h"
#include "fern/interpreter/interpreter.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Command
{

    friend class CommandTest;

public:

                   Command             (Command&&)=delete;

    Command&       operator=           (Command&&)=delete;

                   Command             (Command const&)=delete;

    Command&       operator=           (Command const&)=delete;

    virtual        ~Command            ()=default;

    virtual int    execute             () const=0;

protected:

                   Command             (int argc,
                                        char** argv);

    int            argc                () const;

    char**         argv                () const;

    Interpreter const& interpreter     () const;

    void           write               (String const& contents,
                                        std::string const& filename) const;

private:

    int            _argc;

    char**         _argv;

    Interpreter    _interpreter;

};

} // namespace fern
