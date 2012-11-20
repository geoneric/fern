#pragma once
#include "Ranally/Util/string.h"


namespace ranally {

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

    virtual int    execute             ()=0;

protected:

                   Command             (int argc,
                                        char** argv);

    int            argc                () const;

    char**         argv                () const;

    String         read                (std::string const& fileName);

    void           write               (String const& contents,
                                        std::string const& fileName);

private:

    int            _argc;

    char**         _argv;

};

} // namespace ranally
