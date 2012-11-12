#pragma once
#include "Command.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExecuteCommand:
    public Command
{

    friend class ExecuteCommandTest;

public:

                   ExecuteCommand      (int argc,
                                        char** argv);

                   ~ExecuteCommand     () noexcept(true) =default;

                   ExecuteCommand      (ExecuteCommand&&)=delete;

    ExecuteCommand& operator=          (ExecuteCommand&&)=delete;

                   ExecuteCommand      (ExecuteCommand const&)=delete;

    ExecuteCommand& operator=          (ExecuteCommand const&)=delete;

    int            execute             ();

private:

    void           execute             (String const& xml);

};

} // namespace ranally
