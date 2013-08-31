#pragma once
#include "geoneric/command/command.h"


namespace geoneric {

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

                   ~ExecuteCommand     ()=default;

                   ExecuteCommand      (ExecuteCommand&&)=delete;

    ExecuteCommand& operator=          (ExecuteCommand&&)=delete;

                   ExecuteCommand      (ExecuteCommand const&)=delete;

    ExecuteCommand& operator=          (ExecuteCommand const&)=delete;

    int            execute             () const;

private:

    void           execute             (ModuleVertexPtr const& tree) const;

};

} // namespace geoneric
