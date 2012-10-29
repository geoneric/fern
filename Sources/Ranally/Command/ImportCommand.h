#pragma once
#include "Command.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
    longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

    \sa        .
*/
class ImportCommand:
    public Command
{

    friend class ImportCommandTest;

public:

                   ImportCommand       (int argc,
                                        char** argv);

                   ~ImportCommand      ();

    int            execute             ();

private:

};

} // namespace ranally
