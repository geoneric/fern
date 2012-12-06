#pragma once
#include "ranally/command/command.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DescribeCommand:
    public Command
{

    friend class DescribeCommandTest;

public:

                   DescribeCommand     (int argc,
                                        char** argv);

                   DescribeCommand     (DescribeCommand&&)=delete;

    DescribeCommand& operator=         (DescribeCommand&&)=delete;

                   DescribeCommand     (DescribeCommand const&)=delete;

    DescribeCommand& operator=         (DescribeCommand const&)=delete;

                   ~DescribeCommand    () noexcept(true); // =default;

    int            execute             () const;

private:

    void           describe            (ScriptVertexPtr const& tree) const;

};

} // namespace ranally