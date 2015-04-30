// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/command/command.h"
#include "fern/language/io/io_client.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExecuteCommand:
    private IOClient,
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

} // namespace language
} // namespace fern
