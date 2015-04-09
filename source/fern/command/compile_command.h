// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/command/command.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class CompileCommand:
    public Command
{

    friend class CompileCommandTest;

public:

                   CompileCommand      (int argc,
                                        char** argv);

                   ~CompileCommand     ()=default;

                   CompileCommand      (CompileCommand&&)=delete;

    CompileCommand& operator=          (CompileCommand&&)=delete;

                   CompileCommand      (CompileCommand const&)=delete;

    CompileCommand& operator=          (CompileCommand const&)=delete;

    int            execute             () const;

private:

    int            compile_to_fern     (int argc,
                                        char** argv) const;

    int            compile_to_cpp      (int argc,
                                        char** argv) const;

    String         compile_to_dot_ast  (ModuleVertexPtr const& tree,
                                        int modes) const;

    int            compile_to_dot_ast  (int argc,
                                        char** argv) const;

    String         compile_to_dot_flowgraph(
                                        ModuleVertexPtr const& tree) const;

    int            compile_to_dot_flowgraph(
                                        int argc,
                                        char** argv) const;

    int            compile_to_dot      (int argc,
                                        char** argv) const;

    int            compile_to_python   (int argc,
                                        char** argv) const;

    int            compile_to_xml      (int argc,
                                        char** argv) const;

};

} // namespace fern
