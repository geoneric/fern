// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/path.h"
#include "fern/interpreter/interpreter.h"


namespace fern {

//! A Compiler translates modules to C++ code.
/*!

    \sa        .
*/
class Compiler
{

public:

    // namespace flags {
    //     constexpr int
    //         EMPTY = 1,
    //         DUMP_DRIVER = 1 << 1,
    //         DUMP_CMAKE = 1 << 2;
    // }

    enum CompilerFlags {
        //! Write a test driver for the model being compiled.
        DUMP_DRIVER = 1<<1,

        //! Write CMake project file for the model being compiled.
        DUMP_CMAKE = 1<<2
    };

    using Flags = int;

                   Compiler            (String const& header_extension,
                                        String const& module_extension);

                   ~Compiler           ()=default;

                   Compiler            (Compiler&&)=delete;

    Compiler&      operator=           (Compiler&&)=delete;

                   Compiler            (Compiler const&)=delete;

    Compiler&      operator=           (Compiler const&)=delete;

    void           compile             (Path const& source_module_path,
                                        Path const& destination_module_path,
                                        Flags flags=0);

private:

    Interpreter    _interpreter;

    String const   _header_extension;

    String const   _module_extension;

};

} // namespace fern
