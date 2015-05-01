// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <map>
#include "fern/language/ast/core/name_vertex.h"
#include "fern/language/ast/visitor/ast_visitor.h"


namespace fern {
namespace language {

//! Class for visitors that determine input identifiers and potential outputs.
/*!
    Inputs are the undefined identifiers at any scope. Potential outputs are
    the defined identifiers at global scope. In case multiple output
    identifiers with the same name exist, only the last occurrence is
    treated as an output.

    This visitor can be used multiple times, as long as a ModuleVertex accepts
    the visitor first.

    \todo Keep track of scoping and make sure only global scope defined
          identifiers are marked as outputs. Maybe use a symbol table to
          store outputs per scope?
*/
class IOVisitor:
    public AstVisitor
{

public:

                   IOVisitor           ();

                   ~IOVisitor          ();

                   IOVisitor           (IOVisitor&&)=delete;

    IOVisitor&     operator=           (IOVisitor&&)=delete;

                   IOVisitor           (IOVisitor const&)=delete;

    IOVisitor&     operator=           (IOVisitor const&)=delete;

    std::vector<std::string> const&
                   inputs              () const;

    std::vector<NameVertex const*> const&
                   outputs             () const;

private:

    enum class Mode {
        Defining,
        Using
    };

    Mode           _mode;

    //! Names of input identifiers.
    std::vector<std::string> _inputs;

    //! Pointers to output identifiers.
    std::vector<NameVertex const*> _outputs;

    //! Index of output with a certain name in _outputs.
    std::map<std::string, size_t> _output_id_by_name;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (ModuleVertex& vertex);

    void           Visit               (NameVertex& vertex);

};

} // namespace language
} // namespace fern
