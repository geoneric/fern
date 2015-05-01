// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/visitor/ast_visitor.h"


namespace fern {
namespace language {

//! Base class for visitors emitting dot graphs.
/*!
  The dot graphs are useful for debugging purposes. The graphs are handy
  for visualising the syntax-tree.
*/
class DotVisitor:
    public AstVisitor
{

    friend class DotVisitorTest;

public:

    virtual        ~DotVisitor         ()=default;

                   DotVisitor          (DotVisitor&&)=delete;

    DotVisitor&    operator=           (DotVisitor&&)=delete;

                   DotVisitor          (DotVisitor const&)=delete;

    DotVisitor&    operator=           (DotVisitor const&)=delete;

    std::string const&
                   script              () const;

protected:

                   DotVisitor          ()=default;

    void           set_script          (std::string const& string);

    void           add_script          (std::string const& string);

private:

    std::string    _script;

    virtual void   Visit               (ModuleVertex& vertex)=0;

};

} // namespace language
} // namespace fern
