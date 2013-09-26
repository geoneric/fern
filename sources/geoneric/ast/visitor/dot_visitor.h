#pragma once
#include "geoneric/core/string.h"
#include "geoneric/ast/visitor/ast_visitor.h"


namespace geoneric {

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

    String const&  script              () const;

protected:

                   DotVisitor          ()=default;

    void           set_script          (String const& string);

    void           add_script          (String const& string);

private:

    String         _script;

    virtual void   Visit               (ModuleVertex& vertex)=0;

};

} // namespace geoneric
