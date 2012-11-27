#pragma once
#include "ranally/core/string.h"
#include "ranally/language/visitor.h"


namespace ranally {

//! Base class for visitors emitting dot graphs.
/*!
  The dot graphs are useful for debugging purposes. The graphs are handy
  for visualising the syntax-tree.
*/
class DotVisitor:
    public Visitor
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

    void           setScript           (String const& string);

    void           addScript           (String const& string);

private:

    String         _script;

    virtual void   Visit               (ScriptVertex& vertex)=0;

};

} // namespace ranally
