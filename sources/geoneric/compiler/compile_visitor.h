#pragma once
#include "geoneric/ast/visitor/ast_visitor.h"


namespace geoneric {

class CompileVisitor:
    public AstVisitor
{

public:

                   CompileVisitor      (String const& header_filename);

                   ~CompileVisitor     ();

                   CompileVisitor      (CompileVisitor&&)=delete;

    CompileVisitor& operator=          (CompileVisitor&&)=delete;

                   CompileVisitor      (CompileVisitor const&)=delete;

    CompileVisitor& operator=          (CompileVisitor const&)=delete;

    String const&  header              () const;

    String const&  module              () const;

private:

    String         _header_filename;

    String         _header;

    String         _module;

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (IfVertex& vertex);

    void           Visit               (ModuleVertex& vertex);

    void           Visit               (NameVertex& vertex);

    void           Visit               (NumberVertex<int8_t>& vertex);

    void           Visit               (NumberVertex<int16_t>& vertex);

    void           Visit               (NumberVertex<int32_t>& vertex);

    void           Visit               (NumberVertex<int64_t>& vertex);

    void           Visit               (NumberVertex<uint8_t>& vertex);

    void           Visit               (NumberVertex<uint16_t>& vertex);

    void           Visit               (NumberVertex<uint32_t>& vertex);

    void           Visit               (NumberVertex<uint64_t>& vertex);

    void           Visit               (NumberVertex<float>& vertex);

    void           Visit               (NumberVertex<double>& vertex);

    void           Visit               (OperationVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace geoneric
