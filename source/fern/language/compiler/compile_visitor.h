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
#include "fern/language/operation/core/operations.h"


namespace fern {
namespace language {

class CompileVisitor:
    public AstVisitor
{

public:

                   CompileVisitor      (OperationsPtr const& operations,
                                        std::string const& header_filename);

                   ~CompileVisitor     ();

                   CompileVisitor      (CompileVisitor&&)=delete;

    CompileVisitor& operator=          (CompileVisitor&&)=delete;

                   CompileVisitor      (CompileVisitor const&)=delete;

    CompileVisitor& operator=          (CompileVisitor const&)=delete;

    std::string const&
                   header              () const;

    std::string const&
                   module              () const;

private:

    OperationsPtr  _operations;

    std::string    _header_filename;

    std::string    _header;

    //! A statement of C++ code.
    std::string    _statement;

    //! The C++ code of the translated tree.
    std::vector<std::string> _body;

    std::string    _module;

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

    void           Visit               (StatementVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace language
} // namespace fern
