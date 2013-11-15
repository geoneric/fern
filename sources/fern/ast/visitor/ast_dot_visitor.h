#pragma once
#include <stack>
#include "fern/ast/visitor/dot_visitor.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AstDotVisitor:
    public DotVisitor
{

    friend class AstDotVisitorTest;

public:

    enum Mode {
        Declaring=0x1,
        ConnectingAst=0x2,
        ConnectingCfg=0x4,
        ConnectingUses=0x8
    };

                   AstDotVisitor       (int modes=0);

                   ~AstDotVisitor      ()=default;

                   AstDotVisitor       (AstDotVisitor&&)=delete;

    AstDotVisitor& operator=           (AstDotVisitor&&)=delete;

                   AstDotVisitor       (AstDotVisitor const&)=delete;

    AstDotVisitor& operator=           (AstDotVisitor const&)=delete;

private:

    //! Current mode.
    Mode           _mode;

    //! Modes to process.
    int            _modes;

    //! Names of scopes (module, user defined function, ...).
    std::stack<std::string> _scope_names;

    void           set_mode            (Mode mode);

    void           add_ast_vertex      (AstVertex const& sourceVertex,
                                        AstVertex const& targetVertex);

    // void           add_ast_vertex      (AstVertex const& source_vertex,
    //                                     ScopeVertex const& target_vertex);

    // void           add_ast_vertex      (ScopeVertex const& source_vertex,
    //                                     AstVertex const& target_vertex);

    void           add_cfg_vertices    (AstVertex const& sourceVertex);

    void           add_use_vertices    (NameVertex const& vertex);

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (FunctionCallVertex& vertex);

    void           Visit               (FunctionDefinitionVertex& vertex);

    void           Visit               (IfVertex& vertex);

    void           Visit               (ModuleVertex& vertex);

    void           Visit               (NameVertex& vertex);

    template<typename T>
    void           Visit               (NumberVertex<T>& vertex);

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

    void           Visit               (OperatorVertex& vertex);

    void           Visit               (ReturnVertex& vertex);

    void           Visit               (ScopeVertex& vertex);

    void           Visit               (SentinelVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace fern
