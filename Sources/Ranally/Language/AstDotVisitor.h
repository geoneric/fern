#pragma once
#include "Ranally/Language/DotVisitor.h"


namespace ranally {

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

    void           setMode             (Mode mode);

    void           addAstVertex        (SyntaxVertex const& sourceVertex,
                                        SyntaxVertex const& targetVertex);

    void           addCfgVertices      (SyntaxVertex const& sourceVertex);

    void           addUseVertices      (NameVertex const& vertex);

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (FunctionVertex& vertex);

    void           Visit               (IfVertex& vertex);

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

    void           Visit               (ScriptVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace ranally
