// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/visitor/dot_visitor.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FlowgraphDotVisitor:
    public DotVisitor
{

    friend class FlowgraphDotVisitorTest;

public:

                   FlowgraphDotVisitor ();

                   ~FlowgraphDotVisitor()=default;

                   FlowgraphDotVisitor (FlowgraphDotVisitor&&)=delete;

    FlowgraphDotVisitor& operator=     (FlowgraphDotVisitor&&)=delete;

                   FlowgraphDotVisitor (FlowgraphDotVisitor const&)=delete;

    FlowgraphDotVisitor& operator=     (FlowgraphDotVisitor const&)=delete;

private:

    enum Mode {
        Declaring=0x1,
        ConnectingFlowgraph=0x2
        /// ConnectingOperationArgument=0x4
    };

    //! Current mode.
    Mode           _mode;

    void           set_mode            (Mode mode);

    void           add_flowgraph_vertex(NameVertex const& source_vertex,
                                        AstVertex const& target_vertex);

    void           add_flowgraph_vertex(AstVertex const& source_vertex,
                                        AstVertex const& target_vertex);

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (FunctionCallVertex& vertex);

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

    void           Visit               (ModuleVertex& vertex);

    void           Visit               (StringVertex& vertex);

    void           Visit               (SubscriptVertex& vertex);

    void           Visit               (WhileVertex& vertex);

};

} // namespace fern
