#pragma once
#include <loki/Visitor.h>
#include "ranally/ast/core/ast_vertex.h"


namespace ranally {

class AssignmentVertex;
class FunctionDefinitionVertex;
class FunctionCallVertex;
class IfVertex;
class ModuleVertex;
class NameVertex;
template<typename T>
    class NumberVertex;
class OperationVertex;
class OperatorVertex;
class ReturnVertex;
class ScopeVertex;
class SentinelVertex;
class StringVertex;
class SubscriptVertex;
class WhileVertex;


//! Base class for syntax tree visitors.
/*!
  This class offers default implementations for the visit functions for all
  AstVertex specializations.
*/
class Visitor:
    public Loki::BaseVisitor,
    public Loki::Visitor<AssignmentVertex>,
    public Loki::Visitor<FunctionDefinitionVertex>,
    public Loki::Visitor<FunctionCallVertex>,
    public Loki::Visitor<IfVertex>,
    public Loki::Visitor<NameVertex>,
    public Loki::Visitor<NumberVertex<int8_t>>,
    public Loki::Visitor<NumberVertex<int16_t>>,
    public Loki::Visitor<NumberVertex<int32_t>>,
    public Loki::Visitor<NumberVertex<int64_t>>,
    public Loki::Visitor<NumberVertex<uint8_t>>,
    public Loki::Visitor<NumberVertex<uint16_t>>,
    public Loki::Visitor<NumberVertex<uint32_t>>,
    public Loki::Visitor<NumberVertex<uint64_t>>,
    public Loki::Visitor<NumberVertex<float>>,
    public Loki::Visitor<NumberVertex<double>>,
    public Loki::Visitor<OperatorVertex>,
    public Loki::Visitor<ReturnVertex>,
    public Loki::Visitor<ScopeVertex>,
    public Loki::Visitor<ModuleVertex>,
    public Loki::Visitor<SentinelVertex>,
    public Loki::Visitor<StringVertex>,
    public Loki::Visitor<SubscriptVertex>,
    public Loki::Visitor<WhileVertex>
{

    friend class VisitorTest;

public:

protected:

                   Visitor             ()=default;

    virtual        ~Visitor            ()=default;

                   Visitor             (Visitor&&)=delete;

    Visitor&       operator=           (Visitor&&)=delete;

                   Visitor             (Visitor const&)=delete;

    Visitor&       operator=           (Visitor const&)=delete;

    virtual void   visit_statements    (StatementVertices& statements);

    virtual void   visit_expressions   (ExpressionVertices const& expressions);

    virtual void   Visit               (AssignmentVertex& vertex);

    virtual void   Visit               (IfVertex& vertex);

    virtual void   Visit               (FunctionDefinitionVertex& vertex);

    virtual void   Visit               (FunctionCallVertex& vertex);

    virtual void   Visit               (OperationVertex& vertex);

    virtual void   Visit               (ReturnVertex& vertex);

    virtual void   Visit               (ModuleVertex& vertex);

    virtual void   Visit               (SubscriptVertex& vertex);

    virtual void   Visit               (WhileVertex& vertex);

private:

    virtual void   Visit               (ExpressionVertex& vertex);

    virtual void   Visit               (NameVertex& vertex);

    virtual void   Visit               (NumberVertex<int8_t>& vertex);

    virtual void   Visit               (NumberVertex<int16_t>& vertex);

    virtual void   Visit               (NumberVertex<int32_t>& vertex);

    virtual void   Visit               (NumberVertex<int64_t>& vertex);

    virtual void   Visit               (NumberVertex<uint8_t>& vertex);

    virtual void   Visit               (NumberVertex<uint16_t>& vertex);

    virtual void   Visit               (NumberVertex<uint32_t>& vertex);

    virtual void   Visit               (NumberVertex<uint64_t>& vertex);

    virtual void   Visit               (NumberVertex<float>& vertex);

    virtual void   Visit               (NumberVertex<double>& vertex);

    virtual void   Visit               (OperatorVertex& vertex);

    virtual void   Visit               (ScopeVertex& vertex);

    virtual void   Visit               (SentinelVertex& vertex);

    virtual void   Visit               (StatementVertex& vertex);

    virtual void   Visit               (StringVertex& vertex);

    virtual void   Visit               (AstVertex& vertex);

};


//! Macro that will call the macro passed in for each numeric value type.
/*!
*/
#define VISIT_NUMBER_VERTICES(                                                 \
        macro)                                                                 \
    macro(int8_t)                                                              \
    macro(int16_t)                                                             \
    macro(int32_t)                                                             \
    macro(int64_t)                                                             \
    macro(uint8_t)                                                             \
    macro(uint16_t)                                                            \
    macro(uint32_t)                                                            \
    macro(uint64_t)                                                            \
    macro(float)                                                               \
    macro(double)

} // namespace ranally
