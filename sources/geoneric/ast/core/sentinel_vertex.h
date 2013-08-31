#pragma once
#include "ranally/ast/core/ast_vertex.h"


namespace ranally {

class SentinelVertex:
    public AstVertex
{

public:

                   SentinelVertex      ();

                   ~SentinelVertex     ()=default;

                   SentinelVertex      (SentinelVertex&&)=delete;

    SentinelVertex& operator=          (SentinelVertex&&)=delete;

                   SentinelVertex      (SentinelVertex const&)=delete;

    SentinelVertex& operator=          (SentinelVertex const&)=delete;

};

} // namespace ranally
