#pragma once
#include "ranally/ast/core/syntax_vertex.h"


namespace ranally {

class SentinelVertex:
    public SyntaxVertex
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
