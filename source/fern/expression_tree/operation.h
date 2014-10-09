#pragma once


namespace fern {

struct LocalOperation
{
};


struct Add: LocalOperation
{
    Add()
        : name("add")
    {
    }

    std::string name;
};


struct NeighborhoodOperation
{
};


struct GlobalOperation
{
};

} // namespace fern
