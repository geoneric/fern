#pragma once


#define FERN_STATIC_ASSERT( \
        trait, \
        ...) \
    static_assert(trait<__VA_ARGS__>::value, "Assuming " #trait);
