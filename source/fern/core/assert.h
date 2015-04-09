// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


#define FERN_STATIC_ASSERT( \
        trait, \
        ...) \
    static_assert(trait<__VA_ARGS__>::value, "Assuming " #trait);
