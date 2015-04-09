// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
// VT_BOOL and VT_STRING are not exposed in Python, so they shouldn't
// be passed in.


//     case_(VT_INT8, int8_t)      
//     case_(VT_UINT16, uint16_t)  
//     case_(VT_INT16, int16_t)    

#define SWITCH_ON_VALUE_TYPE(   \
    value_type,                 \
    case_)                      \
switch(value_type) {            \
    case_(VT_UINT8, uint8_t)    \
    case_(VT_UINT32, uint32_t)  \
    case_(VT_INT32, int32_t)    \
    case_(VT_UINT64, uint64_t)  \
    case_(VT_INT64, int64_t)    \
    case_(VT_FLOAT32, float)    \
    case_(VT_FLOAT64, double)   \
    case VT_BOOL:               \
    case VT_INT8:               \
    case VT_UINT16:             \
    case VT_INT16:              \
    case VT_STRING: {           \
        assert(false);          \
    }                           \
}


//     case_(VT_INT8, int8_t, __VA_ARGS__)      
//     case_(VT_UINT16, uint16_t, __VA_ARGS__)  
//     case_(VT_INT16, int16_t, __VA_ARGS__)    

#define SWITCH_ON_VALUE_TYPE1(               \
    value_type,                              \
    case_,                                   \
    ...)                                     \
switch(value_type) {                         \
    case_(VT_UINT8, uint8_t, __VA_ARGS__)    \
    case_(VT_UINT32, uint32_t, __VA_ARGS__)  \
    case_(VT_INT32, int32_t, __VA_ARGS__)    \
    case_(VT_UINT64, uint64_t, __VA_ARGS__)  \
    case_(VT_INT64, int64_t, __VA_ARGS__)    \
    case_(VT_FLOAT32, float, __VA_ARGS__)    \
    case_(VT_FLOAT64, double, __VA_ARGS__)   \
    case VT_BOOL:                            \
    case VT_INT8:               \
    case VT_UINT16:             \
    case VT_INT16:              \
    case VT_STRING: {                        \
        assert(false);                       \
    }                                        \
}


//     case_(VT_INT8, int8_t, __VA_ARGS__)      
//     case_(VT_UINT16, uint16_t, __VA_ARGS__)  
//     case_(VT_INT16, int16_t, __VA_ARGS__)    

#define SWITCH_ON_VALUE_TYPE2(               \
    value_type,                              \
    case_,                                   \
    ...)                                     \
switch(value_type) {                         \
    case_(VT_UINT8, uint8_t, __VA_ARGS__)    \
    case_(VT_UINT32, uint32_t, __VA_ARGS__)  \
    case_(VT_INT32, int32_t, __VA_ARGS__)    \
    case_(VT_UINT64, uint64_t, __VA_ARGS__)  \
    case_(VT_INT64, int64_t, __VA_ARGS__)    \
    case_(VT_FLOAT32, float, __VA_ARGS__)    \
    case_(VT_FLOAT64, double, __VA_ARGS__)   \
    case VT_BOOL:                            \
    case VT_INT8:               \
    case VT_UINT16:             \
    case VT_INT16:              \
    case VT_STRING: {                        \
        assert(false);                       \
    }                                        \
}


//     case_(VT_INT8, int8_t, __VA_ARGS__)      
//     case_(VT_UINT16, uint16_t, __VA_ARGS__)  
//     case_(VT_INT16, int16_t, __VA_ARGS__)    

#define SWITCH_ON_VALUE_TYPE3(               \
    value_type,                              \
    case_,                                   \
    ...)                                     \
switch(value_type) {                         \
    case_(VT_UINT8, uint8_t, __VA_ARGS__)    \
    case_(VT_UINT32, uint32_t, __VA_ARGS__)  \
    case_(VT_INT32, int32_t, __VA_ARGS__)    \
    case_(VT_UINT64, uint64_t, __VA_ARGS__)  \
    case_(VT_INT64, int64_t, __VA_ARGS__)    \
    case_(VT_FLOAT32, float, __VA_ARGS__)    \
    case_(VT_FLOAT64, double, __VA_ARGS__)   \
    case VT_BOOL:                            \
    case VT_INT8:               \
    case VT_UINT16:             \
    case VT_INT16:              \
    case VT_STRING: {                        \
        assert(false);                       \
    }                                        \
}


#define SWITCH_ON_FLOATING_POINT_VALUE_TYPE( \
    value_type,                              \
    case_)                                   \
switch(value_type) {                         \
    case_(VT_FLOAT32, float)                 \
    case_(VT_FLOAT64, double)                \
    default: {                               \
        assert(false);                       \
    }                                        \
}
