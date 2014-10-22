#pragma once


#define UNARY_FUNCTION_OVERLOAD(              \
    algorithm,                                \
    type,                                     \
    value_type)                               \
PyArrayObject* algorithm##_##value_type(      \
    type value)                               \
{                                             \
    return algorithm<value_type##_t>(value);  \
}


#define BINARY_FUNCTION_OVERLOAD(                                        \
    algorithm,                                                           \
    type1,                                                               \
    value_type1,                                                         \
    type2,                                                               \
    value_type2)                                                         \
PyArrayObject* algorithm##_##value_type1##_##value_type2(                \
    type1 value1,                                                        \
    type2 value2)                                                        \
{                                                                        \
    return algorithm<value_type1##_t, value_type2##_t>(value1, value2);  \
}


#define PARSE_UNARY_ARGUMENTS                           \
PyObject* value_object;                                 \
                                                        \
if(!PyArg_ParseTuple(arguments, "O", &value_object)) {  \
    return nullptr;                                     \
}


#define PARSE_BINARY_ARGUMENTS                                           \
PyObject* value1_object;                                                 \
PyObject* value2_object;                                                 \
                                                                         \
if(!PyArg_ParseTuple(arguments, "OO", &value1_object, &value2_object)) { \
    return nullptr;                                                      \
}


#define TRY_UNARY_ALGORITHM(                                              \
    algorithm)                                                            \
PyObject* result{nullptr};                                                \
                                                                          \
try {                                                                     \
    UnaryAlgorithmKey key(data_type(value_object));                       \
                                                                          \
    if(algorithm##_overloads.find(key) == algorithm##_overloads.end()) {  \
        raise_unsupported_overload_exception(value_object);               \
        result = nullptr;                                                 \
    }                                                                     \
    else {                                                                \
        result = algorithm##_overloads[key](value_object);                \
    }                                                                     \
                                                                          \
    assert((PyErr_Occurred() && result == nullptr) ||                     \
        (!PyErr_Occurred() && result != nullptr));                        \
}                                                                         \


#define TRY_BINARY_ALGORITHM(                                                \
    algorithm)                                                               \
PyObject* result{nullptr};                                                   \
                                                                             \
try {                                                                        \
    BinaryAlgorithmKey key(data_type(value1_object),                         \
        data_type(value2_object));                                           \
                                                                             \
    if(algorithm##_overloads.find(key) == algorithm##_overloads.end()) {     \
        raise_unsupported_overload_exception(value1_object, value2_object);  \
        result = nullptr;                                                    \
    }                                                                        \
    else {                                                                   \
        result = algorithm##_overloads[key](value1_object, value2_object);   \
    }                                                                        \
                                                                             \
    assert((PyErr_Occurred() && result == nullptr) ||                        \
        (!PyErr_Occurred() && result != nullptr));                           \
}                                                                            \


#define CATCH_AND_RETURN                                     \
catch(std::runtime_error const& exception) {                 \
    PyErr_SetString(PyExc_RuntimeError, exception.what());   \
    assert(result == nullptr);                               \
}                                                            \
catch(std::exception const& exception) {                     \
    PyErr_SetString(PyExc_StandardError, exception.what());  \
    assert(result == nullptr);                               \
}                                                            \
                                                             \
assert((PyErr_Occurred() && result == nullptr) ||            \
    (!PyErr_Occurred() && result != nullptr));               \
assert(result != Py_None);                                   \
return result;


#define UNARY_ALGORITHM(            \
    algorithm)                      \
PyObject* sqrt(                     \
    PyObject* /* self */,           \
    PyObject* arguments)            \
{                                   \
    PARSE_UNARY_ARGUMENTS           \
    TRY_UNARY_ALGORITHM(algorithm)  \
    CATCH_AND_RETURN                \
}


#define BINARY_ALGORITHM(            \
    algorithm)                       \
PyObject* add(                       \
    PyObject* /* self */,            \
    PyObject* arguments)             \
{                                    \
    PARSE_BINARY_ARGUMENTS           \
    TRY_BINARY_ALGORITHM(algorithm)  \
    CATCH_AND_RETURN                 \
}
