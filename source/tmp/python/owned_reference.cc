#include "fern/python/owned_reference.h"


namespace fern {
namespace python {

//! Default construct an instance.
/*!
  The layered object is initialized to NULL.
*/
OwnedReference::OwnedReference()

    : _object(NULL)

{
}


//! Construct an instance based on \a object passed in.
/*!
  \param     object Object to layer in the instance.

  This constructor assumeѕ the reference count is already incremented, as all
  generic Python operations do (functions whose name begins with PyObject_,
  PyNumber_, PySequence_ or PyMapping_).

  In case you are passing in an object whose reference count doesn't reflect
  that the reference is owned by us, then call inc_ref() after constructing the
  instance.
*/
OwnedReference::OwnedReference(
    PyObject* object)

    : _object(object)

{
}


// //! Copy construct an instance based on \a other passed in.
// /*!
//   \param     other Instance to copy.
// 
//   The reference count of the new layered object is incremented.
// */
// OwnedReference::OwnedReference(
//     OwnedReference const& other)
// 
//     : _object(other._object)
// 
// {
//     assert(false);
//     if(_object != NULL) {
//         inc_ref();
//     }
// }


//! Assign \a object to the instance.
/*!
  \param     object Object to layer in the instance.
  \return    Reference to the instance.

  Nothing happens in case \a object is already layered in the instance.
  Otherwise the reference count of the layered reference is decremented before
  assigning the new reference. The reference count of the new reference is not
  changed.
*/
OwnedReference& OwnedReference::operator=(
    PyObject* object)
{
    if(_object == object) {
        // inc_ref();
    }
    else {
        if(_object != NULL) {
            dec_ref();
        }

        _object = object;
    }

    return *this;
}


//! Destruct instance.
/*!
  The reference count of the layered object is decremented.
*/
OwnedReference::~OwnedReference()
{
    if(_object != NULL) {
        dec_ref();
    }
}


// OwnedReference::operator bool() const
// {
//     return _object != NULL;
// }


//! Return the address of the (pointer to) the layered object.
/*!
  \return    Address of the address of the layered object.

  Using this address, a new pointer to a Python object can be stored in this
  instance. It is assumed that no Python object is layered in the instance
  yet.
*/
PyObject** OwnedReference::operator&()
{
    assert(_object == NULL);
    return &_object;
}


//! Return the layered Python object.
/*!
  \return    The layered Python object or NULL if no object is set.
*/
OwnedReference::operator PyObject*()
{
    return _object;
}


void OwnedReference::inc_ref()
{
    assert(_object != NULL);
    Py_INCREF(_object);
}


void OwnedReference::dec_ref()
{
    assert(_object != NULL);
    Py_DECREF(_object);
}

} // namespace python
} // namespace fern
