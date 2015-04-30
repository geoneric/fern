// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/python/borrowed_reference.h"


namespace fern {
namespace python {

//! Default construct an instance.
/*!
  The layered object is initialized to NULL.
*/
BorrowedReference::BorrowedReference()

    : _object(NULL)

{
}


//! Construct an instance based on \a object passed in.
/*!
  \param     object Object to layer in the instance.
*/
BorrowedReference::BorrowedReference(
    PyObject* object)

    : _object(object)

{
}


// //! Copy construct an instance based on \a other passed in.
// /*!
//   \param     other Instance to copy.
// 
//   The reference count of the previous and new layered objects are not changed.
// */
// BorrowedReference::BorrowedReference(
//     BorrowedReference const& other)
//
//     : _object(other._object)
// 
// {
// }


// BorrowedReference& BorrowedReference::operator=(
//     PyObject* object)
// {
// 
//     // Discard of the current _object. We borrowed it.
//     _object = object;
//     return *this;
// }


// //! Return whether the layered object is not NULL.
// /*!
//   \return    true or false
// */
// BorrowedReference::operator bool() const
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
PyObject** BorrowedReference::operator&()
{
    assert(_object == NULL);
    return &_object;
}


//! Return the layered Python object.
/*!
  \return    The layered Python object or NULL if no object is set.
*/
BorrowedReference::operator PyObject*()
{
    return _object;
}

} // namespace python
} // namespace fern
