#ifndef INCLUDED_RANALLY_PYTHON_BORROWEDREFERENCE
#define INCLUDED_RANALLY_PYTHON_BORROWEDREFERENCE

#include <Python.h>



namespace ranally {
namespace python {

//! Class for holding a borrowed reference to a Python object.
/*!
  This class exists mainly for documentation purposes, so the user knows the
  layered object is borrowed, not owned.

  \sa        OwnedReference
*/
class BorrowedReference
{

  friend class BorrowedReferenceTest;

public:

                   BorrowedReference   ();

                   BorrowedReference   (PyObject* object);

                   BorrowedReference   (BorrowedReference const& other);

                   ~BorrowedReference  ();

  BorrowedReference& operator=         (PyObject* object);

                   operator bool       () const;

  PyObject**       operator&           ();

                   operator PyObject*  ();

  template<class T>
                   operator T*         ();

private:

  PyObject*        _object;

};



//! Return the layered Python object cast to \a T*
/*!
  \tparam    T Type to cast the layered Python object to.
  \return    Python object cast to T*.
  \warning   A Python object must be layered in the instance.
*/
template<class T>
inline BorrowedReference::operator T*()
{
  assert(_object);
  return (T*)_object;
}

} // namespace python
} // namespace ranally

#endif
