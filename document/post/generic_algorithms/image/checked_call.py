import functools
import sys
import traceback


def checked_call(
        function):
    """
    Execute *function*. In case an exception is thrown, the traceback is
    written to sys.stderr, and 1 is returned. If no exception is thrown, the
    *function*'s result is returned, or 0 if *function* didn't return anything.

    This function is useful when creating a commandline application.
    """
    @functools.wraps(function)
    def wrapper(
            *args,
            **kwargs):
        result = 0
        try:
            result = function(*args, **kwargs)
        except:
            traceback.print_exc(file=sys.stderr)
            result = 1
        return 0 if result is None else result
    return wrapper
