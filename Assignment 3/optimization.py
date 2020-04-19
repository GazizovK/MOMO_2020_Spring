from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
import numpy as np
import time
from utils import get_line_search_tool


def make_history(history, labels, values):
    if history is not None:
        for label, value in zip(labels, values):
            if label == 'x' and value.size > 2:
                continue
            history[label].append(value)


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    labels = ['time', 'residual_norm', 'x']
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.

    if max_iter is None:
        max_iter = b.size

    start = time.time()

    g = matvec(x_0) - b
    x_k = np.copy(x_0)
    d = -g
    b_norm = np.linalg.norm(b)
    msg = 'success'
    i = 0
    while np.linalg.norm(g) > tolerance * b_norm:

        if i > max_iter:
            msg = 'iterations exceed'
            if display:
                print(msg)
            return x_k, msg, history

        hist_values = [time.time() - start, np.linalg.norm(g), x_k]
        make_history(history, labels, hist_values)

        g_sq = g.dot(g)
        x_k = x_k + g_sq / matvec(d).dot(d) * d
        g = matvec(x_k) - b
        d = -g + g.dot(g) / g_sq * d

        i += 1

    hist_values = [time.time() - start, np.linalg.norm(g), x_k]
    make_history(history, labels, hist_values)

    if display:
        print(msg)
    return x_k, msg, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    labels = ['func', 'time', 'grad_norm', 'x']
    line_search_tool = get_line_search_tool(line_search_options)
    H = deque()
    x_k = np.copy(x_0)
    i = 0
    l = 10

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.

    def bfgs_multiply(v, H, gamma):
        if H:
            H1 = H.copy()
            s, y = H1.pop()
            v1 = v - s.dot(v) / y.dot(s) * y
            z = bfgs_multiply(v1, H1, gamma)
            return z + (s.dot(v) - y.dot(z)) / y.dot(s) * s
        else:
            return gamma * v

    def lbfgs_direction(grad, H):
        if H:
            s, y = H[-1]
            gamma = y.dot(s) / y.dot(y)
            return bfgs_multiply(-grad, H, gamma)
        else:
            return -grad

    start = time.time()
    grad_0 = oracle.grad(x_k)
    grad_k = np.copy(grad_0)
    grad_k_prev = None
    x_k_prev = None
    msg = 'success'
    while grad_k.dot(grad_k) > tolerance * grad_0.dot(grad_0):

        if i > max_iter:
            msg = 'iterations exceed'
            if display:
                print(msg)
            return x_k, msg, history

        hist_values = [oracle.func(x_k), time.time() - start, np.linalg.norm(grad_k), x_k]
        make_history(history, labels, hist_values)

        if x_k_prev is not None:
            H.append((x_k - x_k_prev, grad_k - grad_k_prev))
            if len(H) > l:
                H.popleft()

        d_k = lbfgs_direction(grad_k, H)

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k_prev = np.copy(x_k)
        grad_k_prev = np.copy(grad_k)
        x_k = x_k + alpha_k * d_k
        grad_k = oracle.grad(x_k)

        i += 1

    hist_values = [oracle.func(x_k), time.time() - start, np.linalg.norm(grad_k), x_k]
    make_history(history, labels, hist_values)

    if display:
        print(msg)

    return x_k, msg, history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    labels = ['func', 'time', 'grad_norm', 'x']
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    i = 0

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    start = time.time()
    grad_0 = oracle.grad(x_k)
    grad_k = np.copy(grad_0)
    msg = 'success'
    while grad_k.dot(grad_k) > tolerance * grad_0.dot(grad_0):

        if i > max_iter:
            msg = 'iterations exceed'
            if display:
                print(msg)
            return x_k, msg, history

        hist_values = [oracle.func(x_k), time.time() - start, np.linalg.norm(grad_k), x_k]
        make_history(history, labels, hist_values)

        matvec = lambda v: oracle.hess_vec(x_k, v)
        mu_k = np.min([0.5, np.sqrt(np.linalg.norm(grad_k))])
        while True:
            d_k, _, _ = conjugate_gradients(matvec, -grad_k, -grad_k, mu_k)
            if grad_k.dot(d_k) <= 0:
                break
            mu_k = mu_k / 10.

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha_k * d_k
        grad_k = oracle.grad(x_k)

        i += 1

    hist_values = [oracle.func(x_k), time.time() - start, np.linalg.norm(grad_k), x_k]
    make_history(history, labels, hist_values)

    if display:
        print(msg)
    return x_k, msg, history
