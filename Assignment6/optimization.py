from collections import defaultdict
import numpy as np
from time import time


def make_history(history, labels, values):
    if history is not None:
        for label, value in zip(labels, values):
            if label == 'x' and value.size > 2:
                continue
            history[label].append(value)


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
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
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    labels = ['func', 'time', 'duality_gap', 'x']
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    dual_gap_k = oracle.duality_gap(x_k)
    min_x = np.copy(x_k)
    min_f = np.copy(f_k)
    k = 0
    msg = 'success'
    start = time()

    while dual_gap_k > tolerance:
        if k > max_iter:
            msg = 'iterations_exceeded'
            if display:
                print(msg)
            return x_k, msg, history

        hist_values = [f_k, time() - start, dual_gap_k, x_k]
        make_history(history, labels, hist_values)

        sub_grad_k = oracle.subgrad(x_k)
        sub_grad_k = sub_grad_k / np.linalg.norm(sub_grad_k)

        alpha_k = alpha_0 / np.sqrt(k + 1)
        x_k -= sub_grad_k * alpha_k

        f_k = oracle.func(x_k)
        dual_gap_k = oracle.duality_gap(x_k)

        if f_k < min_f:
            min_f = np.copy(f_k)
            min_x = np.copy(x_k)

        k += 1

    hist_values = [f_k, time() - start, dual_gap_k, x_k]
    make_history(history, labels, hist_values)

    if display:
        print(msg)
    return min_x, msg, history


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    labels = ['func', 'time', 'duality_gap', 'x', 'search_counter']

    x_k = np.copy(x_0)
    L_k = L_0
    f_k = oracle._f.func(x_k)
    grad_k = oracle.grad(x_k)
    dual_gap_k = oracle.duality_gap(x_k)
    k = 0
    msg = 'success'
    start = time()

    search_counter = 0
    while dual_gap_k >= tolerance:

        hist_values = [oracle.func(x_k), time() - start, dual_gap_k, x_k, search_counter]
        make_history(history, labels, hist_values)

        if k >= max_iter:
            msg = 'iterations_exceeded'
            if display:
                print(msg)
            return x_k, msg, history
        search_counter = 0
        while True:
            search_counter += 1
            y = oracle.prox(x_k - 1.0 / L_k * grad_k, 1.0 / L_k)
            z = f_k + grad_k.dot(y - x_k) + L_k / 2 * (y - x_k).dot(y - x_k) + oracle._h.func(y)
            if oracle.func(y) <= z:
                break
            L_k *= 2

        x_k = np.copy(y)
        L_k = max(L_0, L_k / 2)

        f_k = oracle._f.func(x_k)
        grad_k = oracle.grad(x_k)
        dual_gap_k = oracle.duality_gap(x_k)

        k += 1

    hist_values = [oracle.func(x_k), time() - start, dual_gap_k, x_k, search_counter]
    make_history(history, labels, hist_values)

    if display:
        print(msg)
    return x_k, msg, history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps for best point on every step of the algorithm
    """
    # TODO: Implement
    history = defaultdict(list) if trace else None
    labels = ['func', 'time', 'duality_gap', 'search_counter']

    A_k = 0.0
    a_k = None
    A_k_next = None
    v_k = np.copy(x_0)
    x_k = np.copy(x_0)
    L_k = L_0
    phi_k = oracle.func(x_k)
    dual_gap_k = oracle.duality_gap(x_k)
    k = 0
    msg = 'success'
    start = time()
    M_k = 0.0
    M_prev = 0.0
    min_x = np.copy(x_k)
    min_phi = np.copy(phi_k)
    a_sum = 0.0
    prev_asum = 0.0
    search_counter = 0

    while dual_gap_k > tolerance:

        hist_values = [min_phi, time() - start, dual_gap_k, search_counter]
        make_history(history, labels, hist_values)
#        print('x_k', x_k, 'iter', k, 'f', oracle.func(x_k), 'min', min_phi, 'minx', min_x)
        if k >= max_iter:
            msg = 'iterations_exceeded'
            if display:
                print(msg)
            return x_k, msg, history

        search_counter = 0
        while True:
            search_counter += 1
            a_k = (1 + np.sqrt(1 + 4 * L_k * A_k)) / (2 * L_k)
            A_k_next = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_k_next

            M_k = M_prev + a_k * oracle.grad(y_k)
            a_sum = prev_asum + a_k
            v_k_next = oracle.prox(x_0 - M_k, a_sum)
            x_k_next = (A_k * x_k + a_k * v_k_next) / A_k_next

            z = oracle._f.func(y_k) + oracle.grad(y_k).dot(x_k_next - y_k) + \
                L_k / 2 * (x_k_next - y_k).dot(x_k_next - y_k)
            if oracle._f.func(x_k_next) <= z:
                break
            L_k = 2 * L_k

            phi_k = oracle.func(x_k_next)
            if phi_k < min_phi:
                min_phi = np.copy(phi_k)
                min_x = np.copy(x_k_next)

            fy_k = oracle.func(y_k)
            if fy_k < min_phi:
                min_phi = np.copy(fy_k)
                min_x = np.copy(y_k)

        L_k = L_k / 2

        x_k = x_k_next
        v_k = v_k_next
        A_k = A_k_next
        prev_asum = a_sum
        M_prev = M_k

        phi_k = oracle.func(x_k)
        if phi_k < min_phi:
            min_phi = np.copy(phi_k)
            min_x = np.copy(x_k)

        fy_k = oracle.func(y_k)
        if fy_k < min_phi:
            min_phi = np.copy(fy_k)
            min_x = np.copy(y_k)
        dual_gap_k = oracle.duality_gap(x_k)

        #print('y', oracle.func(y_k))
        k += 1

    hist_values = [min_phi, time() - start, dual_gap_k, search_counter]
    make_history(history, labels, hist_values)

    if display:
        print(msg)
    return min_x, msg, history
