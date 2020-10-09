import numpy as mat


def NumGrad(function, theta, parameters=()):
    """
    In this method we have implemented a code for finding out the Gradient by Numerical approach so that we can
    compare the values of gradient solved by Numerical approach and Analytical approach and debug our gradient code
    if there is any discrepancy in both set of values.

    what is Numerical and Analytical approach? --if we know calculus we can easily differentiate both the approaches.
    As we know most of the time the way we find derivatives of any function is by using already cooked up formulas
    for ex- d(x^n)/dx = nx^n-1, or d(sin(x))/dx = cos(x), this are one of the formulas which we get by using the
    original definition of derivative and  finding derivative of our function using these formulas known as
    Analytical approach. When instead of using cooked up formulas we directly put our function in original definition
    of derivative and solve it that process is known as Numerical approach.

    Now the original formula of derivative which we get by geometric interpretation is some time known as forward
    difference since the Moving point  we take is on the right side of 'X' and when we take the moving point on the
    left side and calculate the derivative that is known as Backward difference. But both these derivatives give the
    error 'dx' in big O notation O(dx) but if we take the central difference that means we take 'dx' on both side the
    error we get in O notation is O(dx^2) that means if are taking 'dx' = 10^-3 our error will be 10^-6 which is much
    lesser as compare to other two numerical approach.

    central difference  = (f(x+dx) - f(x-dx)) / 2dx

    :param function: is the function the user want to find a derivative of Numerically.
    :param theta: is the input parameter user want to calculate.
    :param parameters: are the rest of the arguments which function will accept.
    :return: Numerically Calculated derivatives of given function.
    """

    # Now we are going to calculate derivative of i-th X value with respect to dx

    numerical_grad = mat.zeros(theta.shape)
    ithVector = mat.zeros(theta.shape)
    dx = pow(10, -4)

    for i in range(len(theta)):
        ithVector[i] = dx

        forward_val = function(mat.add(theta, ithVector), *parameters)
        backward_val = function(mat.subtract(theta, ithVector), *parameters)

        # compute central difference
        numerical_grad[i] = mat.divide(mat.subtract(forward_val, backward_val), (2 * dx))
        ithVector[i] = 0

    return numerical_grad
