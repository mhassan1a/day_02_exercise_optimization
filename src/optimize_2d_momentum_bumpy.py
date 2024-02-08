"""Implement gradient descent in 2d."""

import numpy as np

from util import write_movie


def bumpy_function(pos: np.ndarray) -> np.ndarray:
    """Return values from a bumpy function.

    This bumpy functions is hard to optimize.
    It will require momentum.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The height value z.
    """
    return (
        pos[0] * pos[0]
        + pos[1] * pos[1]
        + np.cos(pos[0] * 2 * np.pi)
        + np.sin(pos[1] * 2 * np.pi)
    )


def bumpy_grad(pos: np.ndarray) -> np.ndarray:
    """Return the gradient of the bumpy function.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The gradient at position [x, y]
    """
    # TODO: Implement me!
    dz_dx = 2 * pos[0] - 2 * np.pi * np.sin(2 * np.pi * pos[0])
    dz_dy = 2 * pos[1] + 2 * np.pi * np.cos(2 * np.pi * pos[1])

    return np.array([dz_dx, dz_dy])



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    nx, ny = (1001, 1001)
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    mx, my = np.meshgrid(x, y)
    pos = np.stack((mx, my))
    mz = bumpy_function(pos)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mx, my, mz, cmap=cm.coolwarm)
    fig.colorbar(surf)

    plt.show()

    plt.contourf(mx, my, mz)
    plt.colorbar()

    start_pos = np.array((2.9, -2.9))
    step_size = 0.1  # TODO: Choose your step size.
    alpha = 0.1  # TODO: Choose your momentum term.
    step_total = 1000 # TODO: Choose the total number of steps.

    pos_list = [start_pos]
    velocity_vec = np.array((0.0, 0.0))
    # TODO implement gradient descent with momentum.
    for step in range(step_total):
        grad = bumpy_grad(pos_list[-1])
        velocity_vec = alpha * velocity_vec - step_size * grad
        pos_list.append(pos_list[-1] + velocity_vec)

    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    else:
        plt.plot(pos_list[-1][0], pos_list[-1][1], "og")
    plt.show()

    write_movie(mx, my, mz, pos_list, "writer_grad_bumpy_plot")
