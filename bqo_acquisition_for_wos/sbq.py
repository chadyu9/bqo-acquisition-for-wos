import jax.numpy as jnp
import scipy.integrate
from jax.scipy.optimize import minimize


class SBQ:
    """
    A class representing a Sequential Bayesian Quadrature (SBQ) procedure without , assuming the model's integral bounds are boxed regions.
    """

    def __init__(
        self,
        kernel: callable,
        pdf: callable,
        bounds,
        X_init: jnp.array,
        Y_init: jnp.array = jnp.array([]).reshape(-1, 1),
    ):
        # Assign the kernel, pdf, and integration bounds of the SBQ model
        self.kernel = kernel
        self.pdf = pdf
        self.bounds = bounds

        # Initialize X, Y, and zs arrays
        self.X = X_init if X_init.ndim == 2 else X_init.reshape(-1, 1)
        self.Y = Y_init if Y_init.ndim == 2 else Y_init.reshape(-1, 1)
        self.zs = jnp.array([])

        # Update the zs array with the z values for the current X
        self.update_zs()

    def compute_z_value(self, x):
        """
        Compute the z value for a given x using the kernel and pdf.
        Args:
            x: The input point for which to compute the z value.
        Returns:
            The computed z value.
        """
        return scipy.integrate.nquad(
            lambda y: self.kernel(x, y) * self.pdf(y), self.bounds
        )[0]

    def update_zs(self):
        """
        Update the zs array with the z values for the current X, if not yet done.
        This is used to avoid recomputing the zs array every time.
        """
        while len(self.zs) < len(self.X):
            # Compute the z value for the current X
            z = self.compute_z_value(self.X[len(self.zs)])
            self.zs = jnp.append(self.zs, z)

    def add_Y(self, new_Y):
        """
        Update the Y array with new observations.
        Args:
            new_Y: New observations to be added to the Y array.
        """
        if new_Y.ndim == 1:
            new_Y = new_Y.reshape(-1, 1)
        self.Y = jnp.vstack((self.Y, new_Y))

    def reset_Y(self):
        """
        Reset the Y array to an empty state.
        This is useful for reinitializing the model with new observations.
        """
        self.Y = jnp.array([]).reshape(-1, 1)

    def posterior_mean(self):
        """
        Compute the posterior mean of the BQ model with current data.
        Returns:
            jnp.array: Posterior mean vector.
        """
        return self.zs.T @ jnp.linalg.solve(self.kernel(self.X, self.X), self.Y)

    def aug_posterior_variance(self, x):
        """
        Compute the posterior variance of the BQ model with current data, plus a new point x.
        Args:
            x: The new point to include in the posterior variance computation.
        Returns:
            jnp.array: Posterior variance vector.
        """
        # Add the new point x to the existing X data
        augmented_X = jnp.vstack((self.X, x))

        # Update the zs array with the new point's z value
        new_z = self.compute_z_value(x)
        augmented_zs = jnp.append(self.zs, new_z)

        # Compute the integral of the kernal matrix
        kernel_int = scipy.integrate.nquad(
            lambda y1, y2: self.kernel(y1, y2) * self.pdf(y1) * self.pdf(y2),
            [self.bounds[0] * 2, self.bounds[1] * 2],
        )[0]

        # Return the posterior variance with the new point included
        return kernel_int - augmented_zs.T @ jnp.linalg.solve(
            self.kernel(augmented_X, augmented_X), augmented_zs
        )

    def run_sbq_procedure(self, n_steps=20):
        """
        Run the SBQ procedure for a specified number of steps according to minimization of the augmented posterior variance.
        Args:
            n_steps (int): The number of steps to run the SBQ procedure.
        """

        for _ in range(n_steps):
            # Compute the minimizer of the augmented posterior variance
            next_min = minimize(
                lambda x: self.aug_posterior_variance(x),
                jnp.zeros(self.X.shape[1]),
            ).x.reshape(-1, 1)
            self.X = jnp.vstack((self.X, next_min))

            # Update the zs array with the new point's z value
            self.update_zs()
