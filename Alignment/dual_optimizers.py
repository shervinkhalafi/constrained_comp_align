import numpy as np
class NuPIController:
    def __init__(self, theta_0,xi_0, nu=0, kappa_p=5, kappa_i=2):
        """
        Initializes the ν PI Controller.

        Args:
            nu (float): EMA of PI/error coefficient (0 < nu <= 1). 0 means no memory.
            kappa_p (float or np.ndarray): Proportional gain. Controls damping/stability/overshoot. >1 recomended. 5-20 is a good range.
            kappa_i (float or np.ndarray): Integral gain. Controls convergence speed. GA learning rate step ranges usually work (1-5).
            xi_0 (float or np.ndarray): Initial EMA of error. Setting it to the constrain level is recommended.
            theta_0 (float or np.ndarray): Initial parameter vector.
        """
        self.nu = nu
        self.kappa_p = kappa_p
        self.kappa_i = kappa_i
        self.xi_prev = xi_0
        self.theta_prev = theta_0
        self.theta = theta_0
        self.t = 0  # Time step

    def update(self, e_t):
        """
        Performs a single update step of the ν PI Controller.

        Args:
            e_t (float or np.ndarray): Current constraint violation.

        Returns:
            theta_new (float or np.ndarray): Updated parameter vector.
        """
        # Update EMA of error
        xi_t = self.nu * self.xi_prev + (1 - self.nu) * e_t

        if self.t == 0:
            # Initial update
            theta_new = self.theta_prev + self.kappa_i * e_t + self.kappa_p * self.xi_prev
        else:
            # Recursive update
            theta_new = self.theta_prev + self.kappa_i * e_t + self.kappa_p * (xi_t - self.xi_prev)
        
        # Clip theta to be non-negative
        theta_new = np.clip(theta_new, 0, None)

        # Update state for next iteration
        self.xi_prev = xi_t
        self.theta_prev = theta_new
        self.theta = theta_new
        self.t += 1

        return self.theta