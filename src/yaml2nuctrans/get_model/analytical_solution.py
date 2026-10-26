import numpy as np
from astropy import units as u
from astropy import constants as const
import radioactivedecay as rd
from scipy import special
from decimal import *


class Analytical_Sol:
    """
    Class for calculating the analytical solution of 1D coupled decay, diffusion,
    advection and sorption process model. The
    """

    @staticmethod
    def full_coupled_single_1d_steady_state(c_init, k, R, D, v, x):
        """
        steady-state solution of single specie transport equation when t and t0 go to infinity.

        Args:
            c_init (float): initial condition.
            k (float): decay rate.
            R (float): retardation factor.
            D (float): diffusion coefficient.
            v (float): pore velocity.
            x (np.ndarry): an array of distance.

        Returns:
            c (np.ndarry): concentration of nuclide over distance at the steady-state.

        """
        mu = k * R
        u = (v**2 + 4 * mu * D) ** 0.5

        c = c_init * np.exp((v - u) * x / (2 * D))
        return c

    @staticmethod
    def auxiliary_SumProduct(i, concentrations, half_lifes):
        """
        calculate the auxiliary variables for converting the single specie transport
        to multiple species transport model.

        Args:
            i (int): nuclide index.
            concentrations (list): a list of concentrations for nuclides.
            half_lifes (np.ndarray): half_life for nuclides.

        Returns:
            float: auxiliary variable.
        """
        sum = 0
        for j in range(i):
            product = 1
            for l in range(j, i):
                product *= half_lifes[l] / (half_lifes[l] - half_lifes[i])
            sum += product * concentrations[j]
        return sum

    def full_coupled_single_1d(
        self,
        c_init,
        k,
        R,
        D,
        v,
        c_bc_le_t_interim,
        t_interim,
        x,
        t,
        num_approx,
        scaling,
    ):
        """
        calculate the analytical solution for a single specie transport process.

        Args:
            c_init (float): initial condition.
            k (float): decay rate.
            R (float): retardation factor.
            D (float): diffusion coefficient.
            v (float): pore velocity.
            c_bc_le_t_interim (float): boundary condition.
            t_interim (float): when time is larger than t_interim, the boundary condition
            will set to be zero.
            x (np.ndarry): an array of distance.
            t (float): the final time.
            num_approx (bool): use an approximation for the multiplication between
            exponential and complementary error fuctions.
            scaling (bool): non-dimensionalising the equation.

        Returns:
            np.ndarry: concentration of nuclide over distance at a given time.
        """

        if t == 0:
            return np.ones(len(np.array(x))) * c_init

        mu = k * R
        if scaling:
            L = np.max(x)
            T = t

            diffusion_term = 1.0
            advection_term = v * L / D
            dc_dt_term = R * L * L / (D * T)
            decay_term = k * R * L * L / D

            t_interim = t_interim / T
            t = t / T
            x = x / L

            D = diffusion_term
            v = advection_term
            R = dc_dt_term
            mu = decay_term

        c_sol_1 = c_bc_le_t_interim * self.full_coupled_single_1d_H(
            mu, R, D, v, x, t, num_approx=num_approx
        ) + c_init * self.full_coupled_single_1d_M(
            mu, R, D, v, x, t, num_approx=num_approx
        )

        if t <= t_interim:
            c_sol = c_sol_1

        if t > t_interim:
            c_sol = c_sol_1 - c_bc_le_t_interim * self.full_coupled_single_1d_H(
                mu, R, D, v, x, t - t_interim, num_approx=num_approx
            )
        return c_sol

    def full_coupled_single_1d_H(self, mu, R, D, v, x, t, num_approx):
        """
        Helper function H when cacluate the analytical solution

        Args:
            mu (float): decay rate * R.
            R (float): retardation factor.
            D (float): diffusion coefficient.
            v (float): pore velocity.
            x (np.ndarry): an array of distance.
            t (float): the final time.
            num_approx (bool): use an approximation for the multiplication between
            exponential and complementary error fuctions.

        Returns:
            np.ndarry: partial result from the analytical solution.
        """
        u = (v**2 + 4 * mu * D) ** 0.5
        if num_approx:
            H1 = 0.5 * self.approx_ExpTimesErfc(
                (v - u) * x / (2 * D), (R * x - u * t) / (2 * (D * R * t) ** 0.5)
            )
            H2 = 0.5 * self.approx_ExpTimesErfc(
                (v + u) * x / (2 * D), (R * x + u * t) / (2 * (D * R * t) ** 0.5)
            )

        else:  # exp(-large_number) is zero,  erfc(>2) is zero.
            H1 = (
                0.5
                * np.exp((v - u) * x / (2 * D))
                * special.erfc((R * x - u * t) / (2 * (D * R * t) ** 0.5))
            )

            H2 = (
                0.5
                * np.exp((v + u) * x / (2 * D))
                * special.erfc((R * x + u * t) / (2 * (D * R * t) ** 0.5))
            )

        return H1 + H2

    def full_coupled_single_1d_M(self, mu, R, D, v, x, t, num_approx):
        """
        Helper function M when cacluate the analytical solution

        Args:
            mu (float): decay rate * R.
            R (float): retardation factor.
            D (float): diffusion coefficient.
            v (float): pore velocity.
            x (np.ndarry): an array of distance.
            t (float): the final time.
            num_approx (bool): use an approximation for the multiplication between
            exponential and complementary error fuctions.

        Returns:
            np.ndarry: partial result from the analytical solution.
        """
        if num_approx:
            M11 = 0.5 * self.approx_ExpTimesErfc(
                -mu * t / R, (R * x - v * t) / (2 * (D * R * t) ** 0.5)
            )
            M12 = 0.5 * self.approx_ExpTimesErfc(
                -mu * t / R + v * x / D, (R * x + v * t) / (2 * (D * R * t) ** 0.5)
            )
        else:
            M11 = (
                0.5
                * np.exp(-mu * t / R)
                * special.erfc((R * x - v * t) / (2 * (D * R * t) ** 0.5))
            )

            M12 = (
                0.5
                * np.exp(-mu * t / R + v * x / D)
                * special.erfc((R * x + v * t) / (2 * (D * R * t) ** 0.5))
            )

        M1 = -(M11 + M12)
        M2 = np.exp(-mu * t / R)
        return M1 + M2

    @staticmethod
    def approx_ExpTimesErfc(yv, xv):
        """
        an approximation for the multiplication between exponential and complementary error fuctions.

        Args:
            yv (np.ndarry or float): values inside expoential function.
            xv (np.ndarry or float): values inside complementary error function.

        Returns:
            np.ndarry: an approximation for exp(yv)*erfc(xv).
        """
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.42141741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        ya = np.append(np.array([]), yv)
        xa = np.append(np.array([]), xv)
        W = np.array([])
        if len(ya) == 1:
            ya = np.full_like(xa, ya)

        for y, x in zip(ya, xa):
            t = 1.0 / (1.0 + p * abs(x))
            W1 = (
                a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
            ) * np.exp(y - x**2)
            if x >= 0:
                W = np.append(W, W1)
            else:
                W = np.append(W, 2 * np.exp(y) - W1)

        return W

    def full_coupled_decay_chain_1d(
        self, c_inlet, c_init, k, R, D, v, t_interim, x, t, num_approx, scaling
    ):
        """
        calculate the analytical solution for multiple species transport process. The analytical solution is constrained by constant initial and boundary
        conditions as well as an identical diffusion coefficient, retardation factor, and pore velocity for each nuclide.

        Args:
            c_inlet (np.ndarry or list): boundary conditions.
            c_init (np.ndarry or list): initial conditions.
            k (np.ndarry or list): decay rates.
            R (np.ndarry or list): retardation factor.
            D (np.ndarry or list): diffusion coefficient.
            v (float): pore velocity.
            t_interim (float): when time is larger than t_interim, the boundary condition
            will set to be zero.
            x (np.ndarry): an array of distance.
            t (float): the final time.
            num_approx (bool): use an approximation for the multiplication between
            exponential and complementary error fuctions.
            scaling (bool): non-dimensionalising the equation.

        Returns:
            np.ndarry: concentrations of nuclides over distance at a given time.
        """
        number_of_nuclides = len(c_inlet)
        a = [
            c_inlet[i] + self.auxiliary_SumProduct(i, c_inlet[:i], k[: i + 1])
            for i in range(number_of_nuclides)
        ]
        a_init = [
            c_init[i] + self.auxiliary_SumProduct(i, c_init[:i], k[: i + 1])
            for i in range(number_of_nuclides)
        ]
        c_interim = []
        for i in range(len(a)):
            c_interim.append(
                self.full_coupled_single_1d(
                    c_init=a_init[i],
                    k=k[i],
                    R=R[i],
                    D=D[i],
                    v=v,
                    c_bc_le_t_interim=a[i],
                    t_interim=t_interim,
                    x=x,
                    t=t,
                    num_approx=num_approx,
                    scaling=scaling,
                )
            )

        c_final = [c_interim[0]]
        for i in range(1, number_of_nuclides):
            c_final.append(
                c_interim[i] - self.auxiliary_SumProduct(i, c_final[:i], k[: i + 1])
            )

        return c_final

    @staticmethod
    def batemann(init, decay_rates, t):
        """
        analytical solution for batemann equation when the inital conditions for all the child nuclides are zero.

        Args:
            init (float): initial condition for the parent nuclide.
            decay_rates (np.ndarry or list): decay rates for all the nuclides
            t (float): the final time.

        Returns:
            np.ndarry: concentrations for all the nuclides at a given time.
        """
        product1 = np.prod(list(map(Decimal, list(decay_rates)))[0:-1])
        N = len(decay_rates)
        sum_nuclide = Decimal(0)
        for i in range(N):
            numerator = Decimal(-decay_rates[i] * t).exp()
            product2 = Decimal(1)
            for j in range(N):
                if i == j:
                    product2 *= Decimal(1.0)
                else:
                    product2 *= Decimal(decay_rates[j]) - Decimal(decay_rates[i])
            sum_nuclide += Decimal(numerator) / Decimal(product2)

        nuclide_t = Decimal(init) * Decimal(product1) * Decimal(sum_nuclide)

        return nuclide_t

    def batemann_decay_chain(self, inits, decay_rates, t):
        """
        A general analytical solution for the batemann equation.

        Args:
            inits(np.ndarry or list): initial conditions for all the nuclides.
            decay_rates (np.ndarry or list): decay rates for all the nuclides.
            t (float): the final time.

        Returns:
            np.ndarry: concentrations for all the nuclides at a given time.
        """
        N = len(decay_rates)

        c_nuclides = list(map(Decimal, list(np.zeros(N))))
        for k in range(N):
            decay_rates = decay_rates[k:]
            N1 = len(decay_rates)
            for i in range(N1):
                c_nuclides[i + k] = c_nuclides[i + k] + self.batemann(
                    inits[k], decay_rates[0 : i + 1], t
                )
        return np.array(c_nuclides, dtype=np.float64)

    def parlett_recurrence_HP(self, nuclides, inits, decay_rates, t, branching_decay):
        """
        A general analytical solution for the decay chain after decays to a certain time with high precision.

        Args:
            nuclides (list): a list of string contains nuclides name.
            inits (np.ndarry):initial conditions for all the nuclides.
            decay_rates (np.ndarry or list): decay rates for all the nuclides.
            t (float): the final time.
            branching_decay(bool): If it is True, consider the branching decay, otherwise only consider the linear decay.

        Returns:
            np.ndarry: concentrations for all the nuclides at a given time.
        """
        N = len(decay_rates)
        F = np.zeros((N, N))
        b = self.partial_decay_constants(nuclides, decay_rates, branching_decay)
        lamda = decay_rates

        for i in range(N):
            F[i][i] = Decimal(-lamda[i] * t).exp()
            # print(Decimal(-lamda[i] * t).exp())

        for p in range(N - 1):
            for j in range(N - p - 1):
                i = j + p + 1
                # s =  Decimal(b[i][j]) * ( Decimal(F[i][i]) -  Decimal(F[j][j]))
                s = Decimal(b[i][j]) * (
                    Decimal(-lamda[i] * t).exp() - Decimal(-lamda[j] * t).exp()
                )

                for k in range(j + 1, i):
                    s += Decimal(F[i][k]) * Decimal(b[k][j]) - Decimal(
                        b[i][k]
                    ) * Decimal(F[k][j])
                F[i][j] = s / (Decimal(-lamda[i]) + Decimal(lamda[j]))

        return F @ inits

    def parlett_recurrence(self, nuclides, inits, decay_rates, t, branching_decay):
        """
        A general analytical solution for the decay chain after decays to a certain time.

        Args:
            nuclides (list): a list of string contains nuclides name.
            inits (np.ndarry):initial conditions for all the nuclides.
            decay_rates (np.ndarry or list): decay rates for all the nuclides.
            t (float): the final time.
            branching_decay(bool): If it is True, consider the branching decay, otherwise only consider the linear decay.

        Returns:
            np.ndarry: concentrations for all the nuclides at a given time.
        """

        N = len(decay_rates)
        F = np.zeros((N, N))
        b = self.partial_decay_constants(nuclides, decay_rates, branching_decay)
        lamda = decay_rates

        for i in range(N):
            F[i][i] = np.exp(-lamda[i] * t)

        for p in range(N - 1):
            for j in range(N - p - 1):
                i = j + p + 1
                s = b[i][j] * (F[i][i] - F[j][j])

                for k in range(j + 1, i):
                    s += F[i][k] * b[k][j] - b[i][k] * F[k][j]
                F[i][j] = s / (-lamda[i] + lamda[j])

        return F @ inits

    @staticmethod
    def partial_decay_constants(nuclides, decay_rates, branching_decay):
        """
        Function for calculate partial decay constants.

        Args:
            nuclides (list): a list of string contains nuclides name.
            decay_rates (np.ndarry or list): decay rates for all the nuclides.
            branching_decay(bool): If it is True, consider the branching decay, otherwise only consider the linear decay.

        Return:
            b (np.ndarry): N*N matrix.
        """
        N = len(nuclides)
        b = np.zeros((N, N))
        if not branching_decay:
            for i in range(N - 1):
                b[i + 1][i] = decay_rates[i]
            return b

        for i in range(N - 1):
            nuclide_progeny = rd.Nuclide(nuclides[i]).progeny()
            nuclide_branching_fractions = rd.Nuclide(nuclides[i]).branching_fractions()
            for progeny, r in zip(nuclide_progeny, nuclide_branching_fractions):
                pro_index = nuclides.index(progeny)
                b[pro_index][i] = r * decay_rates[i]
        return b

    def process_concentrations(
        self,
        nuclides: list,
        c_init,
        c_inlet,
        t_interim: float,
        t_timeseries: np.ndarray,
        x_grids: np.ndarray,
        decay: bool = False,
        diffusion: bool = False,
        advection: bool = False,
        sorption: bool = False,
        num_approx: bool = False,
        scaling: bool = False,
        branching_decay: bool = False,
        diffusion_coefficients=None,
        media_porosities=None,
        darcy_flux=None,
        media_densities=None,
        sorption_coefficients=None,
    ):
        """
        Calculate the analytical spatial and temporal concentration fields.

        Args:
            nuclides (list): a list of string contains nuclides name.
            c_init (np.ndarray or list): initial conditions for all the nuclides.
            c_inlet (np.ndarray or list): boundary conditions for all the nuclides.
            t_interim (float): when time is larger than t_interim, the boundary condition
            will set to be zero.
            t_timeseries (np.ndarray): an array of time steps.
            x_grids (np.ndarray): an array of distance.
            decay (bool, optional): consider the decay process. Defaults to False.
            diffusion (bool, optional): consider the diffusion process. Defaults to False.
            advection (bool, optional): consider the advection process. Defaults to False.
            sorption (bool, optional): consider  the sorption process. Defaults to False.
            num_approx (bool, optional): use an approximation for the multiplication between
            exponential and complementary error fuctions.
            scaling (bool): non-dimensionalising the equation.
            branching_decay(bool): If it is True, consider the branching decay, otherwise only consider the linear decay.
            diffusion_coefficients (np.ndarray or list, optional): effective diffusion
            coefficients of the nuclides. Required when diffusion is True.
            media_porosities (np.ndarray or list, optional): porosities of the media. Required when
            diffusion, advection or sorption is True.
            darcy_flux (np.ndarray or list, optional): darcy flux of the media. Required when advection is True.
            media_densities (np.ndarray or list, optional): densities of the media. Required when sorption is True.
            sorption_coefficients (np.ndarray or list, optional): sorption coefficients of the nuclides.
            Required when sorption is True.

        Returns:
            np.ndarray: spatial and temporal concentration fields.
        """
        number_of_nuclides = len(nuclides)
        R = np.ones(number_of_nuclides)
        v = 0.0
        k = np.zeros(number_of_nuclides)

        c_nuclides = []

        nuclides_half_life = [rd.Nuclide(nuclide).half_life() for nuclide in nuclides]
        nuclides_decay_rates = np.log(2) / np.array(nuclides_half_life)

        if diffusion:
            D = np.array(diffusion_coefficients, dtype=np.float64)
            if len(D) > 1 and len(set(D)) != 1:
                raise NotImplementedError(
                    "The analytical solution is only applicable when the effective diffusion coefficients of all "
                    "nuclides are identical!"
                )

        elif (not (advection or sorption)) and decay:  # if decay only
            for t in t_timeseries:
                c_nuclides.append(
                    self.parlett_recurrence_HP(
                        nuclides=nuclides,
                        inits=c_init,
                        decay_rates=nuclides_decay_rates,
                        t=t,
                        branching_decay=branching_decay,
                    )
                )
            c_nuclides = np.stack([c_nuclides for _ in x_grids], axis=2)
            return c_nuclides
        else:
            raise NotImplementedError(
                f"Analytical solution is not avaible for diffusion={diffusion},decay={decay},"
                f"advection={advection},sorption={sorption} coupled processes"
            )

        if advection:
            v_arr = np.array(darcy_flux, dtype=np.float64) / np.array(
                media_porosities, dtype=np.float64
            )
            if len(v_arr) > 1 and len(set(v_arr)) != 1:
                raise NotImplementedError(
                    "The analytical solution is only applicable when the pore velocity of all "
                    "media is identical."
                )
            v = v_arr[0]

        if sorption:
            sorption_coefficient_nuclides = np.array(
                sorption_coefficients, dtype=np.float64
            )
            # single media
            R = 1 + np.array(sorption_coefficient_nuclides) * float(
                media_densities[0]
            ) / float(media_porosities[0])
            if len(R) > 1 and len(set(R)) != 1:
                raise NotImplementedError(
                    "The analytical solution is only applicable when the sorption coefficients "
                    "of all nuclides are identical."
                )

        if decay:
            k = nuclides_decay_rates

        c_nuclides = []
        for t in t_timeseries:
            c_t = self.full_coupled_decay_chain_1d(
                c_inlet=c_inlet,
                c_init=c_init,
                k=k,
                R=R,
                D=D,
                v=v,
                t_interim=t_interim,
                x=x_grids,
                t=t,
                num_approx=num_approx,
                scaling=scaling,
            )
            c_nuclides.append(c_t)

        return np.array(c_nuclides)
