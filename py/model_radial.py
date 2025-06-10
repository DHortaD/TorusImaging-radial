from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NewType
from warnings import warn

import astropy.table as at
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import jax.typing as jtp
import jaxopt
import numpy.typing as npt
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt import Bisection
from jaxopt.base import OptStep
from typing_extensions import TypedDict

import inspect
from scipy.stats import binned_statistic
from torusimaging import data
from torusimaging.model_helpers import monotonic_quadratic_spline

from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem
from scipy.stats import binned_statistic_2d


from torusimaging.jax_helpers import simpson

__all__ = ["TorusImaging1D"]

length_pt = u.get_physical_type("length")
velocity_pt = u.get_physical_type("speed")

EParams = NewType("EParams", dict[int, dict[str, jax.Array | float]])


class TorusImaging1DParams(TypedDict, total=False):
    vcirc0: jax.Array | float
    vel0: jax.Array | float
    ln_Omega0: jax.Array | float
    e_params: EParams
    label_params: dict[str, jax.Array | float]


class TorusImaging1D:
    r"""A flexible and customizable interface for fitting and MCMC sampling an Orbital
    Torus Imaging model. This implementation assumes that you are working in a 1 degree
    of freedom phase space with position coordinate ``q`` and velocity coordinate ``p``.

    Notation:

    * :math:`\Omega_0` or ``Omega0``: A scale frequency used to compute the
      elliptical radius ``r_e``. This is the asymptotic orbital frequency at zero
      action.
    * :math:`r_e` or ``r_e``: The elliptical radius
      :math:`r_e = \sqrt{q^2\, \Omega_0 + p^2 \, \Omega_0^{-1}}`.
    * :math:`\theta_e` or ``theta_e``: The elliptical angle defined as
      :math:`\tan{\theta_e} = \frac{q}{p}\,\Omega_0`.
    * :math:`r` or ``r``: The distorted elliptical radius
      :math:`r = r_e \, f(r_e, \theta_e)`, which is close  to :math:`\sqrt{J}` (the
      action) and so we sometimes call it the "proxy action" below. :math:`f(\cdot)`
      is the distortion function defined below.
    * :math:`f(r_e, \theta_e)`: The distortion function is a Fourier expansion,
      defined as: :math:`f(r_e, \theta_e) = 1 + \sum_m e_m(r_e)\,\cos(m\,\theta_e)`
    * :math:`J` or ``J``: The action.
    * :math:`\theta` or ``theta``: The conjugate angle.

    Parameters
    ----------
    label_func
        A function that specifies the dependence of the label function on the distorted
        radius :math:`r`.
    e_funcs
        A dictionary that provides functions that specify the dependence of the Fourier
        distortion coefficients :math:`e_m(r_e)`. Keys should be the (integer) "m" order
        of the distortion term (for the distortion function), and values should be
        Python callable objects that can be passed to `jax.jit()`. The first argument of
        each of these functions should be the elliptical radius :math:`r_e` or ``re``.
    regularization_func
        An optional function that computes a regularization term to add to the
        log-likelihood function when optimizing.
    units
        The unit system to work in. Default is to use the "galactic" unit system from
        Gala: (kpc, Myr, Msun, radian).

    """

    def __init__(
        self,
        label_func: Callable[[float], float],
        e_funcs: dict[int, Callable[[float], float]],
        regularization_func: Callable[[Any], float] | None = None,
        units: UnitSystem = galactic,
    ):
        self.label_func = jax.jit(label_func)
        self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}

        # Unit system:
        self.units = UnitSystem(units)

        if regularization_func is None:
            regularization_func = lambda *_, **__: 0.0  # noqa: E731
        self.regularization_func = regularization_func

    # ---------------------------------------------------------------------------------
    # Internal functions used within likelihood functions:
    #
    @partial(jax.jit, static_argnames=["self"])
    def _get_elliptical_coords(
        self,
        pos: float | jax.Array,
        vel: float | jax.Array,
        pos0: float,
        vel0: float,
        ln_Omega0: float,
    ) -> tuple[float | jax.Array, float | jax.Array]:
        r"""Compute the raw elliptical radius :math:`r_e` (``r_e``) and angle
        :math:`\theta_e'` (``theta_e``)
        """
        
        x = (pos - pos0) * jnp.sqrt(jnp.exp(ln_Omega0))
        y = (vel - vel0) / jnp.sqrt(jnp.exp(ln_Omega0))

        r_e = jnp.sqrt(x**2 + y**2)
        t_e = jnp.arctan2(y, x)

        return r_e, t_e

    @partial(jax.jit, static_argnames=["self"])
    def _get_es(self, r_e: float, e_params: EParams) -> dict[int, float]:
        """Compute the Fourier m-order distortion coefficients"""
        es = {}
        for m, pars in e_params.items():
            es[m] = self.e_funcs[m](r_e, **pars)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def _get_r(
        self,
        r_e: float,
        theta_e: float,
        e_params: EParams,
    ) -> jax.Array:
        """Compute the distorted radius :math:`r`"""
        es = self._get_es(r_e, e_params)
        return r_e * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(m * theta_e) for m, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_theta(
        self,
        r_e: float,
        theta_e: float,
        e_params: EParams,
    ) -> jax.Array:
        """Compute the phase angle"""
        es = self._get_es(r_e, e_params)
        # TODO: why is the π/2 needed below??
        return theta_e - jnp.sum(
            jnp.array(
                [m / (jnp.pi / 2) * e * jnp.sin(m * theta_e) for m, e in es.items()]
            ),
            axis=0,
        )

    @partial(jax.jit, static_argnames=["self"])
    def _get_r_e(
        self,
        r: float,
        theta_e: float,
        e_params: EParams,
        Bisection_kwargs: dict[str, Any],
    ) -> float:
        """Compute the elliptical radius :math:`r_e` by inverting the distortion
        transformation from :math:`r`
        """
        Bisection_kwargs = dict(Bisection_kwargs)
        Bisection_kwargs.setdefault("lower", 0.0)
        Bisection_kwargs.setdefault("upper", 1.0)
        Bisection_kwargs.setdefault("maxiter", 30)
        Bisection_kwargs.setdefault("tol", 1e-4)

        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self._get_r(x, tt_prime, ee_params)
            - rrz,
            jit=True,
            unroll=True,
            check_bracket=False,
            **Bisection_kwargs,
        )
        return float(bisec.run(r, rrz=r, tt_prime=theta_e, ee_params=e_params).params)

    @partial(jax.jit, static_argnames=["self"])
    def _get_pos(
        self,
        r: float,
        theta_e: float,
        params: TorusImaging1DParams,
        Bisection_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Compute the position given the distorted radius and elliptical angle"""
        r_e = self._get_r_e(r, theta_e, params["e_params"], Bisection_kwargs)
        return r_e * jnp.sin(theta_e) / jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_vel(
        self,
        r: float,
        theta_e: float,
        params: TorusImaging1DParams,
        Bisection_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Compute the velocity given the distorted radius and elliptical angle"""
        rzp = self._get_r_e(r, theta_e, params["e_params"], Bisection_kwargs)
        return rzp * jnp.cos(theta_e) * jnp.sqrt(jnp.exp(params["ln_Omega0"]))

    @partial(jax.jit, static_argnames=["self"])
    def _get_label(
        self,
        pos: float,
        vel: float,
        params: TorusImaging1DParams,
    ) -> jax.Array:

        pos0 = R - (R*vphi)/params['vcirc0']
        
        r_e, th_e = self._get_elliptical_coords(
            pos,
            vel,
            pos0=pos0,
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        r = self._get_r(r_e, th_e, params["e_params"])
        return self.label_func(r, **params["label_params"])

    @partial(jax.jit, static_argnames=["self", "N_grid", "Bisection_kwargs"])
    def _get_T_J_theta(
        self,
        R: float,
        vphi: float,
        vel: float,
        vcirc: float,
        params: TorusImaging1DParams,
        N_grid: int,
        Bisection_kwargs: dict[str, Any],
    ) -> tuple[float, float, float]:

        pos = R - (R*vphi)/vcirc
        pos0 = R - (R*vphi)/params['vcirc0']
        
        re_, the_ = self._get_elliptical_coords(
            R, 
            pos,
            vel,
            vcirc,
            pos0=pos0,
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        r = self._get_r(re_, the_, params["e_params"])

        dpos_dthe_func = jax.vmap(
            jax.grad(self._get_pos, argnums=1), in_axes=[None, 0, None, None]
        )

        get_vel = jax.vmap(self._get_vel, in_axes=[None, 0, None, None])

        # Grid of theta_prime to do the integral over:
        the_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        v_th = get_vel(r, the_grid, params, Bisection_kwargs)
        dz_dthp = dpos_dthe_func(r, the_grid, params, Bisection_kwargs)

        Tz = 4 * simpson(dz_dthp / v_th, the_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * v_th, the_grid)

        thp_partial = jnp.linspace(0, the_, N_grid)
        v_th_partial = get_vel(r, thp_partial, params, Bisection_kwargs)
        dpos_dthe_partial = dpos_dthe_func(r, thp_partial, params, Bisection_kwargs)
        dt = simpson(dpos_dthe_partial / v_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_T_J_theta = jax.vmap(_get_T_J_theta, in_axes=[None, 0, 0, None, None, None])

    @partial(jax.jit, static_argnames=["self"])
    def _get_de_dr_es(self, r_e: float, e_params: EParams) -> dict[int, float]:
        """Compute the derivatives of the Fourier m-order distortion coefficient
        functions
        """
        d_es = {}
        for m, pars in e_params.items():
            d_es[m] = jax.grad(self.e_funcs[m], argnums=0)(r_e, **pars)
        return d_es

    # ---------------------------------------------------------------------------------
    # Public API
    #
    # @u.quantity_input
    # def compute_elliptical(
    #     self,
    #     pos: u.Quantity[length_pt],
    #     vel: u.Quantity[velocity_pt],
    #     params: TorusImaging1DParams,
    # ) -> tuple[u.Quantity, u.Quantity]:
    #     """Compute the elliptical radius :math:`r_e` (``r_e``) and angle
    #     :math:`\theta_e'` (``theta_e``)

    #     Parameters
    #     ----------
    #     pos
    #         The position values.
    #     vel
    #         The velocity values.
    #     params
    #         A dictionary of model parameters.
    #     """

    #     x = pos.decompose(self.units).value
    #     v = vel.decompose(self.units).value
    #     re, te = self._get_elliptical_coords(
    #         x,
    #         v,
    #         pos0=params["pos0"],
    #         # R0=params["R0"],
    #         # vphi0=params["vphi0"],
    #         # Lz0=params['Lz0'],
    #         # vcirc0=params["vcirc0"],
    #         vel0=params["vel0"],
    #         ln_Omega0=params["ln_Omega0"],
    #     )
    #     return (
    #         re
    #         * self.units["length"]
    #         / (self.units["angle"] ** 0.5 / self.units["time"] ** 0.5),
    #         te * self.units["angle"],
    #     )

    # @u.quantity_input
    # def compute_action_angle(
    #     self,
    #     pos: u.Quantity[length_pt],
    #     vel: u.Quantity[velocity_pt],
    #     params: TorusImaging1DParams,
    #     N_grid: int = 32,
    #     Bisection_kwargs: dict[str, Any] | None = None,
    # ) -> at.QTable:
    #     """Compute the vertical period, action, and angle given input phase-space
    #     coordinates.

    #     Parameters
    #     ----------
    #     pos
    #         The position values.
    #     vel
    #         The velocity values.
    #     params
    #         A dictionary of model parameters.
    #     N_grid
    #         The number of grid points to use in estimating the action integral.
    #     """
    #     x = pos.decompose(self.units).value
    #     v = vel.decompose(self.units).value

    #     if Bisection_kwargs is None:
    #         Bisection_kwargs = {}

    #     T, J, th = self._get_T_J_theta(x, v, params, N_grid, Bisection_kwargs)

    #     tbl = at.QTable()
    #     tbl["T"] = T * self.units["time"]
    #     tbl["Omega"] = 2 * jnp.pi * u.rad / tbl["T"]
    #     tbl["J"] = J * self.units["length"] ** 2 / self.units["time"]
    #     tbl["theta"] = th * self.units["angle"]

    #     return tbl

    @partial(jax.jit, static_argnames=["self"])
    def _get_acc(
        self,
        R: float,
        vphi: float,
        vel: float,
        vcirc: float,
        params: TorusImaging1DParams,
    ) -> jax.Array:

        pos = R - (R*vphi)/vcirc
        pos0 = R - (R*vphi)/params['vcirc0']
        
        r_e, _ = self._get_elliptical_coords(
            pos, 
            0.0,
            pos0 = pos0,
            vel0=0.0,
            ln_Omega0=params["ln_Omega0"],
        )

        Om = jnp.exp(params["ln_Omega0"])

        es = self._get_es(r_e, params["e_params"])
        de_dres = self._get_de_dr_es(r_e, params["e_params"])

        numer = 1 + jnp.sum(
            jnp.array(
                [(es[m] + de_dres[m] * r_e) for m in self.e_funcs]
            )
        )
        denom = 1 + jnp.sum(
            jnp.array(
                [(es[m] + de_dres[m] * r_e) - m**2 * es[m] for m in self.e_funcs]
            )
        )
        pos0 = params['pos0']
        return -(Om**2) * (pos - pos0) * numer / denom

    _get_dacc_dpos = jax.grad(_get_acc, argnums=1)
    _get_dacc_dpos_vmap = jax.vmap(_get_dacc_dpos, in_axes=(None, 0, None))

    # @u.quantity_input
    # def get_acceleration(
    #     self,
    #     pos: u.Quantity[length_pt],
    #     params: TorusImaging1DParams,
    # ) -> u.Quantity:
    #     """Compute the acceleration as a function of position in the limit as velocity
    #     goes to zero

    #     Parameters
    #     ----------
    #     pos
    #         The position values.
    #     params
    #         A dictionary of model parameters.
    #     """
    #     x = jnp.atleast_1d(pos.decompose(self.units).value)
    #     in_shape = x.shape
    #     x = x.ravel()

    #     get_acc = jax.vmap(self._get_acc, in_axes=[0, None])
    #     res = get_acc(x, params)
    #     return res.reshape(in_shape) * self.units["acceleration"]

    # @u.quantity_input
    # def get_acceleration_deriv(
    #     self,
    #     pos: u.Quantity[length_pt],
    #     params: TorusImaging1DParams,
    # ) -> u.Quantity:
    #     """Compute the derivative of the acceleration with respect to position as a
    #     function of position in the limit as velocity goes to zero

    #     Parameters
    #     ----------
    #     pos
    #         The position values.
    #     params
    #         A dictionary of model parameters.
    #     """
    #     x = jnp.atleast_1d(pos.decompose(self.units).value)
    #     in_shape = x.shape
    #     x = x.ravel()

    #     res = self._get_dacc_dpos_vmap(x, params)
    #     return res.reshape(in_shape) * self.units["acceleration"] / self.units["length"]

    # @u.quantity_input
    # def get_label(
    #     self,
    #     pos: u.Quantity[length_pt],
    #     vel: u.Quantity[velocity_pt],
    #     params: TorusImaging1DParams,
    # ) -> jax.Array:
    #     """Compute the model predicted label value given the input phase-space
    #     coordinates
    #     """
    #     x = pos.decompose(self.units).value
    #     v = vel.decompose(self.units).value
    #     return self._get_label(x.ravel(), v.ravel(), params).reshape(x.shape)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(
        self,
        params: TorusImaging1DParams,
        R: jax.Array,
        vphi: jax.Array,
        vel: jax.Array,
        vcirc: float,
        counts: jax.Array,
    ) -> jax.Array:
        """Compute the log-likelihood of the Poisson likelihood function. This should
        be used when the label you are modeling is the log-number of stars per pixel,
        i.e. the phase-space density itself.

        Note: the input position and velocity arrays must already be converted to the
        unit system of the model.
        """
        # Expected number:
        ln_Lambda = self._get_label(R, vphi, vel, vcirc, params)

        # gammaln(x+1) = log(factorial(x))
        return (counts * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(counts + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def ln_gaussian_likelihood(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        label: jax.Array,
        label_err: jax.Array,
    ) -> jax.Array:
        """Compute the log-likelihood of the Gaussian likelihood function.

        Note: the input position and velocity arrays must already be converted to the
        unit system of the model.
        """
        model_label = self._get_label(pos, vel, params)
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective_poisson(
        self,
        params: TorusImaging1DParams,
        R: jax.Array,
        vphi: jax.Array,
        vel: jax.Array,
        vcirc: float,
        counts: npt.ArrayLike,
    ):
        f_val = self.ln_poisson_likelihood(params, R, vphi, vel, vcirc, counts)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    @partial(jax.jit, static_argnames=["self"])
    def objective_gaussian(
        self,
        params: TorusImaging1DParams,
        pos: jax.Array,
        vel: jax.Array,
        label: jax.Array,
        label_err: jax.Array,
    ):
        f_val = self.ln_gaussian_likelihood(params, pos, vel, label, label_err)
        return -(f_val - self.regularization_func(self, params)) / pos.size

    def optimize(
        self,
        params0: dict,
        objective: Literal["poisson", "gaussian"],
        bounds: tuple[dict] | None = None,
        jaxopt_kwargs: dict | None = None,
        **data: u.Quantity | jtp.ArrayLike,
    ) -> OptStep:
        """Optimize the model parameters given the input data using
        `jaxopt.ScipyboundedMinimize`.

        Parameters
        ----------
        params0
            The initial values of the parameters.
        objective
            The string name of the objective function to use (either "poisson" or
            "gaussian").
        bounds
            The bounds on the parameters. This can either be a tuple of dictionaries, or
            a dictionary of tuples (keyed by parameter names) to specify the lower and
            upper bounds for each parameter.
        jaxopt_kwargs
            Any keyword arguments passed to ``jaxopt.ScipyBoundedMinimize``.
        **data
            Passed through to the objective function.

        """
        import numpy as np

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault("maxiter", 16384)

        vals, treedef = jax.tree_util.tree_flatten(params0)
        params0 = treedef.unflatten([np.array(x, dtype=np.float64) for x in vals])

        jaxopt_kwargs.setdefault("method", "L-BFGS-B")
        optimizer = jaxopt.ScipyBoundedMinimize(
            fun=getattr(self, f"objective_{objective}"),
            **jaxopt_kwargs,
        )

        data = {k: jnp.array(v) for k, v in data.items()}

        if bounds is not None:
            # Detect packed bounds (a single dict):
            if isinstance(bounds, dict):
                bounds = self.unpack_bounds(bounds)

            res = optimizer.run(init_params=params0, bounds=bounds, **data)

        else:
            res = optimizer.run(init_params=params0, **data)

        # warn if optimization was not successful, set state if successful
        if not res.state.success:
            warn(
                "Optimization failed! See the returned result object for more "
                "information, but the model state was not updated",
                stacklevel=1,
            )

        return res

    @classmethod
    def unpack_bounds(cls, bounds: dict) -> tuple[dict]:
        """Split a bounds dictionary that is specified like: {"key": (lower, upper)}
        into two bounds dictionaries for the lower and upper bounds separately, e.g.,
        for the example above: {"key": lower} and {"key": upper}.
        """
        import numpy as np

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}

            d = np.array(d)
            assert d.shape[0] == 2
            return d

        # Make sure all tuples / lists become arrays:
        clean_bounds = clean_dict(bounds)

        vals, treedef = jax.tree_util.tree_flatten(clean_bounds)

        bounds_l = treedef.unflatten([np.array(x[0], dtype=np.float64) for x in vals])
        bounds_r = treedef.unflatten([np.array(x[1], dtype=np.float64) for x in vals])

        return bounds_l, bounds_r

    def check_e_funcs(
        self, e_params: EParams, r_e_max: float
    ) -> tuple[bool, npt.NDArray]:
        """Check that the parameter values and functions used for the e functions are
        valid given the condition that d(r)/d(r_e) > 0.
        """
        import numpy as np

        # TODO: 16 is a magic number
        r_es = np.linspace(np.sqrt(1e-3), np.sqrt(r_e_max), 16) ** 2

        # TODO: potential issue if order of arguments in e_funcs() call is different
        # from the order of the values in the e_params dictionary...
        d_em_d_re_funcs = {
            m: jax.vmap(
                jax.grad(self.e_funcs[m], argnums=0),
                in_axes=[0] + [None] * len(e_params[m]),
            )
            for m in self.e_funcs
        }

        thes = np.linspace(0, np.pi / 2, 128)
        checks = np.zeros((len(r_es), len(thes)))

        for j, th_e in enumerate(thes):
            checks[:, j] = jnp.sum(
                jnp.array(
                    [
                        jnp.cos(m * th_e)
                        * (
                            e_func(r_es, **e_params[m])
                            + r_es * d_em_d_re_funcs[m](r_es, *e_params[m].values())
                        )
                        for m, e_func in self.e_funcs.items()
                    ]
                ),
                axis=0,
            )

        # This condition has to be met such that d(r_z)/d(r_z') > 0 at all theta_z':
        return np.all(checks > -1), checks

    def get_crlb(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
        inv: bool = False,
    ) -> npt.NDArray:
        """Returns the Cramer-Rao lower bound matrix for the parameters evaluated at the
        input parameter values.

        To instead return the Fisher information matrix, specify ``inv=True``.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)

        def wrapper(flat_params, data, sizes):
            arrs = []
            i = 0
            for size in sizes:
                arrs.append(jnp.array(flat_params[i : i + size]))
                i += size
            params = jax.tree_util.tree_unflatten(treedef, arrs)
            ll = getattr(self, f"ln_{objective}_likelihood")(params, **data)
            return -(ll - self.regularization_func(self, params))

        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        fisher = jax.hessian(wrapper)(flat_params, data, sizes)
        if inv:
            return fisher
        return np.linalg.inv(fisher)

    def get_crlb_uncertainties(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
    ) -> dict[str, dict | npt.ArrayLike]:
        """Compute the uncertainties on the parameters using the diagonal of the
        Cramer-Rao lower bound matrix (see :meth:`get_crlb`).
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]

        fisher_inv = self.get_crlb(params, data, objective=objective)
        diag = np.diag(fisher_inv).copy()
        diag[(diag < 0) | (diag > 1e18)] = 0.0
        flat_param_uncs = np.sqrt(diag)

        arrs = []
        i = 0
        for size in sizes:
            arrs.append(jnp.array(flat_param_uncs[i : i + size]))
            i += size
        return jax.tree_util.tree_unflatten(treedef, arrs)

    def get_crlb_error_samples(
        self,
        params: TorusImaging1DParams,
        data: dict[str, npt.ArrayLike],
        objective: str = "gaussian",
        size: int = 1,
        seed: int | None = None,
        list_of_samples: bool = True,
    ) -> list[dict] | dict[str, dict | npt.ArrayLike]:
        """Generate Gaussian samples of parameter values centered on the input parameter
        values with covariance matrix set by the Cramer-Rao lower bound matrix.
        """
        import numpy as np

        treedef = jax.tree_util.tree_structure(params)
        flattened = jax.tree_util.tree_flatten(params)[0]
        sizes = [x.size for x in flattened]
        flat_params = np.concatenate([np.atleast_1d(x) for x in flattened])

        crlb = self.get_crlb(params, data, objective=objective)
        diag = np.diag(crlb)
        bad_idx = np.where((diag < 0) | (diag > 1e18))[0]

        for i in bad_idx:
            crlb[i] = crlb[:, i] = 0.0
            crlb[i, i] = 1.0

        rng = np.random.default_rng(seed=seed)
        samples = rng.multivariate_normal(flat_params, crlb, size=size)

        for i in bad_idx:
            samples[:, i] = np.nan

        arrs = []
        i = 0
        for size_ in sizes:
            arrs.append(jnp.array(samples[..., i : i + size_]))
            i += size_

        if list_of_samples:
            samples = []
            for n in range(size):
                samples.append(
                    jax.tree_util.tree_unflatten(treedef, [arr[n] for arr in arrs])
                )
            return samples

        return jax.tree_util.tree_unflatten(treedef, arrs)

    def mcmc_run_label(
        self,
        binned_data: dict,
        p0: dict,
        bounds: tuple[dict] | None = None,
        rng_seed: int = 0,
        num_steps: int = 1000,
        num_warmup: int = 1000,
    ) -> tuple[Any, list[dict]]:
        """Currently only supports uniform priors on all parameters, specified by the
        input bounds.

        Parameters
        ----------
        binned_data
            A dictionary containing the binned label moment data.
        p0
            The initial values of the parameters.
        bounds
            The bounds on the parameters, used to define uniform priors on the
            parameters. This can either be a tuple of dictionaries, or a dictionary of
            tuples (keyed by parameter names) to specify the lower and upper bounds for
            each parameter.
        rng_seed
            The random number generator seed.
        num_steps
            The number of MCMC steps to take.
        num_warmup
            The number of warmup or burn-in steps to take to tune the NUTS sampler.

        Returns
        -------
        state
            The HMCState object returned by BlackJAX.
        mcmc_samples
            A list of dictionaries containing the parameter values for each MCMC sample.
        """
        import blackjax
        import numpy as np

        # First check that objective evaluates to a finite value:
        mask = (
            np.isfinite(binned_data["label"])
            & np.isfinite(binned_data["label_err"])
            & (binned_data["label_err"] > 0)
        )
        data = {
            "pos": binned_data["pos"].decompose(self.units).value[mask],
            "vel": binned_data["vel"].decompose(self.units).value[mask],
            "label": binned_data["label"][mask],
            "label_err": binned_data["label_err"][mask],
        }
        test_val = self.objective_gaussian(p0, **data)
        if not np.isfinite(test_val):
            msg = "Objective function evaluated to non-finite value"
            raise RuntimeError(msg)

        lb, ub = self.unpack_bounds(bounds)
        lb_arrs = jax.tree_util.tree_flatten(lb)[0]
        ub_arrs = jax.tree_util.tree_flatten(ub)[0]

        def logprob(p):
            lp = 0.0
            pars, _ = jax.tree_util.tree_flatten(p)
            for i in range(len(pars)):
                lp += jnp.where(
                    jnp.any(pars[i] < lb_arrs[i]) | jnp.any(pars[i] > ub_arrs[i]),
                    -jnp.inf,
                    0.0,
                )

            lp += self.ln_gaussian_likelihood(p, **data)

            lp -= self.regularization_func(self, p)

            return lp

        rng_key = jax.random.PRNGKey(rng_seed)
        warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
        (state, parameters), _ = warmup.run(rng_key, p0, num_steps=num_warmup)

        kernel = blackjax.nuts(logprob, **parameters).step  # pylint: disable=no-member
        states = inference_loop(rng_key, kernel, state, num_steps)

        # Get the pytree structure of a single sample based on the starting point:
        treedef = jax.tree_util.tree_structure(p0)
        arrs, _ = jax.tree_util.tree_flatten(states.position)

        mcmc_samples = []
        for n in range(arrs[0].shape[0]):
            mcmc_samples.append(
                jax.tree_util.tree_unflatten(treedef, [arr[n] for arr in arrs])
            )

        return states, mcmc_samples

def _get_bins_tuple(bins, units=None):
    if isinstance(bins, dict):
        bins = (bins["pos"], bins["vel"])

    if units is not None:
        bins = [b.decompose(units).value if hasattr(b, "unit") else b for b in bins]
    else:
        bins = [b.value if hasattr(b, "unit") else b for b in bins]

    return bins

def _get_arr(x, units):
    if units is not None:
        return x.decompose(units).value if hasattr(x, "unit") else x
    return x.value if hasattr(x, "unit") else x
    
@u.quantity_input
def get_binned_label(
    R: u.Quantity[u.kpc],
    vphi: u.Quantity[u.km / u.s],
    vel: u.Quantity[u.km / u.s],
    vcirc: u.Quantity[u.km / u.s],
    label: npt.ArrayLike,
    bins: dict[str, u.Quantity] | tuple,
    moment: str = "mean",
    label_err: npt.ArrayLike | None = None,
    units: UnitSystem | None = None,
    s: float | None = None,
    s_N_thresh: int | None = 128,
) -> dict[str, u.Quantity | npt.NDArray]:
    """Bin the data in pixels of phase-space coordinates (pos, vel) and return the
    mean (or other moment) of the label values in each pixel.

    Parameters
    ----------
    pos
        The position values.
    vel
        The velocity values.
    label
        The label values.
    bins
        A specification of the bins. This can either be a tuple, where the order
        is assumed to be (pos, vel), or a dictionary with keys "pos" and "vel".
    moment
        The type of moment to compute. Currently only supports "mean".
    label_err
        The measurement error for each label value.
    units
        The unit system to work in.
    s
        The intrinsic scatter of label values within each pixel. If not provided,
        this will be estimated from the data.
    s_N_thresh
        If the intrinsic scatter ``s`` is not specified, this sets the threshold for the
        number of objects per bin required to estimate the intrinsic scatter.

    Returns
    -------
    dict
        Keys are "pos", "vel", "counts", "label", and "label_err".
    """
    if moment != "mean":
        msg = "Only the mean is currently supported."
        raise NotImplementedError(msg)

    pre_err_state = np.geterr()
    np.seterr(divide="ignore", invalid="ignore")

    pos = R - (R*vphi)/vcirc

    pos = _get_arr(pos, units)
    vel = _get_arr(vel, units)
    bins = _get_bins_tuple(bins, units)

    xc = 0.5 * (bins[0][:-1] + bins[0][1:])
    yc = 0.5 * (bins[1][:-1] + bins[1][1:])
    xc, yc = np.meshgrid(xc, yc)

    # binned = {
    #     "R": xc * units["length"],
    #     "vphi": xc * units["length"],
    #     "vel": yc * units["length"] / units["time"],
    #     "vcirc": xc * units["length"],
    # }
    binned = {
        "pos": xc * units["length"],
        "vel": yc * units["length"] / units["time"],
    }

    # For bin numbers and other stuff below:
    counts_stat = binned_statistic_2d(
        pos, vel, None, bins=bins, statistic="count", expand_binnumbers=True
    )
    counts = counts_stat.statistic

    if label_err is None:
        # No label errors provided - assume dominated by intrinsic scatter
        label_err = np.zeros_like(label)

        if s is None:
            # estimate just doing the stddev of bins with more than N objects
            std_stat = binned_statistic_2d(
                pos, vel, label, bins=bins, statistic=np.nanstd
            )
            s = np.nanmean(std_stat.statistic[counts > s_N_thresh])

    if s is None:
        # Label errors provided, but no intrinsic scatter provided - need to estimate
        # this for bins with many objects
        high_N_bins = np.stack(np.where(counts > s_N_thresh)) + 1
        high_N_bins = high_N_bins[
            :, np.argsort(counts[np.where(counts > s_N_thresh)])[::-1]
        ]
        high_N_bins = high_N_bins[:, np.any(high_N_bins != 1, axis=0)]

        # TODO: magic number - this limits to a subset of the 16 most populated bins
        s_trials = []
        for bin_idx in high_N_bins.T[:16]:
            bin_mask = np.all(counts_stat.binnumber == bin_idx[:, None], axis=0)
            # to get bin location: stat.x_edge[bin_idx[0]], stat.y_edge[bin_idx[1]]

            s_trials.append(
                _infer_intrinsic_scatter(
                    label[bin_mask], label_err[bin_mask], nan_safe=True
                )
            )
        s = np.nanmean(s_trials)

        if not np.isfinite(s):
            msg = "Failed to determine intrinsic scatter from label data"
            raise ValueError(msg)

    if np.all(label_err == 0):
        # No label errors provided
        stat_mean = binned_statistic_2d(
            pos, vel, label, bins=bins, statistic=np.nanmean
        )
        mean = stat_mean.statistic
        mean_err = 0.0

    else:
        # Compute the mean and the "error on the mean" in each bin:
        stat_mean1 = binned_statistic_2d(
            pos,
            vel,
            label / label_err**2,
            bins=bins,
            statistic="sum",
        )
        stat_mean2 = binned_statistic_2d(
            pos, vel, 1 / label_err**2, bins=bins, statistic="sum"
        )
        mean = stat_mean1.statistic / stat_mean2.statistic
        mean_err = np.sqrt(1 / stat_mean2.statistic)

    binned["counts"] = counts.T
    binned["label"] = mean.T
    binned["label_err"] = np.sqrt(mean_err**2 + s**2 / counts).T
    binned["label_err"][~np.isfinite(binned["label_err"])] = np.nan
    # binned["s"] = s

    np.seterr(**pre_err_state)

    return binned
        
def inference_loop(
    rng_key: jax.random.PRNGKey, kernel: Any, initial_state: Any, num_samples: int
) -> Any:
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

#################################### model_spline
__all__ = ["TorusImaging1DSpline"]


def label_func_base(
    r: jtp.ArrayLike, label_vals: jtp.ArrayLike, knots: jtp.ArrayLike
) -> jax.Array:
    return monotonic_quadratic_spline(knots, label_vals, r)


def e_func_base(
    r_e: jtp.ArrayLike, vals: jtp.ArrayLike, sign: float, knots: jtp.ArrayLike
) -> jax.Array:
    return sign * monotonic_quadratic_spline(
        knots, jnp.concatenate((jnp.array([0.0]), jnp.exp(vals))), r_e
    )


def regularization_func_default(
    model: TorusImaging1D,
    params: TorusImaging1DParams,
    label_l2_sigma: float,
    label_smooth_sigma: float,
    e_l2_sigmas: dict[int, float],
    e_smooth_sigmas: dict[int, float],
    dacc_dpos_scale: float = 1e-4,
    dacc_strength: float = 1.0,
) -> jax.Array:
    p = 0.0

    if dacc_strength > 0:
        # Soft rectifier regularization meant to keep d(acc)/d(pos) < 0
        # (i.e. this tries to enforce positive density)
        for m in model.e_funcs:
            z_knots = model._e_knots[m][1:] / jnp.sqrt(jnp.exp(params["ln_Omega0"]))
            daz = model._get_dacc_dpos_vmap(z_knots, params) / dacc_dpos_scale
            p += dacc_strength * jnp.sum(jnp.log(1 + jnp.exp(daz)))

    # L2 regularization to keep the value of the functions small:
    for m, func in model.e_funcs.items():
        p += jnp.sum(
            (func(model._e_knots[m], **params["e_params"][m]) / e_l2_sigmas[m]) ** 2
        )

    p += jnp.sum(
        (
            model.label_func(model._label_knots, **params["label_params"])
            / label_l2_sigma
        )
        ** 2
    )

    # L2 regularization for smoothness:
    for m in params["e_params"]:
        diff = params["e_params"][m]["vals"][1:] - params["e_params"][m]["vals"][:-1]
        p += jnp.sum((diff / e_smooth_sigmas[m]) ** 2)

    diff = (
        params["label_params"]["label_vals"][2:]
        - params["label_params"]["label_vals"][1:-1]
    )
    p += jnp.sum((diff / label_smooth_sigma) ** 2)

    return p


def _infer_intrinsic_scatter(y, y_err, nan_safe=False):
    import jax
    import jax.numpy as jnp
    import jaxopt

    @jax.jit
    def ln_likelihood(p, y, y_err):
        V = jnp.exp(2 * p["ln_s"]) + y_err**2
        return jnp.nansum(-0.5 * ((y - p["mean"]) ** 2 / V + jnp.log(2 * jnp.pi * V)))

    @jax.jit
    def neg_ln_likelihood(p, y, y_err):
        return -ln_likelihood(p, y, y_err)

    opt = jaxopt.ScipyMinimize(
        method="L-BFGS-B", fun=neg_ln_likelihood, tol=1e-10, maxiter=1000
    )

    p0 = {"mean": np.nanmean(y), "ln_s": jnp.log(np.nanstd(y))}
    res = opt.run(p0, y=y, y_err=y_err)

    s = np.exp(res.params["ln_s"])
    if not res.state.success:
        if nan_safe:
            s = np.nan
        else:
            msg = (
                "Failed to determine error-deconvolved estimate of the intrinsic "
                "scatter in a phase-space pixel."
            )
            raise RuntimeError(msg)

    return s


class TorusImaging1DSpline(TorusImaging1D):
    """A version of the ``TorusImaging1D`` model that uses splines to model the label function and the Fourier coefficient :math:`e_m` functions.

    Parameters
    ----------
    label_knots
        The spline knot locations for the label function.
    e_knots
        A dictionary keyed by m integers with values as the spline knot locations
        for the e functions.
    e_signs
        A dictionary keyed by m integers with values as the signs of the e
        functions.
    regularization_func
        A function that takes in a ``TorusImaging1DSpline`` instance and a parameter
        dictionary and returns an additional regularization term to add to the
        log-likelihood.
    units
        A Gala :class:`gala.units.UnitSystem` instance.
    """

    def __init__(
        self,
        label_knots: jtp.ArrayLike,
        e_knots: dict[int, jtp.ArrayLike],
        e_signs: dict[int, float | int],
        regularization_func: Callable[[Any], jax.Array] | None = None,
        units: UnitSystem = galactic,
    ):
        self._label_knots = jnp.array(label_knots)
        label_func = partial(label_func_base, knots=self._label_knots)

        self._e_signs = {m: float(v) for m, v in e_signs.items()}
        self._e_knots = {m: jnp.array(knots) for m, knots in e_knots.items()}
        e_funcs = {
            m: partial(e_func_base, sign=e_signs[m], knots=knots)
            for m, knots in self._e_knots.items()
        }

        super().__init__(label_func, e_funcs, regularization_func, units)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._label_knots,
                self._e_knots,
                self._e_signs,
                self.regularization_func,
                self.units,
            ),
        )

    @classmethod
    def auto_init(
        cls,
        binned_data: dict[str, jtp.ArrayLike],
        label_knots: int | npt.ArrayLike,
        e_knots: dict[int, int | npt.ArrayLike],
        e_signs: dict[int, float | int] | None = None,
        regularization_func: Callable[[Any], jax.Array] | bool | None = None,
        units: UnitSystem = galactic,
        label_knots_spacing_power: float = 1.0,
        e_knots_spacing_power: float = 1.0,
        re_max_factor: float = 1.0,
        **kwargs: Any,
    ) -> tuple["TorusImaging1DSpline", dict[str, Any], TorusImaging1DParams]:
        """
        Parameters
        ----------
        binned_data
            A dictionary with keys "pos", "vel", "label", "label_err".
        label_knots
            Either an integer number of knots to use, or an array of knot positions.
        e_knots
            A dictionary keyed by the m order of the e function, with values either
            the number of knots to use, or an array of knot positions.
        e_signs
            A dictionary keyed by the m order of the e function, with values 1 or -1
            to represent the sign of the gradient of the e function.
        regularization_func
            A function that takes in two arguments: a ``TorusImaging1DSpline`` instance
            and a parameter dictionary and returns an additional regularization term to
            add to the log-likelihood. If not specified, this defaults to the
            :func:`torusimaging.model_spline.regularization_function_default` and
            additional arguments to that function must be specified here. The default
            regularization function tries to enforce smoothness on the splines, and that
            the density is positive. It requires the following keyword arguments:
            ``label_l2_sigma, label_smooth_sigma, e_l2_sigmas, e_smooth_sigmas``. If
            `False`, no regularization is applied.
        units
            A Gala :class:`gala.units.UnitSystem` instance.
        **kwargs
            All other keyword arguments are passed to the constructor.
        """
        import astropy.units as u
        import numpy as np
        from astropy.constants import G  # pylint: disable = no-name-in-module

        bounds = {}

        # TODO: assume binned data - but should it be particle data?
        init_Omega = estimate_Omega(binned_data)

        # First estimate r_e_max using the bin limits and estimated frequency:
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            re_max = re_max_factor * np.mean(
                [
                    (binned_data["pos"].max() * np.sqrt(init_Omega))
                    .decompose(units)
                    .value,
                    (binned_data["vel"].max() / np.sqrt(init_Omega))
                    .decompose(units)
                    .value,
                ]
            )

        # -----------------------------------------------------------------------------
        # Label function: knots, bounds, and initial parameters
        #
        label_knots = np.array(label_knots)
        if label_knots.ndim == 0:
            # Integer passed in, so we need to generate the knots:
            label_knots = np.linspace(
                0, re_max**label_knots_spacing_power, label_knots
            ) ** (1 / label_knots_spacing_power)
        label_n_knots = len(label_knots)

        # Set up reasonable bounds for spline parameters - this estimates the slope of
        # the labels at a few places with respect to position. Later, we have to scale
        # by sqrt(Omega) to get the units right for the label function (defined as a
        # function of r, not position).
        # TODO: magic numbers 10 and 4
        vel_mask = (
            np.abs(binned_data["vel"])
            < np.nanpercentile(np.abs(binned_data["vel"]), 10)
        ) & (binned_data["counts"] > 4)
        label_stat = binned_statistic(
            np.abs(binned_data["pos"][vel_mask]),
            binned_data["label"][vel_mask],
            bins=np.linspace(0, binned_data["pos"].max(), 8),
        )
        xc = 0.5 * (label_stat.bin_edges[1:] + label_stat.bin_edges[:-1])
        # TODO: 10 is a magic number
        label_slope = 10 * np.nanmean(np.diff(label_stat.statistic) / np.diff(xc))

        label_slope_sign = np.sign(label_slope)
        dlabel_dpos = np.abs(label_slope)

        if label_slope_sign > 0:
            dlabel_dpos_bounds = (
                np.full(label_n_knots - 1, 0),
                np.full(label_n_knots - 1, dlabel_dpos),
            )
        else:
            dlabel_dpos_bounds = (
                np.full(label_n_knots - 1, -dlabel_dpos),
                np.full(label_n_knots - 1, 0),
            )
        x0 = label_stat.statistic[np.isfinite(label_stat.statistic)][0]
        label_5span = 5 * np.std(
            binned_data["label"][np.isfinite(binned_data["label"])]
        )
        label0_bounds = x0 + np.array([-label_5span, label_5span])

        bounds["label_params"] = {
            "label_vals": (
                np.concatenate(([label0_bounds[0]], dlabel_dpos_bounds[0])),
                np.concatenate(([label0_bounds[1]], dlabel_dpos_bounds[1])),
            )
        }

        # -----------------------------------------------------------------------------
        # e functions: knots, bounds, and initial parameters
        #

        e_knots = {m: np.array(knots) for m, knots in e_knots.items()}
        for m, knots in e_knots.items():
            if knots.ndim == 0:
                # Integer passed in, so we need to generate the knots:
                e_knots[m] = np.linspace(0, re_max**e_knots_spacing_power, knots) ** (
                    1 / e_knots_spacing_power
                )
        e_n_knots = {m: len(knots) for m, knots in e_knots.items()}

        if e_signs is None:
            e_signs = {}
        default_e_signs = {m: (-1.0 if (m / 2) % 2 == 0 else 1.0) for m in e_knots}
        e_signs = {m: e_signs.get(m, default_e_signs[m]) for m in e_knots}

        # Use some hard-set heuristics for e function parameter bounds
        e_bounds = {}
        for m, n in e_n_knots.items():
            # TODO: hard-set magic numbers - both are truly arbitrary
            # Bounds for e functions, in log-space
            # TODO: change name to log_vals?
            e_bounds.setdefault(
                m, {"vals": (jnp.full(n - 1, -16.0), jnp.full(n - 1, 10.0))}
            )
        bounds["e_params"] = e_bounds

        # -----------------------------------------------------------------------------
        # Regularization function
        #
        if regularization_func is False:
            reg_func = None

        else:
            if regularization_func is None:
                regularization_func = regularization_func_default

            # Regularization function could take other arguments that have to be
            # specified as kwargs to this classmethod, as is the case for the default
            # function:
            sig = inspect.signature(regularization_func)
            arg_names = list(sig.parameters.keys())[2:]

            reg_kw = {}
            for arg_name in arg_names:
                p = sig.parameters[arg_name]
                if arg_name not in kwargs and p.default is inspect._empty:
                    msg = (
                        "The regularization function requires additional arguments: "
                        f"{arg_names!s}, which must be passed as keyword arguments to "
                        "this class method"
                    )
                    raise ValueError(msg)
                reg_kw[arg_name] = kwargs.get(arg_name, p.default)

            reg_func = partial(regularization_func, **reg_kw)

        # Initialize model instance:
        obj = cls(
            label_knots=label_knots,
            e_knots=e_knots,
            e_signs=e_signs,
            regularization_func=reg_func,
            units=units,
        )

        # Other parameter bounds:
        # Wide, physical bounds for the log-midplane density
        dens0_bounds = [0.001, 100] * u.Msun / u.pc**3
        bounds["ln_Omega0"] = 0.5 * np.log(
            (4 * np.pi * G * dens0_bounds).decompose(units).value
        )
        bounds["vcirc0"] = ([-100., 100.0] * u.km / u.s).decompose(units).value
        bounds["vel0"] = ([-100.0, 100.0] * u.km / u.s).decompose(units).value

        init_params = obj.estimate_init_params(binned_data, bounds)

        # Need to scale the bounds of the label function derivatives by sqrt(Omega)
        sqrtOmega = np.sqrt(np.exp(init_params["ln_Omega0"]))
        bounds["label_params"]["label_vals"][0][1:] /= sqrtOmega
        bounds["label_params"]["label_vals"][1][1:] /= sqrtOmega

        return obj, bounds, init_params

    def estimate_init_params(
        self, binned_data: dict[str, npt.ArrayLike], bounds: dict[str, Any]
    ) -> TorusImaging1DParams:
        import numpy as np

        Omega0 = estimate_Omega(binned_data).decompose(self.units).value
        p0 = {"vcirc0": 0.0, "vel0": 0.0, "ln_Omega0": np.log(Omega0)}

        # Parameters left to estimate: e_params, label_params

        # e_params
        p0["e_params"] = {
            m: {"vals": bounds["e_params"][m]["vals"][0]}  # lower bound
            for m in self._e_knots
        }
        
        # label_params
        r_e, _ = self._get_elliptical_coords(
            binned_data["pos"].ravel(),
            binned_data["vel"].ravel(),
            pos0=p0['vcirc0'], # equal to 0, just an initial guess vcirc0 here means nothing
            vel0=p0["vel0"],
            ln_Omega0=p0["ln_Omega0"],
        )

        # Estimate the label value near r_e = 0 and slopes for knot values:
        label = binned_data["label"].ravel()
        fin_mask = np.isfinite(label)
        r1, r2 = np.nanpercentile(r_e[fin_mask], [5, 95])
        label0 = np.nanmean(label[(r_e <= r1) & fin_mask])
        label_slope = (np.nanmedian(label[(r_e >= r2) & fin_mask]) - label0) / (r2 - r1)

        p0["label_params"] = {
            "label_vals": np.concatenate(
                (
                    [label0],
                    np.full(len(self._label_knots) - 1, label_slope),
                )
            )
        }

        return p0

def estimate_Omega(binned_data):
    # TODO: percentile values are hard-coded and arbitrary
    inner_mask = (
        np.abs(binned_data["pos"]) < np.nanpercentile(np.abs(binned_data["pos"]), 15)
    ) & (np.abs(binned_data["vel"]) < np.nanpercentile(np.abs(binned_data["vel"]), 15))
    inner_label_val = np.nanmean(binned_data["label"][inner_mask])

    diff = np.abs(binned_data["label"] - inner_label_val)

    ell_mask = diff < np.nanpercentile(diff, 15)
    tmpv = binned_data["vel"][ell_mask]
    tmpz = binned_data["pos"][ell_mask]
    init_Omega = MAD(tmpv) / MAD(tmpz)

    return init_Omega * u.rad


