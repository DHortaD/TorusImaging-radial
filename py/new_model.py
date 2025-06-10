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
    # Internal functions used within likelihood functions

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

    
    # @partial(jax.jit, static_argnames=["self"])
    # def _get_binned_data(
    #     self, 
    #     R: float | jax.Array,
    #     vphi: float | jax.Array,
    #     vel: float | jax.Array, # feed in units of 
    #     vcirc: float,
    #     vcirc0: float,
    #     vel0: float,
    #     label: float | jax.Array,
    # )
        
    #     pos = (R - (R*vphi)/vcirc)*u.kpc
    #     pos0 = (R - (R*vphi)/vcirc0)*u.kpc
    #     vel = vel*u.km/u.s
        
    #     pos = _get_arr(pos, units)
    #     vel = _get_arr(vel, units)
    #     bins = _get_bins_tuple(bins, units)
    
    #     xc = 0.5 * (bins[0][:-1] + bins[0][1:])
    #     yc = 0.5 * (bins[1][:-1] + bins[1][1:])
    #     xc, yc = np.meshgrid(xc, yc)
    
    #     binned = {
    #         "pos": xc * units["length"],
    #         "vel": yc * units["length"] / units["time"],
    #     }
    
        
    #     return binned, pos0
        
    @partial(jax.jit, static_argnames=["self"])
    def _get_elliptical_coords(
        self,
        R: float | jax.Array,
        vphi: float | jax.Array,
        vel: float | jax.Array,
        vcirc: float,
        vcirc0: float,
        vel0: float,
        ln_Omega0: float,
    ) -> tuple[float | jax.Array, float | jax.Array]:
        r"""Compute the raw elliptical radius :math:`r_e` (``r_e``) and angle
        :math:`\theta_e'` (``theta_e``)
        """
        # bdata, pos0 = _get_binned_data(R, vphi, vel, vcirc, vcirc0, vel0)
        pos = (R - (R*vphi)/vcirc)*u.kpc
        pos0 = (R - (R*vphi)/vcirc0)*u.kpc
        
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
    def _get_label(
        self,
        R: float | jax.Array,
        vphi: float | jax.Array,
        vel: float | jax.Array,
        vcirc: float,
        params: TorusImaging1DParams,
    ) -> jax.Array:
        
        r_e, th_e = self._get_elliptical_coords(
            R,
            vphi,
            vel,
            vcirc,
            vcirc0=params['vcirc0'],
            vel0=params["vel0"],
            ln_Omega0=params["ln_Omega0"],
        )
        r = self._get_r(r_e, th_e, params["e_params"])
        return self.label_func(r, **params["label_params"])

        
    @partial(jax.jit, static_argnames=["self"])
    def ln_gaussian_likelihood(
        self,
        params: TorusImaging1DParams,
        R: float | jax.Array,
        vphi: float | jax.Array,
        vel: float | jax.Array,
        vcirc: float,
        label: jax.Array,
        label_err: jax.Array,
    ) -> jax.Array:
        """Compute the log-likelihood of the Gaussian likelihood function.

        Note: the input position and velocity arrays must already be converted to the
        unit system of the model.
        """
        model_label = self._get_label(R, vphi, vel, vcirc, params)
        return -0.5 * jnp.nansum((label - model_label) ** 2 / label_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective_gaussian(
        self,
        params: TorusImaging1DParams,
        R: float | jax.Array,
        vphi: float | jax.Array,
        vel: float | jax.Array,
        vcirc: float,
        label: jax.Array,
        label_err: jax.Array,
    ):
        f_val = self.ln_gaussian_likelihood(params, R, vphi, vel, vcirc, label, label_err)
        return -f_val

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

        

def label_func_base(
    r: jtp.ArrayLike, label_vals: jtp.ArrayLike, knots: jtp.ArrayLike
) -> jax.Array:
    return monotonic_quadratic_spline(knots, label_vals, r)


@jax.jit
def monotonic_quadratic_spline(x, y, x_eval):
    """
    The zeroth element in the knot value array is the value of the spline at x[0], but
    all other values passed in via y are the *derivatives* of the function at the knot
    locations x[1:].
    """

    # Checked that using .at[].set() is faster than making padded arrays and stacking
    x = jnp.array(x)
    y = jnp.array(y)
    x_eval = jnp.array(x_eval)

    N = 3 * (len(x) - 1)
    A = jnp.zeros((N, N))
    b = jnp.zeros((N,))
    A = A.at[0, :3].set([x[0] ** 2, x[0], 1])
    b = b.at[0].set(y[0])
    A = A.at[1, :3].set([2 * x[1], 1, 0])
    b = b.at[1].set(y[1])

    for i, n in enumerate(2 * jnp.arange(1, len(x) - 1, 1), start=1):
        A = A.at[n, 3 * i : 3 * i + 3].set([2 * x[i], 1, 0])
        b = b.at[n].set(y[i])
        A = A.at[n + 1, 3 * i : 3 * i + 3].set([2 * x[i + 1], 1, 0])
        b = b.at[n + 1].set(y[i + 1])

    for j, m in enumerate(jnp.arange(2 * (len(x) - 1), N - 1)):
        A = A.at[m, 3 * j : 3 * j + 3].set([x[j + 1] ** 2, x[j + 1], 1])
        A = A.at[m, 3 * (j + 1) : 3 * (j + 1) + 3].set(
            [-(x[j + 1] ** 2), -x[j + 1], -1]
        )

    A = A.at[-1, 0].set(1.0)

    coeffs = jnp.linalg.solve(A, b)

    # Determine the interval that x lies in
    ind = jnp.digitize(x_eval, x) - 1
    ind = 3 * jnp.clip(ind, 0, len(x) - 2)
    coeff_ind = jnp.stack((ind, ind + 1, ind + 2), axis=0)

    xxx = jnp.stack([x_eval**2, x_eval, jnp.ones_like(x_eval)], axis=0)
    return jnp.sum(coeffs[coeff_ind] * xxx, axis=0)





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

        pos = binned_data['R'] - (binned_data['R']*binned_data['vphi'])/binned_data['vcirc']
        # First estimate r_e_max using the bin limits and estimated frequency:
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            re_max = re_max_factor * np.mean(
                [
                    (pos.max() * np.sqrt(init_Omega))
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
            bins=np.linspace(0, pos.max(), 8),
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
            binned_data["R"].ravel(),
            binned_data["vphi"].ravel(),
            binned_data["vel"].ravel(),
            binned_data["vcirc"],
            vcirc0=p0['vcirc0'], # equal to 0, just an initial guess vcirc0 here means nothing
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
    pos = binned_data['R'] - (binned_data['R']*binned_data['vphi'])/binned_data['vcirc']

    inner_mask = (
        np.abs(pos) < np.nanpercentile(pos), 15)
    ) & (np.abs(binned_data["vel"]) < np.nanpercentile(np.abs(binned_data["vel"]), 15))
    inner_label_val = np.nanmean(binned_data["label"][inner_mask])

    diff = np.abs(binned_data["label"] - inner_label_val)

    ell_mask = diff < np.nanpercentile(diff, 15)
    tmpv = binned_data["vel"][ell_mask]
    tmpz = pos[ell_mask]
    init_Omega = MAD(tmpv) / MAD(tmpz)

    return init_Omega * u.rad










