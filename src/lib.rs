// library

pub mod buckleyleverett;

pub mod aquifer;
pub mod crm;
use std::f64::consts::PI;

pub use crate::crm::{q_bhp, q_crm_perpair, q_primary};
use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

pub use crate::buckleyleverett::{
    breakthrough_sw, fractional_flow_water_flat, k_rel_oil, k_rel_water, water_front_velocity,
};

#[pymodule]
fn _core<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn calc_a_ij(x_i: f64, y_i: f64, x_j: f64, y_j: f64, y_d: f64, m: ArrayView1<u64>) -> f64 {
        let first_term = 2.0
            * PI
            * y_d
            * (1.0 / 3.0 - y_i / y_d + (y_i.powi(2) + y_j.powi(2)) / (2.0 * y_d.powi(2)));

        let t = m.map(|&mm| {
            (f64::cosh(mm as f64 * PI * (y_d - f64::abs(y_i - y_j)))
                + f64::cosh(mm as f64 * PI * (y_d - y_i - y_j)))
                / f64::sinh(mm as f64 * PI * y_d)
        });
        let s1 = 2.0
            * t.iter()
                .zip(m)
                .map(|(&tm, &mm)| {
                    tm / (mm as f64)
                        * f64::cos(mm as f64 * PI * x_i)
                        * f64::cos(mm as f64 * PI * x_j)
                })
                .sum::<f64>();
        let tn = *t.last().unwrap();
        let s2 = -tn / 2.0
            * f64::ln(
                (1.0 - f64::cos(PI * (x_i + x_j))).powi(2) + f64::sin(PI * (x_i + x_j)).powi(2),
            )
            - tn / 2.0
                * f64::ln(
                    (1.0 - f64::cos(PI * (x_i - x_j))).powi(2) + f64::sin(PI * (x_i - x_j)).powi(2),
                );
        let s3 = -2.0
            * tn
            * m.map(|&mm| {
                1.0 / mm as f64 * f64::cos(mm as f64 * PI * x_i) * f64::cos(mm as f64 * PI * x_j)
            })
            .sum();
        first_term + s1 + s2 + s3
    }

    //wrapper
    #[pyfn(m)]
    #[pyo3(name = "q_primary")]
    fn q_primary_py<'py>(
        py: Python<'py>,
        production: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        gain_producer: f64,
        tau_producer: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let production = production.as_array();
        let time = time.as_array();
        let q = q_primary(production, time, gain_producer, tau_producer);
        q.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm_perpair")]
    fn q_crm_perpair_py<'py>(
        py: Python<'py>,
        injection: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        gains: PyReadonlyArray1<'_, f64>,
        taus: PyReadonlyArray1<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let injection = injection.as_array();
        let time = time.as_array();
        let gains = gains.as_array();
        let taus = taus.as_array();
        let q = q_crm_perpair(injection, time, gains, taus);
        q.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_bhp")]
    fn q_bhp_py<'py>(
        py: Python<'py>,
        pressure_local: PyReadonlyArray1<'_, f64>,
        pressure: PyReadonlyArray2<'_, f64>,
        v_matrix: PyReadonlyArray1<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let pressure_local = pressure_local.as_array();
        let pressure = pressure.as_array();
        let v_matrix = v_matrix.as_array();
        let q = q_bhp(pressure_local, pressure, v_matrix);
        q.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_A_ij")]
    fn calc_a_ij_py(
        _py: Python<'_>,
        x_i: f64,
        y_i: f64,
        x_j: f64,
        y_j: f64,
        y_d: f64,
        m: PyReadonlyArray1<'_, u64>,
    ) -> f64 {
        let m = m.as_array();
        calc_a_ij(x_i, y_i, x_j, y_j, y_d, m)
    }

    #[pyfn(m)]
    #[pyo3(name = "water_front_velocity")]
    /// Front velocity
    ///
    /// $$\begin{equation}
    /// \left(\frac{dx}{dt}\right)_{S_w} = \frac{q_t}{\phi A} \left(\frac{\partial f_w}{\partial S_w}\right)_t
    /// \end{equation}$$
    fn water_front_velocity_py(
        flow_rate: f64,
        phi: f64,
        flow_cross_section: f64,
        viscosity_oil: f64,
        viscosity_water: f64,
        sat_oil: f64,
        sat_water: f64,
        sat_oil_r: f64,
        sat_water_c: f64,
        sat_gas_c: f64,
        n_oil: f64,
        n_water: f64,
    ) -> PyResult<f64> {
        Ok(water_front_velocity(
            flow_rate,
            phi,
            flow_cross_section,
            viscosity_oil,
            viscosity_water,
            sat_oil,
            sat_water,
            sat_oil_r,
            sat_water_c,
            sat_gas_c,
            n_oil,
            n_water,
        ))
    }

    #[pyfn(m)]
    #[pyo3(name = "breakthrough_sw")]
    /// Water saturation at breakthrough
    ///
    /// Find through checking the water front velocity at different saturations
    /// until the tangent line from 0 velocity and residual water saturation is
    /// maximized
    fn breakthrough_sw_py(
        viscosity_oil: f64,
        viscosity_water: f64,
        sat_oil_r: f64,
        sat_water_c: f64,
        sat_gas_c: f64,
        n_oil: f64,
        n_water: f64,
    ) -> PyResult<f64> {
        Ok(breakthrough_sw(
            viscosity_oil,
            viscosity_water,
            sat_oil_r,
            sat_water_c,
            sat_gas_c,
            n_oil,
            n_water,
        ))
    }

    #[pyfn(m)]
    #[pyo3(name = "fractional_flow_water")]
    /// Water fractional flow for an unangled (flat) reservoir
    ///
    /// $$\begin{equation}
    /// f_w = \frac{1}{1 + \frac{k_o}{k_w}\frac{\mu_w}{\mu_o}}
    /// \end{equation}$$
    ///
    /// # Arguments
    /// * k_oil: relative permeability to oil
    /// * k_water: relative permeability to water
    /// * viscosity_water: water viscosity in Pa.s
    fn fractional_flow_water_py(
        k_oil: f64,
        k_water: f64,
        viscosity_oil: f64,
        viscosity_water: f64,
    ) -> PyResult<f64> {
        Ok(fractional_flow_water_flat(
            k_oil,
            k_water,
            viscosity_oil,
            viscosity_water,
        ))
    }

    #[pyfn(m)]
    #[pyo3(name = "k_rel_oil")]
    /// Relative permeability for water following Brooks-Corey
    ///
    /// $$\begin{equation}
    /// k_{ro} = \left(\frac{S_o- S_{or}}{1 - S_{or} - S_{wc}- S_{gc}}\right)^{n_o}
    /// \end{equation}$$
    ///
    fn k_rel_oil_py(
        sat_oil: f64,
        sat_oil_r: f64,
        sat_water_c: f64,
        sat_gas_c: f64,
        n_oil: f64,
    ) -> PyResult<f64> {
        Ok(k_rel_oil(sat_oil, sat_oil_r, sat_water_c, sat_gas_c, n_oil))
    }

    #[pyfn(m)]
    #[pyo3(name = "k_rel_water")]
    /// Relative permeability for water following Brooks-Corey
    ///
    /// $$\begin{equation}
    /// k_{rw} = \left(\frac{S_w- S_{wc}}{1 - S_{or} - S_{wc}- S_{gc}}\right)^{n_o}
    /// \end{equation}$$
    ///
    fn k_rel_water_py(
        sat_water: f64,
        sat_oil_r: f64,
        sat_water_c: f64,
        sat_gas_c: f64,
        n_water: f64,
    ) -> PyResult<f64> {
        Ok(k_rel_water(
            sat_water,
            sat_oil_r,
            sat_water_c,
            sat_gas_c,
            n_water,
        ))
    }

    #[pyfn(m)]
    #[pyo3(name = "w_ek")]
    /// Cumulative water influx from the aquifer
    ///
    /// $$\begin{equation}
    /// W_{ek}(t) = U \sum_{j=0}^{k-1} \Delta p_{j+1}
    /// W_D(t_{Dk} - t_{Dj}) \quad \text{for } k = 1, 2, \ldots, n
    /// \end{equation}$$
    ///
    fn w_ek_py<'py>(
        py: Python<'py>,
        w_d: PyReadonlyArray1<f64>,
        delta_pressure: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let w_d = w_d.as_array();
        let delta_pressure = delta_pressure.as_array();

        let n = w_d.len();
        let mut w_ek = Array1::zeros(n);
        for k in 1..n {
            for j in 0..(k - 1) {
                w_ek[k] += delta_pressure[j + 1] * w_d[k - j];
            }
        }
        w_ek.into_pyarray_bound(py)
    }

    Ok(())
}
