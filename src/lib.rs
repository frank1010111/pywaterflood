// library
pub mod crm;
use std::f64::consts::PI;

pub use crate::crm::{q_bhp, q_crm_perpair, q_primary};
use ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
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
    ) -> &'py PyArray1<f64> {
        let production = production.as_array();
        let time = time.as_array();
        let q = q_primary(production, time, gain_producer, tau_producer);
        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm_perpair")]
    fn q_crm_perpair_py<'py>(
        py: Python<'py>,
        injection: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        gains: PyReadonlyArray1<'_, f64>,
        taus: PyReadonlyArray1<'_, f64>,
    ) -> &'py PyArray1<f64> {
        let injection = injection.as_array();
        let time = time.as_array();
        let gains = gains.as_array();
        let taus = taus.as_array();
        let q = q_crm_perpair(injection, time, gains, taus);
        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_bhp")]
    fn q_bhp_py<'py>(
        py: Python<'py>,
        pressure_local: PyReadonlyArray1<'_, f64>,
        pressure: PyReadonlyArray2<'_, f64>,
        v_matrix: PyReadonlyArray1<'_, f64>,
    ) -> &'py PyArray1<f64> {
        let pressure_local = pressure_local.as_array();
        let pressure = pressure.as_array();
        let v_matrix = v_matrix.as_array();
        let q = q_bhp(pressure_local, pressure, v_matrix);
        q.into_pyarray(py)
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

    Ok(())
}
