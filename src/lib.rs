// library

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn pywaterflood(_py: Python, m: &PyModule) -> PyResult<()> {
    fn q_primary(
        production: ArrayView1<'_, f64>,
        time: ArrayView1<'_, f64>,
        gain_producer: f64,
        tau_producer: f64,
    ) -> Array1<f64> {
        let time_decay = (-&time / tau_producer).mapv(f64::exp);
        time_decay * production[[0]] * gain_producer
    }

    fn q_crm_perpair(
        injection: ArrayView2<'_, f64>,
        time: ArrayView1<'_, f64>,
        gains: ArrayView1<'_, f64>,
        taus: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let n_t = time.raw_dim()[0];
        let n_inj = gains.raw_dim()[0];
        let mut convolve: Array2<f64> = Array2::zeros([n_t, n_inj]);

        for j in 0..n_inj {
            convolve[[0, j]] = (1.0 - ((time[0] - time[1]) / taus[j]).exp()) * injection[[0, j]];
            for k in 1..n_t {
                for m in 1..k + 1 {
                    let time_decay = (1.0 - ((time[m - 1] - time[m]) / taus[j]).exp())
                        * ((time[m] - time[k]) / taus[j]).exp();
                    convolve[[k, j]] += time_decay * injection[[m, j]];
                }
            }
        }
        convolve.dot(&gains)
    }

    fn q_bhp(
        pressure_local: ArrayView1<'_, f64>,
        pressure: ArrayView2<'_, f64>,
        v_matrix: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let n_t: usize = pressure.raw_dim()[0];
        let n_prod: usize = pressure.raw_dim()[1];
        let mut pressure_diff: Array2<f64> = Array2::zeros([n_t, n_prod]);
        for j in 0..n_prod {
            for t in 1..n_t {
                pressure_diff[[t, j]] = pressure_local[t - 1] - pressure[[t, j]]
            }
        }
        pressure_diff.dot(&v_matrix)
    }

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
