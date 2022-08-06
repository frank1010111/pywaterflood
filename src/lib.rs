// library

use ndarray::{Array1, ArrayView1, Array2, ArrayView2, Zip};
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1,PyReadonlyArray2
};
use pyo3::{
    pymodule,
    types::{PyModule},
    PyResult, Python,
};


#[pymodule]
fn pywaterflood(_py: Python, m: &PyModule) -> PyResult<()> {

    fn q_primary(
        production: ArrayView1<'_,f64>,
        time: ArrayView1<'_, f64>,
        gain_producer: f64,
        tau_producer: f64,
    ) -> Array1<f64> {
        let time_decay = (-&time / tau_producer).mapv(f64::exp);
        let q_hat = time_decay * production[[0]] * gain_producer;
        q_hat
    }

    fn q_crm_perpair(
        injection: ArrayView2<'_,f64>,
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
                for m in 1..k+1 {
                    let time_decay = (1.0 - ((time[m - 1] - time[m]) / taus[j]).exp()) * (
                        (time[m] - time[k]) / taus[j]
                    ).exp();
                    convolve[[k, j]] += time_decay * injection[[m, j]];
                }
            }
        }
        let q_hat = convolve.dot(&gains);
        q_hat
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
    ) ->&'py PyArray1<f64> {
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
    )->&'py PyArray1<f64> {
        let injection = injection.as_array();
        let time = time.as_array();
        let gains = gains.as_array();
        let taus = taus.as_array();
        let q = q_crm_perpair(injection, time, gains, taus);
        q.into_pyarray(py)
    }

    Ok(())
}