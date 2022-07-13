// library

use ndarray::{Array1, ArrayView1};
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1
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

    Ok(())
}