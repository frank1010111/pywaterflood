use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub fn q_primary(
    production: ArrayView1<'_, f64>,
    time: ArrayView1<'_, f64>,
    gain_producer: f64,
    tau_producer: f64,
) -> Array1<f64> {
    let time_decay = (-&time / tau_producer).mapv(f64::exp);
    time_decay * production[[0]] * gain_producer
}

pub fn q_crm_perpair(
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

pub fn q_bhp(
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
