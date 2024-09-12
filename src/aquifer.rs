use anyhow::Result;
use core::f64;
use ndarray::{Array, Array1, ArrayView1};
use peroxide::fuga::*;
use puruspe::{Jn, Yn};
use std::f64::consts::PI;
use std::str::FromStr;

pub fn water_dimensionless_infinite_marsal(time_d: ArrayView1<'_, f64>) -> Array1<f64> {
    let water_influx = time_d.mapv(|t| {
        if t <= 1.0 {
            2.0 * t.sqrt() / PI + t / 2.0 - t / 6.0 * (t / PI).sqrt() + t.powi(2) / 16.0
        } else if t <= 100.0 {
            8.1638e-1 + 8.5373e-1 * t - 2.7455e-2 * t.powi(2) + 1.0284e-3 * t.powi(3)
                - 2.274e-5 * t.powi(4)
                + 2.8354e-7 * t.powi(5)
                - 1.8436e-9 * t.powi(6)
                + 4.8534e-12 * t.powi(7)
        } else {
            2.0 * t / t.ln()
        }
    });
    water_influx
}

pub fn water_dimensionless_finite_klins(time_d: ArrayView1<'_, f64>, r_ed: f64) -> Array1<f64> {
    let first_term = 0.5 * (r_ed.powi(2) - 1.0);
    let alphas = get_bessel_roots(r_ed, 10, BesselType::Alpha);
    let second_term = time_d.map(|t| {
        alphas.fold(0f64, |acc, alpha| {
            acc - 2.0 * f64::exp(-alpha.powi(2) * t) * (Jn(1, alpha * r_ed)).powi(2)
                / (alpha.powi(2) * (Jn(0, *alpha)).powi(2) - (Jn(1, alpha * r_ed)).powi(2))
        })
    });
    first_term + second_term
}

pub fn get_bessel_roots(r_ed: f64, n_max: usize, bessel_type: BesselType) -> Array1<f64> {
    let mut sample_alphas = Array::linspace(1e-9, 2.0 * n_max as f64 / r_ed, n_max * 400);
    let mut zero_crossings = vec![0];

    while zero_crossings.len() < n_max {
        let new_sample_alphas = sample_alphas.mapv(|x| 2.0 * x);
        let signs = new_sample_alphas.mapv(|alpha| root_func(alpha, r_ed, &bessel_type).signum());
        let diff_signs = signs.windows(2).into_iter().map(|win| win[1] - win[0]);
        zero_crossings = diff_signs
            .enumerate()
            .filter_map(|(i, d)| if d as i8 != 0 { Some(i) } else { None })
            .collect();
        sample_alphas = new_sample_alphas;
    }

    let zero_crossings = &zero_crossings[..n_max];
    let roots: Vec<f64> = zero_crossings
        .iter()
        .map(|&zc| {
            let root_guess = (sample_alphas[zc - 1], sample_alphas[zc + 1]);
            let problem = Bessel {
                r_ed,
                root_guess,
                bessel_type,
            };
            let finder = BisectionMethod {
                max_iter: 100,
                tol: 1e-6,
            };
            let root = finder.find(&problem).unwrap()[0];
            return root;
        })
        .collect::<Vec<f64>>();

    Array::from(roots)
}

pub struct Bessel {
    r_ed: f64,
    root_guess: (f64, f64),
    bessel_type: BesselType,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BesselType {
    Alpha,
    Beta,
}

impl FromStr for BesselType {
    type Err = ();

    fn from_str(input: &str) -> Result<BesselType, Self::Err> {
        match input {
            "alpha" => Ok(BesselType::Alpha),
            "beta" => Ok(BesselType::Beta),
            _ => Err(()),
        }
    }
}

fn root_func(x: f64, r_ed: f64, bessel_type: &BesselType) -> f64 {
    match &bessel_type {
        BesselType::Alpha => Jn(1, x * r_ed) * Yn(0, x) - Yn(1, x * r_ed) * Jn(0, x),
        BesselType::Beta => Jn(1, x * r_ed) * Yn(1, x) - Jn(1, x) * Yn(1, x * r_ed),
    }
}

impl Bessel {
    fn eval(&self, x: [f64; 1]) -> Result<[f64; 1]> {
        Ok([root_func(x[0], self.r_ed, &self.bessel_type)])
    }
}
impl RootFindingProblem<1, 1, (f64, f64)> for Bessel {
    fn function(&self, x: [f64; 1]) -> Result<[f64; 1]> {
        self.eval(x)
    }
    fn initial_guess(&self) -> (f64, f64) {
        self.root_guess
    }
}
