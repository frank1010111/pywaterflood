// Buckley-Leverett calculations

use ndarray::Array1;

/// The value used for `h` in derivative estimates.
pub const EPSILON: f64 = 5.0e-7;

/// Front velocity in m/d
///
/// Above the breakthrough saturation follows this equation:
///
/// $$\begin{equation}
/// \left(\frac{dx}{dt}\right)_{S_w} = \frac{q_t}{\phi A} \left(\frac{\partial f_w}{\partial S_w}\right)_t
/// \end{equation}$$
///
/// Below the breakthrough saturation, the velocity is equal to the velocity for the equation
/// at breakthrough saturation.
///
/// # Arguments
/// * flow_rate: water flow rate in m^3/d
/// * phi: effective porosity
/// * flow_cross_section: Area flow is through in m^2
/// * viscosity_oil: oil viscosity in Pa.s
/// * viscosity_water: water viscosity in Pa.s
/// * sat_oil: relative permeability to oil
/// * sat_water: relative permeability to water
/// * sat_oil_r: residual oil saturation
/// * sat_water_c: critical (residual) water saturation
/// * sat_gas_c: critical gas saturation
/// * n_oil: Brooks-Corey exponent for oil rel-perm
/// * n_water: Brooks-Corey exponent for water rel-perm
///
/// # Returns
/// the front velocity in m/d
pub fn water_front_velocity(
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
) -> f64 {
    if sat_water < sat_water_c {
        return 0.0;
    }
    if sat_water > 1f64 - sat_oil_r {
        return 0.0;
    }
    let sat_wb = breakthrough_sw(
        viscosity_oil,
        viscosity_water,
        sat_oil_r,
        sat_water_c,
        sat_gas_c,
        n_oil,
        n_water,
    );
    if sat_water < sat_wb {
        return water_front_velocity(
            flow_rate,
            phi,
            flow_cross_section,
            viscosity_oil,
            viscosity_water,
            1.0 - sat_wb,
            sat_wb,
            sat_oil_r,
            sat_water_c,
            sat_gas_c,
            n_oil,
            n_water,
        );
    }
    let flow_constant = flow_rate / (phi * flow_cross_section);
    let k_rel_oil_high = k_rel_oil(sat_oil + EPSILON, sat_oil_r, sat_water_c, sat_gas_c, n_oil);
    let k_rel_oil_low = k_rel_oil(sat_oil - EPSILON, sat_oil_r, sat_water_c, sat_gas_c, n_oil);
    let k_rel_water_high = k_rel_water(
        sat_water + EPSILON,
        sat_oil_r,
        sat_water_c,
        sat_gas_c,
        n_water,
    );
    let k_rel_water_low = k_rel_water(
        sat_water - EPSILON,
        sat_oil_r,
        sat_water_c,
        sat_gas_c,
        n_water,
    );
    let fractional_flow_high = fractional_flow_water_flat(
        k_rel_oil_low,
        k_rel_water_high,
        viscosity_oil,
        viscosity_water,
    );
    let fractional_flow_low = fractional_flow_water_flat(
        k_rel_oil_high,
        k_rel_water_low,
        viscosity_oil,
        viscosity_water,
    );

    let d_fractional_flow_d_saturation =
        (fractional_flow_high - fractional_flow_low) / (2_f64 * EPSILON);
    flow_constant * d_fractional_flow_d_saturation
}

/// Water saturation at breakthrough
///
/// # Arguments
/// * viscosity_oil: oil viscosity in Pa.s
/// * viscosity_water: water viscosity in Pa.s
/// * sat_oil_r: residual oil saturation
/// * sat_water_c: critical (residual) water saturation
/// * sat_gas_c: critical gas saturation
/// * n_oil: Brooks-Corey exponent for oil rel-perm
/// * n_water: Brooks-Corey exponent for water rel-perm
pub fn breakthrough_sw(
    viscosity_oil: f64,
    viscosity_water: f64,
    sat_oil_r: f64,
    sat_water_c: f64,
    sat_gas_c: f64,
    n_oil: f64,
    n_water: f64,
) -> f64 {
    let s_w = Array1::range(sat_water_c, 1.0 - sat_oil_r.min(sat_gas_c), 0.005);
    let tangent_frac_flow_water = s_w.map(|&sat_water| {
        let k_oil = k_rel_oil(1.0 - sat_water, sat_oil_r, sat_water_c, sat_gas_c, n_oil);
        let k_water = k_rel_water(sat_water, sat_oil_r, sat_water_c, sat_gas_c, n_water);
        let frac_flow_water =
            fractional_flow_water_flat(k_oil, k_water, viscosity_oil, viscosity_water);
        frac_flow_water / (sat_water - sat_water_c)
    });
    let index_of_max_tanget = tangent_frac_flow_water
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();
    s_w[index_of_max_tanget]
}

/// Water fractional flow for an unangled (flat) reservoir
///
/// $$f_w = \frac{1}{1 + \frac{k_o}{k_w}\frac{\mu_w}{\mu_o}}$$
///
/// # Arguments
/// * k_oil: relative permeability to oil
/// * k_water: relative permeability to water
/// * viscosity_water: water viscosity in Pa.s
///
pub fn fractional_flow_water_flat(
    k_oil: f64,
    k_water: f64,
    viscosity_oil: f64,
    viscosity_water: f64,
) -> f64 {
    1.0 / (1.0 + k_oil * viscosity_water / (k_water * viscosity_oil))
}

/// Water fractional flow for an angled reservoir
///
/// $$\begin{equation}
/// f_w = \frac{1 +  \frac{k_o A}{\mu_o q_t} \left(\rho_o - \rho_w\right) g \sin \alpha}
/// {1 + \frac{k_o}{k_w}\frac{\mu_w}{\mu_o}}
/// \end{equation}$$
///
/// # Arguments
/// * k_oil: relative permeability to oil
/// * k_water: relative permeability to water
/// * viscosity_oil: oil viscosity in Pa.s
/// * viscosity_water: water viscosity in Pa.s
/// * flow_cross_section: Area to flow (in flow "tube") in m^2
/// * flow_total: total flow rate in m^3/s
/// * dip_angle: angle reservoir is dipping in degrees
///
pub fn fractional_flow_water_angled(
    k_oil: f64,
    k_water: f64,
    viscosity_oil: f64,
    viscosity_water: f64,
    density_oil: f64,
    density_water: f64,
    flow_cross_section: f64,
    flow_total: f64,
    dip_angle: f64,
) -> f64 {
    let gravity = 9.807;
    (1.0 + k_oil * flow_cross_section / (viscosity_oil * flow_total))
        * (density_oil - density_water)
        * gravity
        * f64::sin(dip_angle.to_radians())
        / (1.0 + k_oil * viscosity_water / (k_water * viscosity_oil))
}

/// Relative permeability for oil following Brooks-Corey
///
/// $$\begin{equation}
/// k_{ro} = \left(\frac{S_o- S_{or}}{1 - S_{or} - S_{wc}- S_{gc}}\right)^{n_o}
/// \end{equation}$$
///
/// # Arguments
/// * sat_oil: oil saturation
/// * sat_oil_r: residual oil saturation
/// * sat_water_c: critical (residual) water saturation
/// * sat_gas_c: critical gas saturation
/// * n_oil: Brooks-Corey exponent for oil rel-perm
///
pub fn k_rel_oil(
    sat_oil: f64,
    sat_oil_r: f64,
    sat_water_c: f64,
    sat_gas_c: f64,
    n_oil: f64,
) -> f64 {
    if sat_oil < sat_oil_r {
        return 0.0;
    }
    ((sat_oil - sat_oil_r) / (1f64 - sat_oil_r - sat_water_c - sat_gas_c)).powf(n_oil)
}

/// Relative permeability for water following Brooks-Corey
///
///
/// $$\begin{equation}
/// k_{rw} = \left(\frac{S_w- S_{wc}}{1 - S_{or} - S_{wc}- S_{gc}}\right)^{n_o}
/// \end{equation}$$
///
/// # Arguments
/// * sat_water: water saturation
/// * sat_oil_r: residual oil saturation
/// * sat_water_c: critical (residual) water saturation
/// * sat_gas_c: critical gas saturation
/// * n_water: Brooks-Corey exponent for water rel-perm
///
pub fn k_rel_water(
    sat_water: f64,
    sat_oil_r: f64,
    sat_water_c: f64,
    sat_gas_c: f64,
    n_water: f64,
) -> f64 {
    if sat_water < sat_water_c {
        return 0.0;
    }
    ((sat_water - sat_water_c) / (1f64 - sat_oil_r - sat_water_c - sat_gas_c)).powf(n_water)
}

/// Relative permeability for gas following Brooks-Corey
///
///
/// $$\begin{equation}
/// k_{rw} = \left(\frac{S_g- S_{gc}}{1 - S_{or} - S_{wc}- S_{gc}}\right)^{n_o}
/// \end{equation}$$
///
/// # Arguments
/// * sat_gas: free gas saturation
/// * sat_oil_r: residual oil saturation
/// * sat_water_c: critical (residual) water saturation
/// * sat_gas_c: critical gas saturation
/// * n_gas: Brooks-Corey exponent for gas rel-perm
///
pub fn k_rel_gas(
    sat_gas: f64,
    sat_oil_r: f64,
    sat_water_c: f64,
    sat_gas_c: f64,
    n_gas: f64,
) -> f64 {
    if sat_gas < sat_gas_c {
        return 0.0;
    }
    ((sat_gas - sat_gas_c) / (1f64 - sat_oil_r - sat_water_c - sat_gas_c)).powf(n_gas)
}
