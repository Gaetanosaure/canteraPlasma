/**
 *  @file EEDFTwoTermApproximation.cpp
 *  EEDF Two-Term approximation solver.  Implementation file for class
 *  EEDFTwoTermApproximation.
 */

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/thermo/EEDFTwoTermApproximation.h"
#include "cantera/base/ctexceptions.h"
#include "cantera/numerics/eigen_dense.h"
#include "cantera/numerics/funcs.h"
#include "cantera/thermo/PlasmaPhase.h"
#include "cantera/kinetics/ElectronCollisionPlasmaRate.h"

namespace Cantera
{

typedef Eigen::SparseMatrix<double> SparseMat;

namespace {
constexpr bool DEBUG_SE = false;
constexpr bool DEBUG_SE_ITER = false;

double debugTrapzMoment(const Eigen::VectorXd& grid,
                        const Eigen::VectorXd& f,
                        double power,
                        double emin = 0.0)
{
    if (grid.size() != f.size() || grid.size() < 2) {
        return -1.0;
    }

    double s = 0.0;

    for (Eigen::Index i = 0; i + 1 < grid.size(); i++) {
        const double e0 = grid[i];
        const double e1 = grid[i + 1];

        if (e0 < emin || e1 < emin) {
            continue;
        }

        const double y0 = f[i] * std::pow(std::max(e0, 0.0), power);
        const double y1 = f[i + 1] * std::pow(std::max(e1, 0.0), power);

        s += 0.5 * (y0 + y1) * (e1 - e0);
    }

    return s;
}

void debugEedfSummary(const string& label,
                      const Eigen::VectorXd& grid,
                      const Eigen::VectorXd& f)
{
    if (!DEBUG_SE) {
        return;
    }

    const double normSqrt = debugTrapzMoment(grid, f, 0.5);
    const double norm32 = debugTrapzMoment(grid, f, 1.5);
    const double meanSqrtNorm = normSqrt > 0.0 ? norm32 / normSqrt : -1.0;
    const double tailRaw11p5 = debugTrapzMoment(grid, f, 0.0, 11.5);
    const double tailRaw15p8 = debugTrapzMoment(grid, f, 0.0, 15.8);

    writelog(
        "EEDF-SE-DEBUG SUMMARY {}: npts={}, emin={}, emax={}, "
        "norm_sqrt={}, norm_32={}, mean_sqrt_norm={}, "
        "tail_raw_gt_11p5={}, tail_raw_gt_15p8={}, f_first={}, f_last={}\n",
        label,
        grid.size(),
        grid.size() > 0 ? grid[0] : -1.0,
        grid.size() > 0 ? grid[grid.size() - 1] : -1.0,
        normSqrt,
        norm32,
        meanSqrtNorm,
        tailRaw11p5,
        tailRaw15p8,
        f.size() > 0 ? f[0] : -1.0,
        f.size() > 0 ? f[f.size() - 1] : -1.0
    );
}

size_t debugNearestIndex(const Eigen::VectorXd& grid, double eps)
{
    size_t best = 0;
    double bestDist = 1.0e300;

    for (Eigen::Index i = 0; i < grid.size(); i++) {
        const double d = std::abs(grid[i] - eps);
        if (d < bestDist) {
            bestDist = d;
            best = static_cast<size_t>(i);
        }
    }

    return best;
}

double debugSparseSum(const SparseMat& mat)
{
    double s = 0.0;

    for (int outer = 0; outer < mat.outerSize(); outer++) {
        for (SparseMat::InnerIterator it(mat, outer); it; ++it) {
            s += it.value();
        }
    }

    return s;
}
}

EEDFTwoTermApproximation::EEDFTwoTermApproximation(PlasmaPhase* s)
{
    // Store the PlasmaPhase context used by the solver (pointer to s).
    m_phase = s;
    m_first_call = true;
    m_has_EEDF = false;
    m_gamma = pow(2.0 * ElectronCharge / ElectronMass, 0.5);
}

void EEDFTwoTermApproximation::setLinearGrid(double& kTe_max, size_t& ncell)
{
    m_points = ncell;
    m_gridCenter.resize(m_points);
    m_gridEdge.resize(m_points + 1);
    m_f0.resize(m_points);
    m_f0_edge.resize(m_points + 1);
    for (size_t j = 0; j < m_points; j++) {
        m_gridCenter[j] = kTe_max * ( j + 0.5 ) / m_points;
        m_gridEdge[j] = kTe_max * j / m_points;
    }
    m_gridEdge[m_points] = kTe_max;
    setGridCache();
}

void EEDFTwoTermApproximation::setQuadraticGrid(double& kTe_max, size_t& ncell)
{
    m_points = ncell;

    m_gridCenter.resize(m_points);
    m_gridEdge.resize(m_points + 1);
    m_f0.resize(m_points);
    m_f0_edge.resize(m_points + 1);

    double n = static_cast<double>(m_points);

    for (size_t j = 0; j <= m_points; j++) {
        double x = static_cast<double>(j);
        m_gridEdge[j] = kTe_max * x * (x + 1.0) / (n * (n + 1.0));
    }

    for (size_t j = 0; j < m_points; j++) {
        m_gridCenter[j] = 0.5 * (m_gridEdge[j] + m_gridEdge[j + 1]);
    }

    setGridCache();
}

void EEDFTwoTermApproximation::setGeometricGrid(double& kTe_max, size_t& ncell, double ratio)
{
    m_points = ncell;

    m_gridCenter.resize(m_points);
    m_gridEdge.resize(m_points + 1);
    m_f0.resize(m_points);
    m_f0_edge.resize(m_points + 1);

    double denominator = std::pow(ratio, m_points) - 1.0;

    if (std::abs(denominator) < 1e-14) {
        throw CanteraError("EEDFTwoTermApproximation::setGeometricGrid",
            "Invalid geometric-grid ratio.");
    }

    for (size_t j = 0; j < m_points; j++) {
        m_gridEdge[j] = kTe_max * (std::pow(ratio, j) - 1.0) / denominator;

        m_gridCenter[j] = kTe_max
            * (std::pow(ratio, j + 0.5) - 1.0)
            / denominator;
    }

    m_gridEdge[m_points] = kTe_max;

    setGridCache();
}

void EEDFTwoTermApproximation::setCustomGrid(span<const double> levels)
{
    if (levels.size() < 2) {
        throw CanteraError("EEDFTwoTermApproximation::setCustomGrid",
            "Energy grid must contain at least two edge points.");
    }

    m_points = levels.size() - 1;

    m_gridCenter.resize(m_points);
    m_gridEdge.resize(m_points + 1);
    m_f0.resize(m_points);
    m_f0_edge.resize(m_points + 1);

    for (size_t j = 0; j < m_points + 1; j++) {
        if (!std::isfinite(levels[j])) {
            throw CanteraError("EEDFTwoTermApproximation::setCustomGrid",
                "Energy grid contains a non-finite value.");
        }
        if (levels[j] < 0.0) {
            throw CanteraError("EEDFTwoTermApproximation::setCustomGrid",
                "Energy grid values must be non-negative.");
        }
        if (j > 0 && levels[j] <= levels[j - 1]) {
            throw CanteraError("EEDFTwoTermApproximation::setCustomGrid",
                "Energy grid values must be strictly increasing.");
        }

        m_gridEdge[j] = levels[j];
    }

    for (size_t j = 0; j < m_points; j++) {
        m_gridCenter[j] = 0.5 * (m_gridEdge[j] + m_gridEdge[j + 1]);
    }

    setGridCache();
}

bool EEDFTwoTermApproximation::parameterChanged(double current, double previous, double rtol, double atol) const
{
    if (!std::isfinite(current) || !std::isfinite(previous)) {
        return true;
    }

    double diff = std::abs(current - previous);

    if (diff <= atol) {
        return false;
    }

    double scale = std::max(std::abs(current), std::abs(previous));

    if (scale == 0.0) {
        return diff > atol;
    }

    return diff / scale > rtol;
}

bool EEDFTwoTermApproximation::checkParamsVariation()
{
    if (!m_has_EEDF || !m_f0_ok) {
        m_f0_ok = false;
        return true;
    }

    double density = m_phase->density();
    double temperature = m_phase->temperature();
    double EN = m_phase->reducedElectricField();

    if (!std::isfinite(density) || density <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::checkParamsVariation",
            "Gas mass density must be finite and positive.");
    }

    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::checkParamsVariation",
            "Gas temperature must be finite and positive.");
    }

    if (!std::isfinite(EN) || EN < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::checkParamsVariation",
            "Reduced electric field must be finite and non-negative.");
    }

    if (parameterChanged(temperature, m_temperature_prev,
                         m_temperature_rtol, m_temperature_atol)) {
        m_f0_ok = false;
        return true;
    }

    if (parameterChanged(EN, m_EN_prev, m_EN_rtol, m_EN_atol)) {
        m_f0_ok = false;
        return true;
    }

    if (m_X_targets.size() != m_X_targets_prev.size()) {
        m_f0_ok = false;
        return true;
    }

    for (size_t k = 0; k < m_X_targets.size(); k++) {
        if (std::abs(m_X_targets[k] - m_X_targets_prev[k]) >= m_X_atol) {
            m_f0_ok = false;
            return true;
        }
    }

    if (m_X_superElasticSources.size() != m_X_superElasticSources_prev.size()) {
        m_f0_ok = false;
        return true;
    }

    for (size_t k = 0; k < m_X_superElasticSources.size(); k++) {
        if (std::abs(m_X_superElasticSources[k]
            - m_X_superElasticSources_prev[k]) >= m_X_atol) {
            m_f0_ok = false;
            return true;
        }
    }

    m_f0_ok = true;
    return false;
}

void EEDFTwoTermApproximation::storeCurrentParamsForEEDF()
{
    m_temperature_prev = m_phase->temperature();
    m_EN_prev = m_phase->reducedElectricField();

    m_X_targets_prev = m_X_targets;
    m_X_superElasticSources_prev = m_X_superElasticSources;
    m_f0_ok = true;
}

int EEDFTwoTermApproximation::calculateDistributionFunction()
{
    m_f0_computed_at_last_call = false;

    if (m_first_call) {
        initSpeciesIndexCrossSections();
        m_first_call = false;
        m_f0_ok = false;
    }

    updateMoleFractions();

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG CALC_START: T={}, density={}, molarDensity={}, "
            "EN={}, electricField={}, enableSE={}, applySE={}, "
            "has_EEDF={}, first_call={}, nCollisions={}, nInelastic={}, nElastic={}, "
            "nTargets={}, nSESources={}, gridPoints={}, gridEmax={}\n",
            m_phase->temperature(),
            m_phase->density(),
            m_phase->molarDensity(),
            m_phase->reducedElectricField(),
            m_phase->electricField(),
            m_enableSuperElasticCollisions,
            m_applySuperElasticCollisions,
            m_has_EEDF,
            m_first_call,
            m_phase->nCollisions(),
            m_phase->kInelastic().size(),
            m_phase->kElastic().size(),
            m_X_targets.size(),
            m_X_superElasticSources.size(),
            m_points,
            m_points > 0 ? m_gridEdge[m_points] : -1.0
        );
    }

    if (!checkParamsVariation()) {
        if (DEBUG_SE) {
            writelog("EEDF-SE-DEBUG CALC_SKIP: EEDF cache accepted, no recomputation.\n");
        }
        return 0;
    }

    m_f0_computed_at_last_call = true;

    checkSpeciesNoCrossSection();
    // Start from a cross-section state without super-elastic terms. If the
    // feature is enabled, solveBoltzmannWithSuperElastic() will first compute a
    // predictor EEDF, then activate super-elastic terms for the corrector solve.
    m_applySuperElasticCollisions = false;
    updateCrossSections();

    const double EN = m_phase->reducedElectricField();

    // Helper used to impose a Maxwellian EEDF.
    auto setMaxwellian = [&](double kTe_eV) {
        if (!std::isfinite(kTe_eV) || kTe_eV <= 0.0) {
            throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                "Invalid kTe value for Maxwellian first guess.");
        }

        const double fFloor = 1e-300;

        for (size_t j = 0; j < m_points; j++) {
            double arg = -m_gridCenter[j] / kTe_eV;

            if (arg < -700.0) {
                m_f0(j) = fFloor;
            } else {
                m_f0(j) = std::max(fFloor,
                    2.0 * std::sqrt(1.0 / Pi) * std::pow(kTe_eV, -1.5)
                    * std::exp(arg));
            }
        }

        double fnorm = norm(m_f0, m_gridCenter);

        if (!std::isfinite(fnorm) || fnorm <= 0.0) {
            throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                "Invalid norm for Maxwellian initialization.");
        }

        m_f0 /= fnorm;
    };

    // At very low reduced electric field, force a Maxwellian at the gas
    // temperature and skip the Boltzmann convergence to avoid wrong EEDF convergence and numerical instabilities 
    // when integrating over time.
    if (EN <= EN_min) {
        setMaxwellian(Boltzmann * m_phase->temperature() / ElectronCharge);
    } else {
        // At non-zero reduced electric field, use a hot Maxwellian first guess
        // only when no previously converged EEDF is available.
        if (!m_has_EEDF) {
            if (m_firstguess == "maxwell") {
                setMaxwellian(m_init_kTe);
            } else {
                throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                    "Unknown EEDF first guess '{}'.", m_firstguess);
            }
        }

        solveBoltzmannWithSuperElastic(m_f0);

        // Grid adaptation based on EEDF tail decay. If enabled, this will iteratively adjust the grid 
        // until the EEDF tail decays within the specified bounds given in the YAML file.
        // @todo Implement a more robust version which also varies the number of grid points if necessary. 
        // The current version only adjusts the maximum energy of the grid.

        if (m_adaptGrid) {
            const double fFloor = 1e-300;

            for (size_t n = 0; n < m_maxGridAdaptIterations; n++) {
                double fLeft = std::max(std::abs(m_f0(0)), fFloor);
                double fRight = std::max(std::abs(m_f0(m_points - 1)), fFloor);

                double decades = std::log10(fLeft) - std::log10(fRight);

                if (!std::isfinite(decades)) {
                    throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                        "Non-finite EEDF decay detected during grid adaptation.");
                }

                if (decades < m_minEedfDecay) {
                    // The right boundary is too low: the tail has not decayed enough.
                    double newMaxEnergy = m_kTeMax * (1.0 + m_gridUpdateFactor);

                    Eigen::VectorXd oldGridCenter = m_gridCenter;
                    Eigen::VectorXd oldF0 = m_f0;

                    updateGrid(newMaxEnergy);

                    if (m_maxwellianReset) {
                        if (m_firstguess == "maxwell") {
                            setMaxwellian(m_init_kTe);
                        } else {
                            throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                                "Unknown EEDF first guess '{}'.", m_firstguess);
                        }
                    } else {
                        writelog("Hey! Maxwellian reset is off!\n");
                        projectPreviousEEDFOnCurrentGrid(oldGridCenter, oldF0);
                    }

                    updateCrossSections();
                    solveBoltzmannWithSuperElastic(m_f0);

                } else if (decades > m_maxEedfDecay) {
                    // The right boundary is unnecessarily high.
                    double newMaxEnergy = m_kTeMax / (1.0 + m_gridUpdateFactor);

                    Eigen::VectorXd oldGridCenter = m_gridCenter;
                    Eigen::VectorXd oldF0 = m_f0;

                    updateGrid(newMaxEnergy);

                    if (m_maxwellianReset) {
                        if (m_firstguess == "maxwell") {
                            setMaxwellian(m_init_kTe);
                        } else {
                            throw CanteraError("EEDFTwoTermApproximation::calculateDistributionFunction",
                                "Unknown EEDF first guess '{}'.", m_firstguess);
                        }
                    } else {
                        writelog("Hey! Maxwellian reset is off!\n");
                        projectPreviousEEDFOnCurrentGrid(oldGridCenter, oldF0);
                    }

                    updateCrossSections();
                    solveBoltzmannWithSuperElastic(m_f0);

                }
            }
        }
    }

    // Write the EEDF at grid edges.
    vector<double> f(m_f0.data(), m_f0.data() + m_f0.rows() * m_f0.cols());
    vector<double> x(m_gridCenter.data(),
        m_gridCenter.data() + m_gridCenter.rows() * m_gridCenter.cols());

    for (size_t i = 0; i < m_points + 1; i++) {
        m_f0_edge[i] = linearInterp(m_gridEdge[i], x, f);
    }

    m_has_EEDF = true;

    // Update electron mobility.
    m_electronMobility = computeElectronMobility(m_f0);

    // Store the state for which the current EEDF is valid.
    storeCurrentParamsForEEDF();

    debugEedfSummary("final_center_f0", m_gridCenter, m_f0);
    return 0;
}

void EEDFTwoTermApproximation::converge(Eigen::VectorXd& f0)
{
    double err0 = 0.0;
    double err1 = 0.0;
    double delta = m_delta0;

    if (m_maxn == 0) {
        throw CanteraError("EEDFTwoTermApproximation::converge",
                           "m_maxn is zero; no iterations will occur.");
    }
    if (m_points == 0) {
        throw CanteraError("EEDFTwoTermApproximation::converge",
                           "m_points is zero; the EEDF grid is empty.");
    }
    if (isnan(delta) || delta == 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::converge",
                           "m_delta0 is NaN or zero; solver cannot update.");
    }

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG CONVERGE_START: applySE={}, enableSE={}, "
            "maxn={}, rtol={}, delta0={}, norm_initial={}\n",
            m_applySuperElasticCollisions,
            m_enableSuperElasticCollisions,
            m_maxn,
            m_rtol,
            m_delta0,
            norm(f0, m_gridCenter)
        );
    }

    for (size_t n = 0; n < m_maxn; n++) {
        if (0.0 < err1 && err1 < err0) {
            delta *= log(m_factorM) / (log(err0) - log(err1));
        }

        Eigen::VectorXd f0_old = f0;
        f0 = iterate(f0_old, delta);
        checkFinite("EEDFTwoTermApproximation::converge: f0", asSpan(f0));

        err0 = err1;
        Eigen::VectorXd Df0 = (f0_old - f0).cwiseAbs();
        err1 = norm(Df0, m_gridCenter);
        if (DEBUG_SE && (n < 5 || err1 < 10.0 * m_rtol || n + 1 == m_maxn)) {
            writelog(
                "EEDF-SE-DEBUG CONVERGE_ITER: applySE={}, iter={}, delta={}, "
                "err={}, norm_f0={}, f0_first={}, f0_last={}\n",
                m_applySuperElasticCollisions,
                n,
                delta,
                err1,
                norm(f0, m_gridCenter),
                f0.size() > 0 ? f0[0] : -1.0,
                f0.size() > 0 ? f0[f0.size() - 1] : -1.0
            );
        }
        const size_t minIterations = m_applySuperElasticCollisions ? m_minSuperElasticIterations : 0;
        if (err1 < m_rtol && n >= minIterations) {
            if (DEBUG_SE) {
                writelog(
                    "EEDF-SE-DEBUG CONVERGE_DONE: applySE={}, iter={}, err={}, norm_f0={}\n",
                    m_applySuperElasticCollisions,
                    n,
                    err1,
                    norm(f0, m_gridCenter)
                );
            }
            break;
        } else if (n == m_maxn - 1) {
            throw CanteraError("EEDFTwoTermApproximation::converge", "Convergence failed");
        }
        if (DEBUG_SE && m_applySuperElasticCollisions) {
            debugEedfSummary("corrector_iter_withSE", m_gridCenter, f0);
        }
    }
}

void EEDFTwoTermApproximation::solveBoltzmannWithSuperElastic(Eigen::VectorXd& f0)
{
    // Predictor: solve without super-elastic collisions. This gives the
    // forward electron-impact production rates used to partition lumped
    // excited species.
    m_applySuperElasticCollisions = false;
    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG SOLVE_SE_START: enableSE={}, nSEFlags={}, nSESources={}\n",
            m_enableSuperElasticCollisions,
            m_hasSuperElastic.size(),
            m_k_lg_SuperElasticSources.size()
        );
    }
    updateCrossSections();
    converge(f0);
    debugEedfSummary("after_predictor_noSE", m_gridCenter, f0);

    if (!m_enableSuperElasticCollisions) {
        if (DEBUG_SE) {
            writelog("EEDF-SE-DEBUG SOLVE_SE_EXIT: super-elastic disabled.\n");
        }
        return;
    }

    bool hasAnySuperElastic = false;
    for (bool enabled : m_hasSuperElastic) {
        if (enabled) {
            hasAnySuperElastic = true;
            break;
        }
    }

    if (!hasAnySuperElastic) {
        if (DEBUG_SE) {
            writelog("EEDF-SE-DEBUG SOLVE_SE_EXIT: no active super-elastic channel.\n");
        }
        return;
    }

    // Estimate source partition weights from the predictor EEDF:
    //
    //   w_i = P_i / sum_j P_j
    //
    // where P_i = X_target_i * K_i and j runs over all channels mapped to the
    // same lumped source species.
    updateSuperElasticWeights(f0);

    if (DEBUG_SE) {
        for (size_t k : m_phase->kInelastic()) {
            if (k < m_hasSuperElastic.size() && m_hasSuperElastic[k]) {
                const size_t loc = m_superElasticSourceLoc[k];
                auto rate = m_phase->collisionRate(k);

                writelog(
                    "EEDF-SE-DEBUG SOLVE_SE_WEIGHT_SUMMARY: k={}, collision='{}', "
                    "source='{}', X_source={}, weight={}, prefactor={}, "
                    "target='{}', X_target_norm={}, degeneracyRatio={}, threshold={}\n",
                    k,
                    rate->collisionName(),
                    m_phase->speciesName(m_superElasticSourceSpecies[k]),
                    m_X_superElasticSources[loc],
                    m_superElasticWeights[k],
                    m_X_superElasticSources[loc] * m_superElasticWeights[k],
                    m_phase->speciesName(m_phase->targetIndex(k)),
                    m_X_targets[m_klocTargets[k]],
                    rate->superElasticDegeneracyRatio(),
                    rate->threshold()
                );
            }
        }
    }

    // Corrector: solve with super-elastic scattering-in/out terms included.
    m_applySuperElasticCollisions = true;
    updateCrossSections();
    converge(f0);
    debugEedfSummary("after_corrector_withSE", m_gridCenter, f0);
}

Eigen::VectorXd EEDFTwoTermApproximation::iterate(const Eigen::VectorXd& f0,
                                                  double delta)
{
    // CQM multiple call to vector_* and matrix_*
    // probably extremely inefficient
    // must be refactored!!

    SparseMat PQ(m_points, m_points);
    vector<double> g = vector_g(f0);

    for (size_t k : m_phase->kInelastic()) {
        SparseMat Q_k = matrix_Q(g, k);
        SparseMat P_k = matrix_P(g, k);
        PQ += (Q_k - P_k) * m_X_targets[m_klocTargets[k]];

        if (m_applySuperElasticCollisions &&
            k < m_hasSuperElastic.size() &&
            m_hasSuperElastic[k]) {
            const size_t loc = m_superElasticSourceLoc[k];
            const double X_source = m_X_superElasticSources[loc];
            const double weight = m_superElasticWeights[k];
            const double prefactor = X_source * weight;

            if (prefactor > 0.0) {
                SparseMat Qse_k = matrix_Q_superelastic(g, k);
                SparseMat Pse_k = matrix_P_superelastic(g, k);

                if (DEBUG_SE && DEBUG_SE_ITER) {
                    auto rate = m_phase->collisionRate(k);

                    const double X_target = m_X_targets[m_klocTargets[k]];
                    const double betaPopulation =
                        X_target > 0.0 ? prefactor / X_target : -1.0;
                    const double betaWithDegeneracy =
                        betaPopulation * rate->superElasticDegeneracyRatio();

                    writelog(
                        "EEDF-SE-DEBUG ITER_SE_OPERATOR: k={}, collision='{}', "
                        "target='{}', source='{}', threshold={}, degeneracyRatio={}, "
                        "X_target_norm={}, X_source={}, weight={}, prefactor={}, "
                        "beta_population=prefactor/Xtarget={}, beta_with_g={}, "
                        "Pse_nnz={}, Qse_nnz={}, Pse_sum={}, Qse_sum={}, "
                        "Pse_pref_sum={}, Qse_pref_sum={}\n",
                        k,
                        rate->collisionName(),
                        m_phase->speciesName(m_phase->targetIndex(k)),
                        m_phase->speciesName(m_superElasticSourceSpecies[k]),
                        rate->threshold(),
                        rate->superElasticDegeneracyRatio(),
                        X_target,
                        X_source,
                        weight,
                        prefactor,
                        betaPopulation,
                        betaWithDegeneracy,
                        Pse_k.nonZeros(),
                        Qse_k.nonZeros(),
                        debugSparseSum(Pse_k),
                        debugSparseSum(Qse_k),
                        prefactor * debugSparseSum(Pse_k),
                        prefactor * debugSparseSum(Qse_k)
                    );
                }

                PQ += (Qse_k - Pse_k) * prefactor;
            }
        }
    }

    SparseMat A = matrix_A(f0);
    SparseMat I(m_points, m_points);
    for (size_t i = 0; i < m_points; i++) {
        I.insert(i, i) = 1.0;
    }

    A -= PQ;
    A *= delta;
    A += I;

    Eigen::SparseLU<SparseMat> solver(A);

    if (solver.info() == Eigen::NumericalIssue) {
        throw CanteraError("EEDFTwoTermApproximation::iterate",
            "Error SparseLU solver: NumericalIssue");
    } else if (solver.info() == Eigen::InvalidInput) {
        throw CanteraError("EEDFTwoTermApproximation::iterate",
            "Error SparseLU solver: InvalidInput");
    }

    if (solver.info() != Eigen::Success) {
        throw CanteraError("EEDFTwoTermApproximation::iterate",
            "Error SparseLU solver", "Decomposition failed");
    }

    Eigen::VectorXd f1 = solver.solve(f0);

    if (solver.info() != Eigen::Success) {
        throw CanteraError("EEDFTwoTermApproximation::iterate",
            "Solving failed");
    }

    checkFinite("EEDFTwoTermApproximation::converge: f0", asSpan(f1));
    f1 /= norm(f1, m_gridCenter);

    return f1;
}

double EEDFTwoTermApproximation::integralPQ(double a, double b, double u0, double u1,
                                            double g, double x0)
{
    double A1;
    double A2;
    if (g != 0.0) {
        double expm1a = expm1(g * (-a + x0));
        double expm1b = expm1(g * (-b + x0));
        double ag = a * g;
        double ag1 = ag + 1;
        double bg = b * g;
        double bg1 = bg + 1;
        A1 = (expm1a * ag1 + ag - expm1b * bg1 - bg) / (g*g);
        A2 = (expm1a * (2 * ag1 + ag * ag) + ag * (ag + 2) -
              expm1b * (2 * bg1 + bg * bg) - bg * (bg + 2)) / (g*g*g);
    } else {
        A1 = 0.5 * (b*b - a*a);
        A2 = 1.0 / 3.0 * (b*b*b - a*a*a);
    }

    // The interpolation formula of u(x) = c0 + c1 * x
    double c0 = (a * u1 - b * u0) / (a - b);
    double c1 = (u0 - u1) / (a - b);

    return c0 * A1 + c1 * A2;
}

double EEDFTwoTermApproximation::integralPQWeighted(
    double a, double b, double w0, double w1, double g, double x0)
{
    if (b <= a) {
        return 0.0;
    }

    double I0;
    double I1;

    if (g != 0.0) {
        const double Ea = std::exp(g * (-a + x0));
        const double Eb = std::exp(g * (-b + x0));
        I0 = (Ea - Eb) / g;

        const double expm1a = std::expm1(g * (-a + x0));
        const double expm1b = std::expm1(g * (-b + x0));
        const double ag = a * g;
        const double bg = b * g;

        I1 = (expm1a * (ag + 1.0) + ag
            - expm1b * (bg + 1.0) - bg) / (g * g);
    } else {
        I0 = b - a;
        I1 = 0.5 * (b * b - a * a);
    }

    // Linear interpolation of w(x) = c0 + c1 x.
    const double c0 = (a * w1 - b * w0) / (a - b);
    const double c1 = (w0 - w1) / (a - b);

    return c0 * I0 + c1 * I1;
}

vector<double> EEDFTwoTermApproximation::vector_g(const Eigen::VectorXd& f0)
{
    vector<double> g(m_points, 0.0);
    const double f_min = 1e-300;  // Smallest safe floating-point value

    // Handle first point (i = 0)
    double f1 = std::max(f0(1), f_min);
    double f0_ = std::max(f0(0), f_min);
    g[0] = log(f1 / f0_) / (m_gridCenter[1] - m_gridCenter[0]);

    // Handle last point (i = N)
    size_t N = m_points - 1;
    double fN = std::max(f0(N), f_min);
    double fNm1 = std::max(f0(N - 1), f_min);
    g[N] = log(fN / fNm1) / (m_gridCenter[N] - m_gridCenter[N - 1]);

    // Handle interior points
    for (size_t i = 1; i < N; ++i) {
        double f_up   = std::max(f0(i + 1), f_min);
        double f_down = std::max(f0(i - 1), f_min);
        g[i] = log(f_up / f_down) / (m_gridCenter[i + 1] - m_gridCenter[i - 1]);
    }
    return g;
}

SparseMat EEDFTwoTermApproximation::matrix_P(span<const double> g, size_t k)
{
    SparseTriplets tripletList;
    for (size_t n = 0; n < m_eps[k].size(); n++) {
        double eps_a = m_eps[k][n][0];
        double eps_b = m_eps[k][n][1];
        double sigma_a = m_sigma[k][n][0];
        double sigma_b = m_sigma[k][n][1];
        auto j = static_cast<SparseMat::StorageIndex>(m_j[k][n]);
        double r = integralPQ(eps_a, eps_b, sigma_a, sigma_b, g[j], m_gridCenter[j]);
        double p = m_gamma * r;

        tripletList.emplace_back(j, j, p);
    }
    SparseMat P(m_points, m_points);
    P.setFromTriplets(tripletList.begin(), tripletList.end());
    return P;
}

SparseMat EEDFTwoTermApproximation::matrix_Q(span<const double> g, size_t k)
{
    SparseTriplets tripletList;
    for (size_t n = 0; n < m_eps[k].size(); n++) {
        double eps_a = m_eps[k][n][0];
        double eps_b = m_eps[k][n][1];
        double sigma_a = m_sigma[k][n][0];
        double sigma_b = m_sigma[k][n][1];
        auto i = static_cast<SparseMat::StorageIndex>(m_i[k][n]);
        auto j = static_cast<SparseMat::StorageIndex>(m_j[k][n]);
        double r = integralPQ(eps_a, eps_b, sigma_a, sigma_b, g[j], m_gridCenter[j]);
        double q = m_inFactor[k] * m_gamma * r;

        tripletList.emplace_back(i, j, q);
    }
    SparseMat Q(m_points, m_points);
    Q.setFromTriplets(tripletList.begin(), tripletList.end());
    return Q;
}

SparseMat EEDFTwoTermApproximation::matrix_P_superelastic(
    span<const double> g, size_t k)
{
    SparseTriplets tripletList;

    for (size_t n = 0; n < m_epsSuperElastic[k].size(); n++) {
        const double eps_a = m_epsSuperElastic[k][n][0];
        const double eps_b = m_epsSuperElastic[k][n][1];

        const double w_a = m_epsSigmaSuperElastic[k][n][0];
        const double w_b = m_epsSigmaSuperElastic[k][n][1];

        const auto j =
            static_cast<SparseMat::StorageIndex>(m_jSuperElastic[k][n]);

        const double r = integralPQWeighted(
            eps_a, eps_b, w_a, w_b, g[j], m_gridCenter[j]);

        tripletList.emplace_back(j, j, m_gamma * r);
    }

    SparseMat P(m_points, m_points);
    P.setFromTriplets(tripletList.begin(), tripletList.end());
    return P;
}

SparseMat EEDFTwoTermApproximation::matrix_Q_superelastic(
    span<const double> g, size_t k)
{
    SparseTriplets tripletList;

    for (size_t n = 0; n < m_epsSuperElastic[k].size(); n++) {
        const double eps_a = m_epsSuperElastic[k][n][0];
        const double eps_b = m_epsSuperElastic[k][n][1];

        const double w_a = m_epsSigmaSuperElastic[k][n][0];
        const double w_b = m_epsSigmaSuperElastic[k][n][1];

        const auto i =
            static_cast<SparseMat::StorageIndex>(m_iSuperElastic[k][n]);
        const auto j =
            static_cast<SparseMat::StorageIndex>(m_jSuperElastic[k][n]);

        const double r = integralPQWeighted(
            eps_a, eps_b, w_a, w_b, g[j], m_gridCenter[j]);

        // Super-elastic collisions conserve the number of electrons:
        // one electron before, one electron after. Therefore no ionization /
        // attachment multiplicity factor is applied here.
        tripletList.emplace_back(i, j, m_gamma * r);
    }

    SparseMat Q(m_points, m_points);
    Q.setFromTriplets(tripletList.begin(), tripletList.end());
    return Q;
}

SparseMat EEDFTwoTermApproximation::matrix_A(const Eigen::VectorXd& f0)
{
    vector<double> a0(m_points + 1);
    vector<double> a1(m_points + 1);
    size_t N = m_points - 1;
    // Scharfetter-Gummel scheme
    double nu = netProductionFrequency(f0);
    a0[0] = NAN;
    a1[0] = NAN;
    a0[N+1] = NAN;
    a1[N+1] = NAN;

    // Electron-electron collisions declarations
    double a = 0.0;
    vector<double> A1(m_points + 1, 0.0);
    vector<double> A2(m_points + 1, 0.0);
    vector<double> A3(m_points + 1, 0.0);

    if (m_eeCol) {
        eeColIntegrals(f0, A1, A2, A3, a);
    }

    double nDensity = m_phase->molarDensity() * Avogadro;
    double alpha;
    double E = m_phase->electricField();
    if (m_growth == "spatial") {
        double mu = computeElectronMobility(f0);
        double D = electronDiffusivity(f0);
        alpha = (mu * E - sqrt(pow(mu * E, 2) - 4 * D * nu * nDensity)) / 2.0 / D / nDensity;
    } else {
        alpha = 0.0;
    }

    double sigma_tilde;
    double omega = 2 * Pi * m_phase->electricFieldFrequency();
    for (size_t j = 1; j < m_points; j++) {
        if (m_growth == "temporal") {
            sigma_tilde = m_totalCrossSectionEdge[j] + nu / pow(m_gridEdge[j], 0.5) / m_gamma;
        } else {
            sigma_tilde = m_totalCrossSectionEdge[j];
        }
        double q = omega / (nDensity * m_gamma * pow(m_gridEdge[j], 0.5));
        double W = -m_gamma * m_gridEdge[j] * m_gridEdge[j] * m_sigmaElastic[j];
        double F = sigma_tilde * sigma_tilde / (sigma_tilde * sigma_tilde + q * q);
        double DA = m_gamma / 3.0 * pow(E / nDensity, 2.0) * m_gridEdge[j];
        double DB = m_gamma * m_phase->temperature() * Boltzmann / ElectronCharge * m_gridEdge[j] * m_gridEdge[j] * m_sigmaElastic[j];
        double D = DA / sigma_tilde * F + DB;
        if (m_growth == "spatial") {
            W -= m_gamma / 3.0 * 2 * alpha * E / nDensity * m_gridEdge[j] / sigma_tilde;
        }

        if (m_eeCol) {
            W -= 3 * a * m_ionDegree * A1[j];
            D += 2 * a * m_ionDegree * (A2[j] + pow(m_gridEdge[j], 1.5) * A3[j]);
        }

        double z = W * (m_gridCenter[j] - m_gridCenter[j-1]) / D;
        if (!std::isfinite(z)) {
            throw CanteraError("matrix_A", "Non-finite Peclet number encountered");
        }
        if (std::abs(z) > 500) {
            warn_user("EEDFTwoTermApproximation::matrix_A",
                "Large Peclet number z = {:.3e} at j = {}. "
                "W = {:.3e}, D = {:.3e}, E/N = {:.3e}\n",
                z, j, W, D, E / nDensity);
        }
        a0[j] = W / (1 - std::exp(-z));
        a1[j] = W / (1 - std::exp(z));
    }

    SparseTriplets tripletList;
    // center diagonal
    // zero flux b.c. at energy = 0
    tripletList.emplace_back(0, 0, a0[1]);

    for (size_t j = 1; j < m_points - 1; j++) {
        tripletList.emplace_back(j, j, a0[j+1] - a1[j]);
    }

    // upper diagonal
    for (size_t j = 0; j < m_points - 1; j++) {
        tripletList.emplace_back(j, j+1, a1[j+1]);
    }

    // lower diagonal
    for (size_t j = 1; j < m_points; j++) {
        tripletList.emplace_back(j, j-1, -a0[j]);
    }

    // zero flux b.c.
    tripletList.emplace_back(N, N, -a1[N]);

    SparseMat A(m_points, m_points);
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // plus G
    SparseMat G(m_points, m_points);
    if (m_growth == "temporal") {
        for (size_t i = 0; i < m_points; i++) {
            G.insert(i, i) = 2.0 / 3.0 * (pow(m_gridEdge[i+1], 1.5) - pow(m_gridEdge[i], 1.5)) * nu;
        }
    } else if (m_growth == "spatial") {
        double nDensity = m_phase->molarDensity() * Avogadro;
        for (size_t i = 0; i < m_points; i++) {
            double sigma_c = 0.5 * (m_totalCrossSectionEdge[i] + m_totalCrossSectionEdge[i + 1]);
            G.insert(i, i) = - alpha * m_gamma / 3 * (alpha * (pow(m_gridEdge[i + 1], 2) - pow(m_gridEdge[i], 2)) / sigma_c / 2
                 - E / nDensity * (m_gridEdge[i + 1] / m_totalCrossSectionEdge[i + 1] - m_gridEdge[i] / m_totalCrossSectionEdge[i]));
        }
    }
    return A + G;
}

// These cumulative integrals are evaluated using the composite trapezoidal rule.
// Calling numericalQuadrature independently at each grid edge would require
// O(N^2) work and repeated temporary allocations, whereas the cumulative formulation used here
// gives the same composite-trapezoidal partial integrals in O(N): much faster on grids with tipically 100 to 
// 1000 points that we use to solve the EEDF.
void EEDFTwoTermApproximation::eeColIntegrals(const Eigen::VectorXd& f0, vector<double>& A1, vector<double>& A2, vector<double>& A3, double& a) const
{
    const size_t n = m_points;

    if (static_cast<size_t>(f0.size()) != n) {
        throw CanteraError("EEDFTwoTermApproximation::eeColIntegrals",
            "Inconsistent EEDF size.");
    }

    if (!m_eeCol) {
        A1.assign(n + 1, 0.0);
        A2.assign(n + 1, 0.0);
        A3.assign(n + 1, 0.0);
        a = 0.0;
        return;
    }

    if (!std::isfinite(m_nElectron) || m_nElectron <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::eeColIntegrals",
            "Electron number density must be finite and positive "
            "when electron-electron collisions are enabled.");
    }

    vector<double> f0Edge(n + 1);

    f0Edge[0] = f0(0);
    for (size_t i = 1; i < n; i++) {
        f0Edge[i] = 0.5 * (f0(i - 1) + f0(i));
    }
    f0Edge[n] = f0(n - 1);

    A1.assign(n + 1, 0.0);
    A2.assign(n + 1, 0.0);
    A3.assign(n + 1, 0.0);

    // A1[j] = int_0^eps_j sqrt(eps) f0(eps) d eps
    for (size_t j = 1; j <= n; j++) {
        const double dx = m_gridEdge[j] - m_gridEdge[j - 1];

        const double yLeft =
            std::sqrt(m_gridEdge[j - 1]) * f0Edge[j - 1];
        const double yRight =
            std::sqrt(m_gridEdge[j]) * f0Edge[j];

        A1[j] = A1[j - 1] + 0.5 * dx * (yLeft + yRight);
    }

    // A2[j] = int_0^eps_j eps^(3/2) f0(eps) d eps
    for (size_t j = 1; j <= n; j++) {
        const double dx = m_gridEdge[j] - m_gridEdge[j - 1];

        const double yLeft =
            std::pow(m_gridEdge[j - 1], 1.5) * f0Edge[j - 1];
        const double yRight =
            std::pow(m_gridEdge[j], 1.5) * f0Edge[j];

        A2[j] = A2[j - 1] + 0.5 * dx * (yLeft + yRight);
    }

    // A3[j] = int_eps_j^eps_max f0(eps) d eps
    for (size_t j = n; j > 0; j--) {
        const double dx = m_gridEdge[j] - m_gridEdge[j - 1];

        A3[j - 1] = A3[j]
            + 0.5 * dx * (f0Edge[j - 1] + f0Edge[j]);
    }

    const double kTe = 2.0 / 3.0 * ElectronCharge * A2[n];

    if (!std::isfinite(kTe) || kTe <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::eeColIntegrals",
            "Invalid electron thermal energy for Coulomb logarithm.");
    }

    const double coulombParam =
        12.0 * Pi * std::pow(epsilon_0 * kTe, 1.5)
        / (std::pow(ElectronCharge, 3) * std::sqrt(m_nElectron));

    if (!std::isfinite(coulombParam) || coulombParam <= 1.0) {
        throw CanteraError("EEDFTwoTermApproximation::eeColIntegrals",
            "Invalid Coulomb logarithm argument: {}.", coulombParam);
    }

    a = std::pow(ElectronCharge, 2) * m_gamma
        / (24.0 * Pi * std::pow(epsilon_0, 2))
        * std::log(coulombParam);
}

double EEDFTwoTermApproximation::netProductionFrequency(const Eigen::VectorXd& f0)
{
    double nu = 0.0;
    vector<double> g = vector_g(f0);

    for (size_t k = 0; k < m_phase->nCollisions(); k++) {
        if (m_phase->collisionRate(k)->kind() == "ionization" ||
            m_phase->collisionRate(k)->kind() == "attachment") {
            SparseMat PQ = (matrix_Q(g, k) - matrix_P(g, k)) *
                              m_X_targets[m_klocTargets[k]];
            Eigen::VectorXd s = PQ * f0;
            checkFinite("EEDFTwoTermApproximation::netProductionFrequency: s",
                        asSpan(s));
            nu += s.sum();
        }
    }
    return nu;
}

double EEDFTwoTermApproximation::electronDiffusivity(const Eigen::VectorXd& f0)
{
    vector<double> y(m_points, 0.0);
    double nu = netProductionFrequency(f0);

    for (size_t i = 0; i < m_points; i++) {
        double sigma_tilde = m_totalCrossSectionCenter[i];

        if (m_growth == "temporal") {
            sigma_tilde += nu / m_gamma / pow(m_gridCenter[i], 0.5);
        }

        y[i] = m_gridCenter[i] * f0(i) / sigma_tilde;
    }

    double nDensity = m_phase->molarDensity() * Avogadro;
    auto f = Eigen::Map<const Eigen::ArrayXd>(y.data(), y.size());
    auto x = Eigen::Map<const Eigen::ArrayXd>(
        m_gridCenter.data(), m_gridCenter.size());

    return 1.0 / 3.0 * m_gamma * simpson(f, x) / nDensity;
}

double EEDFTwoTermApproximation::computeElectronMobility(const Eigen::VectorXd& f0)
{
    double nu = netProductionFrequency(f0);
    vector<double> y(m_points + 1, 0.0);

    for (size_t i = 1; i < m_points; i++) {
        double df0 = (f0(i) - f0(i - 1))
            / (m_gridCenter[i] - m_gridCenter[i - 1]);

        double sigma_tilde = m_totalCrossSectionEdge[i];

        if (m_growth == "temporal") {
            sigma_tilde += nu / m_gamma / pow(m_gridEdge[i], 0.5);
        }

        y[i] = m_gridEdge[i] * df0 / sigma_tilde;
    }

    double nDensity = m_phase->molarDensity() * Avogadro;
    auto f = ConstMappedVector(y.data(), y.size());
    auto x = ConstMappedVector(m_gridEdge.data(), m_gridEdge.size());

    return -1.0 / 3.0 * m_gamma * simpson(f, x) / nDensity;
}

double EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF(
    const Eigen::VectorXd& f0, size_t k)
{
    auto rate = m_phase->collisionRate(k);
    auto x = rate->energyLevels();
    auto y = rate->crossSections();

    vector<double> integrand(m_points, 0.0);

    for (size_t i = 0; i < m_points; i++) {
        const double eps = m_gridCenter[i];
        const double sigma =
            linearInterpCrossSectionZeroOutside(eps, x, y);
        integrand[i] = eps * sigma * f0(i);
    }

    auto f = ConstMappedVector(integrand.data(), integrand.size());
    auto epsGrid = ConstMappedVector(m_gridCenter.data(), m_gridCenter.size());

    return m_gamma * simpson(f, epsGrid);
}

void EEDFTwoTermApproximation::updateSuperElasticWeights(
    const Eigen::VectorXd& f0)
{
    m_superElasticWeights.assign(m_phase->nCollisions(), 0.0);

    vector<double> production(m_phase->nCollisions(), 0.0);
    vector<double> productionSum(m_k_lg_SuperElasticSources.size(), 0.0);
    vector<size_t> channelCount(m_k_lg_SuperElasticSources.size(), 0);

    for (size_t k : m_phase->kInelastic()) {
        if (k >= m_hasSuperElastic.size() || !m_hasSuperElastic[k]) {
            continue;
        }

        const size_t sourceLoc = m_superElasticSourceLoc[k];

        const double K = forwardRateCoefficientFromEEDF(f0, k);
        if (!std::isfinite(K) || K < 0.0) {
            throw CanteraError("EEDFTwoTermApproximation::updateSuperElasticWeights",
                "Invalid forward rate coefficient for collision '{}'.",
                m_phase->collisionRate(k)->collisionName());
        }

        // Reduced production flux. The electron density is common to all
        // electron-impact channels and cancels in the partition ratio.
        production[k] = m_X_targets[m_klocTargets[k]] * K;
        productionSum[sourceLoc] += production[k];
        channelCount[sourceLoc]++;
        if (DEBUG_SE) {
            auto rate = m_phase->collisionRate(k);

            writelog(
                "EEDF-SE-DEBUG WEIGHT_RAW: k={}, collision='{}', "
                "target='{}', source='{}', K_forward={}, X_target_norm={}, "
                "production={}, sourceLoc={}, productionSum_now={}, channelCount_now={}\n",
                k,
                rate->collisionName(),
                m_phase->speciesName(m_phase->targetIndex(k)),
                m_phase->speciesName(m_superElasticSourceSpecies[k]),
                K,
                m_X_targets[m_klocTargets[k]],
                production[k],
                sourceLoc,
                productionSum[sourceLoc],
                channelCount[sourceLoc]
            );
        }
    }

    for (size_t k : m_phase->kInelastic()) {
        if (k >= m_hasSuperElastic.size() || !m_hasSuperElastic[k]) {
            continue;
        }

        const size_t sourceLoc = m_superElasticSourceLoc[k];

        if (productionSum[sourceLoc] > 0.0) {
            m_superElasticWeights[k] = production[k] / productionSum[sourceLoc];
        } else if (channelCount[sourceLoc] == 1) {
            // Species-resolved perfect case: there is no ambiguity in the
            // partition, even if the predictor EEDF gives a vanishing forward
            // production rate.
            m_superElasticWeights[k] = 1.0;
        } else {
            // Ambiguous lumped case with no forward production signal. Do not
            // invent an equal partition silently.
            m_superElasticWeights[k] = 0.0;
        }
        if (DEBUG_SE) {
            auto rate = m_phase->collisionRate(k);

            writelog(
                "EEDF-SE-DEBUG WEIGHT_FINAL: k={}, collision='{}', source='{}', "
                "production={}, productionSum={}, channelCount={}, weight={}\n",
                k,
                rate->collisionName(),
                m_phase->speciesName(m_superElasticSourceSpecies[k]),
                production[k],
                productionSum[sourceLoc],
                channelCount[sourceLoc],
                m_superElasticWeights[k]
            );
        }
    }
}

void EEDFTwoTermApproximation::initSpeciesIndexCrossSections()
{
    // Reset all indexing containers. This makes the function safe to call again
    // if the user toggles EEDF model options after construction.
    m_kTargets.clear();
    m_klocTargets.clear();
    m_kOthers.clear();
    m_k_lg_Targets.clear();

    m_kTargets.resize(m_phase->nCollisions());
    m_klocTargets.resize(m_phase->nCollisions());
    m_inFactor.resize(m_phase->nCollisions());

    for (size_t k = 0; k < m_phase->nCollisions(); k++) {
        m_kTargets[k] = m_phase->targetIndex(k);

        auto it = std::find(m_k_lg_Targets.begin(),
                            m_k_lg_Targets.end(),
                            m_kTargets[k]);

        if (it == m_k_lg_Targets.end()) {
            m_k_lg_Targets.push_back(m_kTargets[k]);
            m_klocTargets[k] = m_k_lg_Targets.size() - 1;
        } else {
            m_klocTargets[k] =
                static_cast<size_t>(std::distance(m_k_lg_Targets.begin(), it));
        }

        const auto& kind = m_phase->collisionRate(k)->kind();

        if (kind == "ionization") {
            if (m_growth == "none"){
                m_inFactor[k] = 1;
            }
            else {
                m_inFactor[k] = 2;
            }
            
        } else if (kind == "attachment") {
            m_inFactor[k] = 0;
        } else {
            m_inFactor[k] = 1;
        }
        if (DEBUG_SE) {
            auto rate = m_phase->collisionRate(k);

            writelog(
                "EEDF-SE-DEBUG TARGET_MAP: k={}, collision='{}', equation='{}', "
                "kind='{}', target='{}', targetIndex={}, targetLoc={}, "
                "inFactor={}, threshold={}, degeneracyRatio={}, "
                "correspondingSpecies='{}', product='{}'\n",
                k,
                rate->collisionName(),
                m_phase->collision(k)->equation(),
                rate->kind(),
                m_phase->speciesName(m_kTargets[k]),
                m_kTargets[k],
                m_klocTargets[k],
                m_inFactor[k],
                rate->threshold(),
                rate->superElasticDegeneracyRatio(),
                rate->correspondingSpecies(),
                rate->product()
            );
        }
    }

    m_X_targets.resize(m_k_lg_Targets.size());
    m_X_targets_prev.resize(m_k_lg_Targets.size());

    for (size_t k = 0; k < m_X_targets.size(); k++) {
        const size_t k_glob = m_k_lg_Targets[k];
        m_X_targets[k] = m_phase->moleFraction(k_glob);
        m_X_targets_prev[k] = m_phase->moleFraction(k_glob);
    }

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG TARGET_SUMMARY: nGlobalTargets={}, nCollisions={}\n",
            m_k_lg_Targets.size(),
            m_phase->nCollisions()
        );

        for (size_t k = 0; k < m_k_lg_Targets.size(); k++) {
            writelog(
                "EEDF-SE-DEBUG TARGET_SPECIES: loc={}, species='{}', speciesIndex={}, rawMoleFraction={}\n",
                k,
                m_phase->speciesName(m_k_lg_Targets[k]),
                m_k_lg_Targets[k],
                m_phase->moleFraction(m_k_lg_Targets[k])
            );
        }
    }

    initSuperElasticChannels();

    // Species without forward electron-collision cross-section data.
    // Super-elastic source species are not inserted in m_kTargets, otherwise
    // they would change the normalization of the forward target fractions.
    for (size_t k = 0; k < m_phase->nSpecies(); k++) {
        auto it = std::find(m_kTargets.begin(), m_kTargets.end(), k);
        if (it == m_kTargets.end()) {
            m_kOthers.push_back(k);
        }
    }
}

size_t EEDFTwoTermApproximation::addSuperElasticSourceSpecies(size_t kSource)
{
    auto it = std::find(m_k_lg_SuperElasticSources.begin(),
                        m_k_lg_SuperElasticSources.end(),
                        kSource);

    if (it == m_k_lg_SuperElasticSources.end()) {
        m_k_lg_SuperElasticSources.push_back(kSource);
        return m_k_lg_SuperElasticSources.size() - 1;
    }

    return static_cast<size_t>(
        std::distance(m_k_lg_SuperElasticSources.begin(), it));
}

size_t EEDFTwoTermApproximation::inferSuperElasticSourceSpecies(size_t k) const
{
    auto rate = m_phase->collisionRate(k);
    const string electronName = m_phase->electronSpeciesName();
    const size_t kTarget = m_phase->targetIndex(k);
    const string targetName = m_phase->speciesName(kTarget);

    if (rate->kind() != "excitation") {
        return npos;
    }

    // 1. Explicit override in the cross-section entry.
    if (!rate->correspondingSpecies().empty()) {
        const size_t kSource = m_phase->speciesIndex(rate->correspondingSpecies(), true);

        if (DEBUG_SE) {
            writelog(
                "EEDF-SE-DEBUG INFER_SOURCE: k={}, collision='{}', method='corresponding-species', "
                "correspondingSpecies='{}', sourceIndex={}, sourceName='{}'\n",
                k,
                rate->collisionName(),
                rate->correspondingSpecies(),
                kSource,
                m_phase->speciesName(kSource)
            );
        }

        return kSource;
    }

    // 2. Infer from the associated reaction product. This is the preferred
    // automatic path for lumped species, e.g.
    // Electron + N2 => Electron + N2(A)
    string reactionProduct;
    size_t nHeavyProducts = 0;

    for (const auto& [name, stoich] : m_phase->collision(k)->products) {
        if (name == electronName || stoich <= 0.0) {
            continue;
        }
        reactionProduct = name;
        nHeavyProducts++;
    }

    if (nHeavyProducts == 1 && reactionProduct != targetName) {
        size_t kProduct = m_phase->speciesIndex(reactionProduct, false);
        if (kProduct != npos) {
            if (DEBUG_SE) {
                writelog(
                    "EEDF-SE-DEBUG INFER_SOURCE: k={}, collision='{}', method='reaction-product', "
                    "reactionProduct='{}', sourceIndex={}, sourceName='{}'\n",
                    k,
                    rate->collisionName(),
                    reactionProduct,
                    kProduct,
                    m_phase->speciesName(kProduct)
                );
            }

            return kProduct;
        }
    }

    // 3. Fallback to the collision product, only if it is an actual phase
    // species. This supports species-resolved mechanisms where product already
    // names the physical excited species.
    if (!rate->product().empty() && rate->product() != targetName) {
        size_t kProduct = m_phase->speciesIndex(rate->product(), false);
        if (kProduct != npos) {
            if (DEBUG_SE) {
                writelog(
                    "EEDF-SE-DEBUG INFER_SOURCE: k={}, collision='{}', method='collision-product', "
                    "collisionProduct='{}', sourceIndex={}, sourceName='{}'\n",
                    k,
                    rate->collisionName(),
                    rate->product(),
                    kProduct,
                    m_phase->speciesName(kProduct)
                );
            }

            return kProduct;
        }
    }

    // 4. No valid population reservoir found.
    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG INFER_SOURCE_FAIL: k={}, collision='{}', "
            "kind='{}', target='{}', product='{}', correspondingSpecies='{}', "
            "nHeavyProducts={}, reactionProduct='{}'\n",
            k,
            rate->collisionName(),
            rate->kind(),
            targetName,
            rate->product(),
            rate->correspondingSpecies(),
            nHeavyProducts,
            reactionProduct
        );
    }
    return npos;
}

void EEDFTwoTermApproximation::initSuperElasticChannels()
{
    m_hasSuperElastic.assign(m_phase->nCollisions(), false);
    m_superElasticSourceSpecies.assign(m_phase->nCollisions(), npos);
    m_superElasticSourceLoc.assign(m_phase->nCollisions(), npos);
    m_superElasticWeights.assign(m_phase->nCollisions(), 0.0);
    m_k_lg_SuperElasticSources.clear();

    if (!m_enableSuperElasticCollisions) {
        m_X_superElasticSources.clear();
        m_X_superElasticSources_prev.clear();
        if (DEBUG_SE) {
            writelog("EEDF-SE-DEBUG INIT_SE: disabled, clearing source arrays.\n");
        }
        return;
    }

    for (size_t k : m_phase->kInelastic()) {
        auto rate = m_phase->collisionRate(k);
        if (DEBUG_SE) {
            writelog(
                "EEDF-SE-DEBUG INIT_SE_CANDIDATE: k={}, collision='{}', equation='{}', "
                "kind='{}', target='{}', product='{}', correspondingSpecies='{}', "
                "threshold={}, degeneracyRatio={}\n",
                k,
                rate->collisionName(),
                m_phase->collision(k)->equation(),
                rate->kind(),
                m_phase->speciesName(m_phase->targetIndex(k)),
                rate->product(),
                rate->correspondingSpecies(),
                rate->threshold(),
                rate->superElasticDegeneracyRatio()
            );
        }

        if (rate->kind() != "excitation") {
            continue;
        }

        if (rate->threshold() <= 0.0) {
            warn_user("EEDFTwoTermApproximation::initSuperElasticChannels",
                "Skipping super-elastic EEDF contribution for collision '{}': "
                "threshold must be strictly positive.",
                rate->collisionName());
            continue;
        }

        const size_t kSource = inferSuperElasticSourceSpecies(k);

        if (kSource == npos) {
            warn_user("EEDFTwoTermApproximation::initSuperElasticChannels",
                "Skipping super-elastic EEDF contribution for collision '{}': "
                "no valid 'corresponding-species', reaction product, or phase "
                "species matching 'product' was found.",
                rate->collisionName());
            continue;
        }

        m_hasSuperElastic[k] = true;
        m_superElasticSourceSpecies[k] = kSource;
        m_superElasticSourceLoc[k] = addSuperElasticSourceSpecies(kSource);
        if (DEBUG_SE) {
            writelog(
                "EEDF-SE-DEBUG INIT_SE_ENABLED: k={}, collision='{}', "
                "source='{}', sourceIndex={}, sourceLoc={}, "
                "X_source_raw={}, threshold={}, degeneracyRatio={}\n",
                k,
                rate->collisionName(),
                m_phase->speciesName(kSource),
                kSource,
                m_superElasticSourceLoc[k],
                m_phase->moleFraction(kSource),
                rate->threshold(),
                rate->superElasticDegeneracyRatio()
            );
        }
    }

    m_X_superElasticSources.assign(m_k_lg_SuperElasticSources.size(), 0.0);
    m_X_superElasticSources_prev.assign(m_k_lg_SuperElasticSources.size(), 0.0);

    for (size_t i = 0; i < m_k_lg_SuperElasticSources.size(); i++) {
        const size_t kSource = m_k_lg_SuperElasticSources[i];
        m_X_superElasticSources[i] = m_phase->moleFraction(kSource);
        m_X_superElasticSources_prev[i] = m_phase->moleFraction(kSource);
    }
    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG INIT_SE_SUMMARY: nSESources={}, nCollisions={}\n",
            m_k_lg_SuperElasticSources.size(),
            m_phase->nCollisions()
        );

        for (size_t i = 0; i < m_k_lg_SuperElasticSources.size(); i++) {
            const size_t kSource = m_k_lg_SuperElasticSources[i];

            writelog(
                "EEDF-SE-DEBUG INIT_SE_SOURCE: sourceLoc={}, species='{}', speciesIndex={}, X_raw={}\n",
                i,
                m_phase->speciesName(kSource),
                kSource,
                m_phase->moleFraction(kSource)
            );
        }
    }
}

void EEDFTwoTermApproximation::updateCrossSections()
{
    // Compute sigma_m and sigma_\epsilon
    calculateTotalCrossSection();
    calculateTotalElasticCrossSection();
}

// Update the species mole fractions used for EEDF computation
// Renormalize over species with electron-collision cross-section data.
void EEDFTwoTermApproximation::updateMoleFractions()
{
    double tmp_sum = 0.0;

    for (size_t k = 0; k < m_X_targets.size(); k++) {
        m_X_targets[k] = m_phase->moleFraction(m_k_lg_Targets[k]);
        tmp_sum += m_X_targets[k];
    }

    if (!std::isfinite(tmp_sum) || tmp_sum <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::updateMoleFractions",
            "The sum of target mole fractions is invalid or zero.");
    }

    // Preserve the existing behavior: forward target mole fractions are
    // normalized over species with electron-collision cross-section data.
    for (size_t k = 0; k < m_X_targets.size(); k++) {
        m_X_targets[k] /= tmp_sum;
    }

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG MOLE_TARGET_SUM: rawTargetSum={}, nTargets={}\n",
            tmp_sum,
            m_X_targets.size()
        );

        for (size_t k = 0; k < m_X_targets.size(); k++) {
            writelog(
                "EEDF-SE-DEBUG MOLE_TARGET: loc={}, species='{}', speciesIndex={}, "
                "X_raw={}, X_norm={}\n",
                k,
                m_phase->speciesName(m_k_lg_Targets[k]),
                m_k_lg_Targets[k],
                m_phase->moleFraction(m_k_lg_Targets[k]),
                m_X_targets[k]
            );
        }
    }

    updateSuperElasticMoleFractions();
}

// version 1
// void EEDFTwoTermApproximation::updateSuperElasticMoleFractions(
//     double targetMoleFractionSum)
// {
//     if (!m_enableSuperElasticCollisions) {
//         return;
//     }

//     if (!std::isfinite(targetMoleFractionSum) || targetMoleFractionSum <= 0.0) {
//         throw CanteraError("EEDFTwoTermApproximation::updateSuperElasticMoleFractions",
//             "Invalid normalization factor for super-elastic source mole fractions.");
//     }

//     for (size_t k = 0; k < m_X_superElasticSources.size(); k++) {
//         const size_t k_glob = m_k_lg_SuperElasticSources[k];
//         m_X_superElasticSources[k] =
//             m_phase->moleFraction(k_glob) / targetMoleFractionSum;
//     }
// }

//version 2
void EEDFTwoTermApproximation::updateSuperElasticMoleFractions()
{
    if (!m_enableSuperElasticCollisions) {
        return;
    }

    for (size_t k = 0; k < m_X_superElasticSources.size(); k++) {
    const size_t k_glob = m_k_lg_SuperElasticSources[k];
        m_X_superElasticSources[k] = m_phase->moleFraction(k_glob);

        if (DEBUG_SE) {
            writelog(
                "EEDF-SE-DEBUG MOLE_SE_SOURCE: sourceLoc={}, species='{}', speciesIndex={}, X_source={}\n",
                k,
                m_phase->speciesName(k_glob),
                k_glob,
                m_X_superElasticSources[k]
            );
        }
    }
}

// The former implementation counted the effective cros section as an elastic contribution
// thus double counting the inelastic contributions to the total cross section.
// This implemetation avoids this drawback.
void EEDFTwoTermApproximation::calculateTotalCrossSection()
{
    m_totalCrossSectionCenter.assign(m_points, 0.0);
    m_totalCrossSectionEdge.assign(m_points + 1, 0.0);

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG TOTALCS_START: applySE={}, enableSE={}, nCollisions={}, "
            "nInelastic={}, nElastic={}, nTargets={}, nSESources={}\n",
            m_applySuperElasticCollisions,
            m_enableSuperElasticCollisions,
            m_phase->nCollisions(),
            m_phase->kInelastic().size(),
            m_phase->kElastic().size(),
            m_X_targets.size(),
            m_X_superElasticSources.size()
        );
    }

    std::vector<bool> has_effective_for_target(m_phase->nSpecies(), false);

    // build this list first to avoid scanning through all collisions for each target twice in the loop.
    for (size_t ke : m_phase->kElastic()) {
        if (m_phase->collisionRate(ke)->kind() == "effective") {
            has_effective_for_target[m_phase->targetIndex(ke)] = true;
        }
    }

    for (size_t k = 0; k < m_phase->nCollisions(); k++) {
        const std::string& kind = m_phase->collisionRate(k)->kind();
        const size_t target = m_phase->targetIndex(k);

        // If this target has an effective cross section, then that effective
        // cross section already contains elastic + inelastic contributions.
        // So only the effective cross section contributes to sigma_total.
        if (has_effective_for_target[target] && kind != "effective") {
            continue;
        }

        auto x = m_phase->collisionRate(k)->energyLevels();
        auto y = m_phase->collisionRate(k)->crossSections();

        const double X_target = m_X_targets[m_klocTargets[k]];

        for (size_t i = 0; i < m_points; i++) {
            m_totalCrossSectionCenter[i] += X_target *
                                            linearInterp(m_gridCenter[i], x, y);
        }

        for (size_t i = 0; i < m_points + 1; i++) {
            m_totalCrossSectionEdge[i] += X_target *
                                          linearInterp(m_gridEdge[i], x, y);
        }
    }

    if (DEBUG_SE) {
        for (double epsDbg : {0.1, 1.0, 5.0, 10.0, 11.5, 15.8, 20.0, 25.0, 30.0}) {
            if (m_gridCenter.size() == 0) {
                continue;
            }

            const size_t idx = debugNearestIndex(m_gridCenter, epsDbg);

            writelog(
                "EEDF-SE-DEBUG TOTALCS_FORWARD_SAMPLE: requestedEps={}, gridEps={}, "
                "sigmaTotalForwardOnly={}\n",
                epsDbg,
                m_gridCenter[idx],
                m_totalCrossSectionCenter[idx]
            );
        }
    }

    if (m_applySuperElasticCollisions) {
        for (size_t k : m_phase->kInelastic()) {
            if (k >= m_hasSuperElastic.size() || !m_hasSuperElastic[k]) {
                continue;
            }

            const size_t sourceLoc = m_superElasticSourceLoc[k];
            const double prefactor =
                m_X_superElasticSources[sourceLoc] * m_superElasticWeights[k];

            if (prefactor <= 0.0) {
                continue;
            }

            if (DEBUG_SE) {
                auto rate = m_phase->collisionRate(k);

                writelog(
                    "EEDF-SE-DEBUG TOTALCS_SE_CHANNEL: k={}, collision='{}', source='{}', "
                    "X_source={}, weight={}, prefactor={}, threshold={}, degeneracyRatio={}\n",
                    k,
                    rate->collisionName(),
                    m_phase->speciesName(m_superElasticSourceSpecies[k]),
                    m_X_superElasticSources[sourceLoc],
                    m_superElasticWeights[k],
                    prefactor,
                    rate->threshold(),
                    rate->superElasticDegeneracyRatio()
                );

                for (double epsDbg : {0.1, 1.0, 5.0, 10.0, 11.5, 15.8, 20.0, 25.0, 30.0}) {
                    const double sigmaSE = superElasticCrossSection(k, epsDbg);

                    writelog(
                        "EEDF-SE-DEBUG TOTALCS_SE_SAMPLE: k={}, collision='{}', eps={}, "
                        "sigmaSE={}, prefactorSigmaSE={}\n",
                        k,
                        rate->collisionName(),
                        epsDbg,
                        sigmaSE,
                        prefactor * sigmaSE
                    );
                }
            }

            for (size_t i = 0; i < m_points; i++) {
                m_totalCrossSectionCenter[i] +=
                    prefactor * superElasticCrossSection(k, m_gridCenter[i]);
            }

            for (size_t i = 0; i < m_points + 1; i++) {
                m_totalCrossSectionEdge[i] +=
                    prefactor * superElasticCrossSection(k, m_gridEdge[i]);
            }
        }
    }

    if (DEBUG_SE) {
        for (double epsDbg : {0.1, 1.0, 5.0, 10.0, 11.5, 15.8, 20.0, 25.0, 30.0}) {
            if (m_gridCenter.size() == 0) {
                continue;
            }

            const size_t idx = debugNearestIndex(m_gridCenter, epsDbg);

            writelog(
                "EEDF-SE-DEBUG TOTALCS_FINAL_SAMPLE: requestedEps={}, gridEps={}, "
                "sigmaTotalFinal={}\n",
                epsDbg,
                m_gridCenter[idx],
                m_totalCrossSectionCenter[idx]
            );
        }
    }
}

double EEDFTwoTermApproximation::superElasticCrossSection(size_t k, double eps)
{
    if (eps <= 0.0) {
        return 0.0;
    }

    auto rate = m_phase->collisionRate(k);
    const double U = rate->threshold();

    const double sigmaForward = linearInterpCrossSectionZeroOutside(
        eps + U, rate->energyLevels(), rate->crossSections());

    const double sigmaSE =
        rate->superElasticDegeneracyRatio()
        * (eps + U) / eps
        * sigmaForward;

    if (!std::isfinite(sigmaSE) || sigmaSE < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::superElasticCrossSection",
            "Invalid super-elastic cross section for collision '{}'.",
            rate->collisionName());
    }

    return sigmaSE;
}

// new implementation of the function proposed by nicolas
void EEDFTwoTermApproximation::calculateTotalElasticCrossSection()
{
    m_sigmaElastic.clear();
    m_sigmaElastic.resize(m_points, 0.0);
    for (size_t k : m_phase->kElastic()) {

        // writelog("Enter with an elastic cs\n");
        auto x = m_phase->collisionRate(k)->energyLevels();
        auto y = m_phase->collisionRate(k)->crossSections();
        vector<double> y_elastic(y.data(), y.data() + y.size());

        // Note:
        // moleFraction(m_kTargets[k]) <=> m_X_targets[m_klocTargets[k]]
        double mass_ratio = ElectronMass / (m_phase->molecularWeight(m_kTargets[k]) / Avogadro);

        if (m_phase->collisionRate(k)->kind()=="effective") {

            // writelog("Enter with an elastic cs that is actually an effective\n");

            for (size_t ki : m_phase->kInelastic())
            {
                if(m_phase->targetIndex(ki) == m_phase->targetIndex(k)){
                    // writelog("loop over inelastic processes: process {}\n", ki);
                    auto xi = m_phase->collisionRate(ki)->energyLevels();
                    auto yi = m_phase->collisionRate(ki)->crossSections();
                    for (size_t i = 0; i < x.size(); i++)
                    {
                        y_elastic[i] -= linearInterpCrossSectionZeroOutside(x[i], xi, yi);
                    }
                }
            }
            // check that the reconstructed elastic cross section is non-negative.
            for (size_t i = 0; i < y_elastic.size(); i++) {
                if (y_elastic[i] < 0.0) {
                    if (y_elastic[i] > -1e-30) {
                        y_elastic[i] = 0.0;
                    } else {
                        const std::string& effectiveKind = m_phase->collisionRate(k)->kind();
                        std::string effectiveTarget = m_phase->speciesName(m_phase->targetIndex(k));

                        writelog("Warning: reconstructed elastic cross section is negative "
                                "for collision {} kind {} target {} at energy {}: {}\n",
                                k, effectiveKind, effectiveTarget, x[i], y_elastic[i]);
                        y_elastic[i] = 0.0;
                    }
                }
            }
        }

        for (size_t i = 0; i < m_points; i++) {
            m_sigmaElastic[i] += 2.0 * mass_ratio * m_X_targets[m_klocTargets[k]] *
                                 linearInterp(m_gridEdge[i], x, y_elastic);
        }
    }
}

double EEDFTwoTermApproximation::linearInterpBounded(double x,
                           span<const double> xpts,
                           span<const double> fpts,
                           double below_value,
                           double above_value)
{
    AssertThrowMsg(!xpts.empty(), "linearInterpBounded", "x data empty");
    AssertThrowMsg(!fpts.empty(), "linearInterpBounded", "f(x) data empty");
    AssertThrowMsg(xpts.size() == fpts.size(), "linearInterpBounded",
        "len(xpts) = {}, len(fpts) = {}", xpts.size(), fpts.size());

    if (x < xpts.front()) {
        return below_value;
    }

    if (x > xpts.back()) {
        return above_value;
    }

    return linearInterp(x, xpts, fpts);
}

double EEDFTwoTermApproximation::linearInterpCrossSectionZeroOutside(double x,
                                           span<const double> xpts,
                                           span<const double> fpts)
{
    return linearInterpBounded(x, xpts, fpts, 0.0, 0.0);
}

void EEDFTwoTermApproximation::checkSpeciesNoCrossSection()
{
    // Warn that a species has a significant mole fraction but no forward
    // electron-collision cross-section data. Species used only as
    // super-elastic population sources are excluded when the super-elastic
    // model is enabled.
    for (size_t k : m_kOthers) {
        if (m_enableSuperElasticCollisions) {
            auto it = std::find(m_k_lg_SuperElasticSources.begin(),
                                m_k_lg_SuperElasticSources.end(), k);
            if (it != m_k_lg_SuperElasticSources.end()) {
                continue;
            }
        }

        if (m_phase->moleFraction(k) > m_moleFractionThreshold) {
            writelog("EEDFTwoTermApproximation:checkSpeciesNoCrossSection\n");
            writelog("Warning: The mole fraction of species {} is more than "
                     "0.01 (X = {:.3g}) but it has no cross-section data\n",
                     m_phase->speciesName(k), m_phase->moleFraction(k));
        }
    }
}

void EEDFTwoTermApproximation::setGridCache()
{
    m_sigma.clear();
    m_sigma.resize(m_phase->nCollisions());

    m_eps.clear();
    m_eps.resize(m_phase->nCollisions());

    m_j.clear();
    m_j.resize(m_phase->nCollisions());

    m_i.clear();
    m_i.resize(m_phase->nCollisions());

    m_epsSuperElastic.clear();
    m_epsSuperElastic.resize(m_phase->nCollisions());

    m_epsSigmaSuperElastic.clear();
    m_epsSigmaSuperElastic.resize(m_phase->nCollisions());

    m_jSuperElastic.clear();
    m_jSuperElastic.resize(m_phase->nCollisions());

    m_iSuperElastic.clear();
    m_iSuperElastic.resize(m_phase->nCollisions());

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG GRIDCACHE_START: nCollisions={}, nPoints={}, "
            "gridEmin={}, gridEmax={}\n",
            m_phase->nCollisions(),
            m_points,
            m_points > 0 ? m_gridEdge[0] : -1.0,
            m_points > 0 ? m_gridEdge[m_points] : -1.0
        );
    }

    for (size_t k = 0; k < m_phase->nCollisions(); k++) {
        auto collision = m_phase->collisionRate(k);
        auto x = collision->energyLevels();
        auto y = collision->crossSections();

        // Forward inelastic cache: existing behavior.
        vector<double> eps1(m_points + 1);
        int shiftFactor = (collision->kind() == "ionization" && m_growth!="none") ? 2 : 1;

        for (size_t i = 0; i < m_points + 1; i++) {
            eps1[i] = clip(shiftFactor * m_gridEdge[i] + collision->threshold(),
                           m_gridEdge[0] + 1e-9,
                           m_gridEdge[m_points] - 1e-9);
        }

        vector<double> nodes = eps1;

        for (size_t i = 0; i < m_points + 1; i++) {
            if (m_gridEdge[i] >= eps1[0] &&
                m_gridEdge[i] <= eps1[m_points]) {
                nodes.push_back(m_gridEdge[i]);
            }
        }

        for (size_t i = 0; i < x.size(); i++) {
            if (x[i] >= eps1[0] && x[i] <= eps1[m_points]) {
                nodes.push_back(x[i]);
            }
        }

        std::sort(nodes.begin(), nodes.end());
        auto last = std::unique(nodes.begin(), nodes.end());
        nodes.resize(std::distance(nodes.begin(), last));

        vector<double> sigma0(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++) {
            sigma0[i] = linearInterp(nodes[i], x, y);
        }

        for (size_t i = 1; i < nodes.size(); i++) {
            auto low = std::lower_bound(m_gridEdge.begin(),
                                        m_gridEdge.end(),
                                        nodes[i]);
            m_j[k].push_back(low - m_gridEdge.begin() - 1);
        }

        for (size_t i = 1; i < nodes.size(); i++) {
            auto low = std::lower_bound(eps1.begin(),
                                        eps1.end(),
                                        nodes[i]);
            m_i[k].push_back(low - eps1.begin() - 1);
        }

        for (size_t i = 0; i < nodes.size() - 1; i++) {
            m_sigma[k].push_back({sigma0[i], sigma0[i + 1]});
            m_eps[k].push_back({nodes[i], nodes[i + 1]});
        }

        // Super-elastic cache.
        //
        // Source energy is eps. Destination energy is eps + U.
        // Therefore source intervals must be split by:
        //   - source grid cell boundaries,
        //   - destination grid cell boundaries shifted by -U,
        //   - cross-section tabulation energies shifted by -U.
        if (collision->kind() != "excitation") {
            continue;
        }

        const double U = collision->threshold();
        const double epsMax = m_gridEdge[m_points];

        if (U <= 0.0 || U >= epsMax) {
            continue;
        }

        vector<double> nodesSE;
        nodesSE.push_back(m_gridEdge[0]);
        nodesSE.push_back(epsMax - U);

        // Source-cell boundaries.
        for (double edge : m_gridEdge) {
            if (edge >= 0.0 && edge <= epsMax - U) {
                nodesSE.push_back(edge);
            }
        }

        // Destination-cell boundaries shifted back to source coordinates.
        for (double edge : m_gridEdge) {
            const double shifted = edge - U;
            if (shifted >= 0.0 && shifted <= epsMax - U) {
                nodesSE.push_back(shifted);
            }
        }

        // Cross-section nodes shifted back to source coordinates.
        for (double level : x) {
            const double shifted = level - U;
            if (shifted >= 0.0 && shifted <= epsMax - U) {
                nodesSE.push_back(shifted);
            }
        }

        std::sort(nodesSE.begin(), nodesSE.end());
        auto lastSE = std::unique(nodesSE.begin(), nodesSE.end());
        nodesSE.resize(std::distance(nodesSE.begin(), lastSE));

        const double ratio = collision->superElasticDegeneracyRatio();

        for (size_t n = 0; n + 1 < nodesSE.size(); n++) {
            const double a = nodesSE[n];
            const double b = nodesSE[n + 1];

            if (b <= a) {
                continue;
            }

            const double mid = 0.5 * (a + b);
            const double midOut = mid + U;

            if (mid < m_gridEdge.front() || mid >= m_gridEdge.back() ||
                midOut < m_gridEdge.front() || midOut >= m_gridEdge.back()) {
                continue;
            }

            auto jIt = std::upper_bound(m_gridEdge.begin(),
                                        m_gridEdge.end(),
                                        mid);
            auto iIt = std::upper_bound(m_gridEdge.begin(),
                                        m_gridEdge.end(),
                                        midOut);

            if (jIt == m_gridEdge.begin() || iIt == m_gridEdge.begin()) {
                continue;
            }

            const size_t j =
                static_cast<size_t>(std::distance(m_gridEdge.begin(), jIt) - 1);
            const size_t i =
                static_cast<size_t>(std::distance(m_gridEdge.begin(), iIt) - 1);

            if (j >= m_points || i >= m_points) {
                continue;
            }

            const double sigmaA = linearInterpCrossSectionZeroOutside(
                a + U, x, y);
            const double sigmaB = linearInterpCrossSectionZeroOutside(
                b + U, x, y);

            // Store eps * sigma_se(eps), not sigma_se(eps), to avoid
            // the removable singularity at eps = 0.
            const double wA = ratio * (a + U) * sigmaA;
            const double wB = ratio * (b + U) * sigmaB;

            if (wA == 0.0 && wB == 0.0) {
                continue;
            }

            m_jSuperElastic[k].push_back(j);
            m_iSuperElastic[k].push_back(i);
            m_epsSuperElastic[k].push_back({a, b});
            m_epsSigmaSuperElastic[k].push_back({wA, wB});
        }
        if (DEBUG_SE && collision->kind() == "excitation") {
            writelog(
                "EEDF-SE-DEBUG GRIDCACHE_SE: k={}, collision='{}', "
                "kind='{}', threshold={}, degeneracyRatio={}, "
                "forwardSegments={}, seSegments={}, energyLevels={}, "
                "epsMax={}, seSourceMax={}\n",
                k,
                collision->collisionName(),
                collision->kind(),
                collision->threshold(),
                collision->superElasticDegeneracyRatio(),
                m_eps[k].size(),
                m_epsSuperElastic[k].size(),
                x.size(),
                epsMax,
                epsMax - U
            );

            if (!m_epsSuperElastic[k].empty()) {
                const size_t first = 0;
                const size_t last = m_epsSuperElastic[k].size() - 1;

                writelog(
                    "EEDF-SE-DEBUG GRIDCACHE_SE_FIRST: k={}, "
                    "j={}, i={}, epsA={}, epsB={}, epsOutA={}, epsOutB={}, "
                    "epsSigmaA={}, epsSigmaB={}\n",
                    k,
                    m_jSuperElastic[k][first],
                    m_iSuperElastic[k][first],
                    m_epsSuperElastic[k][first][0],
                    m_epsSuperElastic[k][first][1],
                    m_epsSuperElastic[k][first][0] + U,
                    m_epsSuperElastic[k][first][1] + U,
                    m_epsSigmaSuperElastic[k][first][0],
                    m_epsSigmaSuperElastic[k][first][1]
                );

                writelog(
                    "EEDF-SE-DEBUG GRIDCACHE_SE_LAST: k={}, "
                    "j={}, i={}, epsA={}, epsB={}, epsOutA={}, epsOutB={}, "
                    "epsSigmaA={}, epsSigmaB={}\n",
                    k,
                    m_jSuperElastic[k][last],
                    m_iSuperElastic[k][last],
                    m_epsSuperElastic[k][last][0],
                    m_epsSuperElastic[k][last][1],
                    m_epsSuperElastic[k][last][0] + U,
                    m_epsSuperElastic[k][last][1] + U,
                    m_epsSigmaSuperElastic[k][last][0],
                    m_epsSigmaSuperElastic[k][last][1]
                );
            }
        }
    }
}

double EEDFTwoTermApproximation::norm(const Eigen::VectorXd& f, const Eigen::VectorXd& grid)
{
    string m_quadratureMethod = "simpson";
    Eigen::VectorXd p(f.size());
    for (int i = 0; i < f.size(); i++) {
        p[i] = f(i) * pow(grid[i], 0.5);
    }
    return numericalQuadrature(m_quadratureMethod, p, grid);
}

void EEDFTwoTermApproximation::setGridType(const string& gridType)
{
    if (gridType != "Linear" &&
        gridType != "Quadratic" &&
        gridType != "Geometric") {
        throw CanteraError("EEDFTwoTermApproximation::setGridType",
            "Unknown energy grid type '{}'. Expected Linear, Quadratic or Geometric.",
            gridType);
    }

    m_gridType = gridType;
}

void EEDFTwoTermApproximation::setInitialGridParameters(double initialMaxEnergy,
                                                        size_t nGridCells)
{
    if (!std::isfinite(initialMaxEnergy) || initialMaxEnergy <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::setInitialGridParameters",
            "initialMaxEnergy must be finite and greater than zero.");
    }

    if (nGridCells == 0) {
        throw CanteraError("EEDFTwoTermApproximation::setInitialGridParameters",
            "nGridCells must be greater than zero.");
    }

    m_kTeMax = initialMaxEnergy;
    m_initialGridCells = nGridCells;
}

void EEDFTwoTermApproximation::enableGridAdaptation(bool enabled)
{
    m_adaptGrid = enabled;
}

void EEDFTwoTermApproximation::setGridAdaptationParameters(bool enabled,
                                                           double minDecayDecades,
                                                           double maxDecayDecades,
                                                           double updateFactor,
                                                           size_t maxIterations, 
                                                           bool maxwellian_reset)
{
    if (!std::isfinite(minDecayDecades) || !std::isfinite(maxDecayDecades) ||
        minDecayDecades <= 0.0 || maxDecayDecades <= minDecayDecades) {
        throw CanteraError("EEDFTwoTermApproximation::setGridAdaptationParameters",
            "Require 0 < min_decay_decades < max_decay_decades.");
    }

    if (!std::isfinite(updateFactor) || updateFactor <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::setGridAdaptationParameters",
            "update_factor must be finite and greater than zero.");
    }

    if (maxIterations == 0) {
        throw CanteraError("EEDFTwoTermApproximation::setGridAdaptationParameters",
            "max_iterations must be greater than zero.");
    }

    m_adaptGrid = enabled;
    m_minEedfDecay = minDecayDecades;
    m_maxEedfDecay = maxDecayDecades;
    m_gridUpdateFactor = updateFactor;
    m_maxGridAdaptIterations = maxIterations;
    m_maxwellianReset = maxwellian_reset;
}

void EEDFTwoTermApproximation::updateGrid(double maxEnergy)
{
    if (!std::isfinite(maxEnergy) || maxEnergy <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::updateGrid",
            "Maximum grid energy must be finite and greater than zero.");
    }

    m_kTeMax = maxEnergy;

    if (m_gridType == "Linear") {
        setLinearGrid(m_kTeMax, m_initialGridCells);
    } else if (m_gridType == "Quadratic") {
        setQuadraticGrid(m_kTeMax, m_initialGridCells);
    } else if (m_gridType == "Geometric") {
        setGeometricGrid(m_kTeMax, m_initialGridCells);
    } else {
        throw CanteraError("EEDFTwoTermApproximation::updateGrid",
            "Unknown energy grid type '{}'.", m_gridType);
    }

    // put back the flag at false because since the grid has changed the EEDF must be computed again.
    m_has_EEDF = false;
}

double EEDFTwoTermApproximation::getElectronMobility() const
{
    if (!m_has_EEDF || !std::isfinite(m_electronMobility)) {
        throw CanteraError("EEDFTwoTermApproximation::getElectronMobility",
            "Electron mobility is not available before a valid EEDF has been computed.");
    }
    return m_electronMobility;
}

// Set the reduced electric field threshold below which the EEDF is forced to be Maxwellian at the gas temperature. 
// The input to this function is expected to be in Townsend.
void EEDFTwoTermApproximation::setReducedFieldThresholdBeforeMaxwellianTd(double threshold){
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::setReducedFieldThresholdBeforeMaxwellianTd",
            "Reduced field threshold must be finite and non-negative.");
    }
    EN_min = threshold*1e-21;
}

void checkTolerancePair(const string& name, double rtol, double atol)
{
    if (!std::isfinite(rtol) || !std::isfinite(atol) ||
        rtol < 0.0 || atol < 0.0 || (rtol == 0.0 && atol == 0.0)) {
        throw CanteraError("EEDFTwoTermApproximation::" + name,
            "Tolerances must be finite and non-negative, and at least one of "
            "rtol / atol must be positive.");
    }
}

void checkAbsoluteTolerance(const string& name, double atol)
{
    if (!std::isfinite(atol) || atol < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::" + name,
            "Absolute tolerance must be finite and non-negative.");
    }
}

void EEDFTwoTermApproximation::setTemperatureTolerance(double rtol, double atol)
{
    checkTolerancePair("setTemperatureTolerance", rtol, atol);

    m_temperature_rtol = rtol;
    m_temperature_atol = atol;
    m_f0_ok = false;
}

void EEDFTwoTermApproximation::setReducedElectricFieldTolerance(double rtol,
                                                               double atol)
{
    checkTolerancePair("setReducedElectricFieldTolerance", rtol, atol);

    m_EN_rtol = rtol;
    m_EN_atol = atol;
    m_f0_ok = false;
}

void EEDFTwoTermApproximation::setTargetMoleFractionTolerance(double atol)
{
    checkAbsoluteTolerance("setTargetMoleFractionTolerance", atol);

    m_X_atol = atol;
    m_f0_ok = false;
}

void EEDFTwoTermApproximation::setEEDFRecalculationTolerances(
    double temperatureRtol,
    double temperatureAtol,
    double reducedFieldRtol,
    double reducedFieldAtol,
    double targetMoleFractionAtol)
{
    setTemperatureTolerance(temperatureRtol, temperatureAtol);
    setReducedElectricFieldTolerance(reducedFieldRtol, reducedFieldAtol);
    setTargetMoleFractionTolerance(targetMoleFractionAtol);

    m_f0_ok = false;
}

void EEDFTwoTermApproximation::setupEeCol(const double ionDegree,
                                          const double nElectron)
{
    if (!std::isfinite(ionDegree) || ionDegree < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::setupEeCol",
            "Ionization degree must be finite and non-negative.");
    }

    if (!std::isfinite(nElectron) || nElectron < 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::setupEeCol",
            "Electron number density must be finite and non-negative.");
    }

    const bool oldActive = m_eeCol;
    const bool oldRelevant =
        m_enableEeCollisions &&
        m_ionDegree > m_eeIonDegreeThreshold &&
        m_nElectron > 0.0;

    m_ionDegree = ionDegree;
    m_nElectron = nElectron;

    // If the user enabled e-e collisions, the model is active. The magnitude is controlled by m_ionDegree.
    m_eeCol = m_enableEeCollisions && m_nElectron > 0.0;

    // The model is relevant if the ionization degree is above the threshold and e-e collisions are enabled.
    const bool newRelevant =
        m_eeCol &&
        m_ionDegree > m_eeIonDegreeThreshold;

    bool needRecompute = oldActive != m_eeCol || oldRelevant != newRelevant;

    if (newRelevant) {
        needRecompute =
            needRecompute ||
            parameterChanged(m_ionDegree, m_ionDegree_tmp,
                             m_eeUpdateRtol, 0.0) ||
            parameterChanged(m_nElectron, m_nElectron_tmp,
                             m_eeUpdateRtol, 0.0);
    }

    if (needRecompute) {
        m_f0_ok = false;
    }

    m_ionDegree_tmp = m_ionDegree;
    m_nElectron_tmp = m_nElectron;
}
void EEDFTwoTermApproximation::enableElectronElectronCollisions(bool enable)
{
    if (m_enableEeCollisions != enable) {
        m_enableEeCollisions = enable;
        m_f0_ok = false;
    }
}

void EEDFTwoTermApproximation::enableSuperElasticCollisions(bool enable)
{
    if (m_enableSuperElasticCollisions != enable) {
        m_enableSuperElasticCollisions = enable;
        m_applySuperElasticCollisions = false;

        // Force reinitialization of source-species mapping on the next EEDF
        // calculation. initSpeciesIndexCrossSections() clears its containers,
        // so this is safe.
        m_first_call = true;
        m_f0_ok = false;
    }
}

void EEDFTwoTermApproximation::projectPreviousEEDFOnCurrentGrid(
    const Eigen::VectorXd& oldGridCenter,
    const Eigen::VectorXd& oldF0)
{
    if (oldGridCenter.size() != oldF0.size() || oldGridCenter.size() < 2) {
        throw CanteraError("EEDFTwoTermApproximation::projectPreviousEEDFOnCurrentGrid",
            "Previous EEDF and grid must have matching sizes of at least two points.");
    }

    const double fFloor = 1e-300;

    vector<double> oldGrid(oldGridCenter.data(),
        oldGridCenter.data() + oldGridCenter.size());

    vector<double> oldF(oldF0.data(),
        oldF0.data() + oldF0.size());

    for (size_t j = 0; j < m_points; j++) {
        m_f0(j) = std::max(fFloor,
            linearInterpBounded(m_gridCenter[j], oldGrid, oldF, fFloor, fFloor));
    }

    double fnorm = norm(m_f0, m_gridCenter);

    if (!std::isfinite(fnorm) || fnorm <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::projectPreviousEEDFOnCurrentGrid",
            "Invalid norm after projecting previous EEDF onto the adapted grid.");
    }

    m_f0 /= fnorm;
}
}
