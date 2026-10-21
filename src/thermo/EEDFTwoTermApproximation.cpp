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
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>

namespace Cantera
{

typedef Eigen::SparseMatrix<double> SparseMat;

namespace {
constexpr bool DEBUG_SE = false;
constexpr bool DEBUG_SE_ITER = false;

// Optional model-only extrapolation diagnostic. It never controls Emax
// or acceptance. Leave disabled for production/timing benchmarks.
constexpr bool AUTO_TAIL_MODEL_DIAGNOSTIC_ENABLED = false;

// Automatic-grid robustness constants.
constexpr double EEDF_NEGATIVE_REL_TOL = 1.0e-12;
constexpr double RATE_NEGATIVE_ABS_TOL = 1.0e-60; // [m^3/s]
constexpr double RATE_ROUNDOFF_FACTOR = 256.0;

// Pseudo-time backtracking used by converge().
// A rejected nonlinear trial is recomputed from the same previous EEDF
// with a smaller pseudo-time step.
constexpr double EEDF_DELTA_BACKTRACK_FACTOR = 0.5;

// Allows a reduction of delta by up to 2^16 = 65536.
// This is only used for pathological iterations.
constexpr size_t EEDF_DELTA_BACKTRACK_MAX_RETRIES = 16;

// --------------------------------------------------------------------------
// Adaptive under-relaxation of Automatic-grid interior node positions.
//
// Large proposed remeshes are applied without damping so that the grid can
// move rapidly toward the physically relevant region.
//
// Close to the grid fixed point, the remeshing is progressively damped to
// suppress oscillations caused by highly grid-sensitive observables.
// --------------------------------------------------------------------------

constexpr double AUTO_GRID_RELAXATION_MIN_FACTOR = 0.5;

// Maximum normalized grid displacement below which relaxation is active.
// For shift >= this value, alpha = 1 and the proposed grid is applied fully.
constexpr double AUTO_GRID_RELAXATION_FULL_SHIFT = 0.08;

// --------------------------------------------------------------------------
// TEMPORARY Automatic-grid monitor diagnostics.
//
// This instrumentation does NOT change the grid algorithm.
// It only writes diagnostic CSV files for selected E/N and cell counts.
// --------------------------------------------------------------------------

constexpr bool DEBUG_AUTO_GRID_MONITOR = false;

// Only dump these two reduced fields [Td].
constexpr double DEBUG_AUTO_GRID_EN_1_TD = 1.0;
constexpr double DEBUG_AUTO_GRID_EN_2_TD = 15.848931924611133;
constexpr double DEBUG_AUTO_GRID_EN_TOL_TD = 1.0e-6;

constexpr const char* AUTO_GRID_MONITOR_DEBUG_FILE =
    "automatic_grid_monitor_debug.csv";

constexpr const char* AUTO_GRID_EDGES_DEBUG_FILE =
    "automatic_grid_edges_debug.csv";

constexpr const char* AUTO_GRID_ITERATION_DEBUG_FILE =
    "automatic_grid_iteration_debug.csv";

// Predictor-corrector iteration currently calling buildAutomaticGrid().
thread_local size_t DEBUG_AUTO_GRID_PC_ITERATION = 0;

// Monotonically increasing identifier for each dumped grid proposal.
thread_local size_t DEBUG_AUTO_GRID_BUILD_COUNTER = 0;

// Identifier corresponding to the current call to buildAutomaticGrid().
// Zero means that the current state is not one of the targeted diagnostics.
thread_local size_t DEBUG_AUTO_GRID_CURRENT_BUILD_ID = 0;


struct AutomaticGridDebugPoint
{
    double f = 0.0;
    double curvature = 0.0;

    double temperatureImportance = 0.0;

    double maxCollisionImportance = 0.0;
    string maxCollisionName;
    string controllerName;

    double n2v1resImportance = 0.0;
    double n2v1resSigma = 0.0;

    double o2v1resImportance = 0.0;
    double o2v1resSigma = 0.0;

    double o2v2resImportance = 0.0;
    double o2v2resSigma = 0.0;

    double o2v1Importance = 0.0;
    double o2v1Sigma = 0.0;

    double totalImportance = 0.0;

    double adaptiveDensity = 0.0;
    double baselineDensity = 0.0;

    double thresholdGuard = 0.0;
    double thresholdGuardDensity = 0.0;

    double rhoBeforeSmoothing = 0.0;
};


bool debugAutomaticGridTarget(
    double enTd,
    size_t nCells)
{
    if (!DEBUG_AUTO_GRID_MONITOR) {
        return false;
    }

    const bool targetCells =
        nCells == 200 ||
        nCells == 400;

    const bool targetField =
        std::abs(enTd - DEBUG_AUTO_GRID_EN_1_TD)
            <= DEBUG_AUTO_GRID_EN_TOL_TD
        ||
        std::abs(enTd - DEBUG_AUTO_GRID_EN_2_TD)
            <= DEBUG_AUTO_GRID_EN_TOL_TD;

    return targetCells && targetField;
}


// Avoid commas / quotes inside CSV string fields.
string debugAutomaticGridCsvSafe(string value)
{
    for (char& c : value) {
        if (c == ',') {
            c = ';';
        } else if (c == '"') {
            c = '\'';
        }
    }

    return value;
}

/// END OF TEMPORARY TIAGNOSTIC FILES

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
    // Preserve the configured ratio when restoring the legacy grid or changing Emax.
    m_geometricGridRatio = ratio;
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

    // The cached distribution must never survive a change of physical branch,
    // regardless of the numerical tolerance on E/N.
    if ((EN <= EN_min) != m_lastEEDFWasForcedMaxwellian) {
        m_f0_ok = false;
        return true;
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

    // These values must represent the state of the last computed EEDF,
    // rather than the state of the previous call to setupEeCol().
    m_ionDegree_tmp = m_ionDegree;
    m_nElectron_tmp = m_nElectron;

    m_f0_ok = true;
}

vector<double>
EEDFTwoTermApproximation::makeAutomaticBootstrapGrid(
    double maxEnergy,
    size_t nCells) const
{
    if (!std::isfinite(maxEnergy) || maxEnergy <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::makeAutomaticBootstrapGrid",
            "Maximum energy must be finite and positive.");
    }

    if (nCells < 3) {
        throw CanteraError(
            "EEDFTwoTermApproximation::makeAutomaticBootstrapGrid",
            "At least three cells are required.");
    }

    const double Es =
        m_automaticBootstrapStretchEnergy;

    if (!std::isfinite(Es) || Es <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::makeAutomaticBootstrapGrid",
            "Invalid bootstrap stretching energy.");
    }

    vector<double> edges(nCells + 1, 0.0);

    const double logarithmicSpan =
        std::log1p(maxEnergy / Es);

    for (size_t j = 0; j <= nCells; j++) {
        const double u =
            static_cast<double>(j)
            / static_cast<double>(nCells);

        edges[j] =
            Es * std::expm1(
                u * logarithmicSpan);
    }

    edges.front() = 0.0;
    edges.back() = maxEnergy;

    return edges;
}

void EEDFTwoTermApproximation::setForcedMaxwellianGridParameters(
    const string& type, size_t cells, double decayDecades)
{
    if (type != "Linear" && type != "Quadratic" && type != "Geometric") {
        throw CanteraError("EEDFTwoTermApproximation::setForcedMaxwellianGridParameters",
            "Grid type must be Linear, Quadratic, or Geometric.");
    }
    if (cells < 3 || !std::isfinite(decayDecades) || decayDecades <= 0.0 ||
        decayDecades > 250.0) {
        throw CanteraError("EEDFTwoTermApproximation::setForcedMaxwellianGridParameters",
            "Expected at least 3 cells and finite decay decades in (0, 250].");
    }
    m_forcedMaxwellianGridType = type;
    m_forcedMaxwellianGridCells = cells;
    m_forcedMaxwellianDecayDecades = decayDecades;
    m_f0_ok = false;
}

void EEDFTwoTermApproximation::buildForcedMaxwellianGrid(double gasKTeEv)
{
    if (!std::isfinite(gasKTeEv) || gasKTeEv <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::buildForcedMaxwellianGrid",
            "Gas electron thermal energy must be finite and positive.");
    }
    const size_t n = m_forcedMaxwellianGridCells;
    const double emax = gasKTeEv * std::log(10.0) * m_forcedMaxwellianDecayDecades;
    if (!std::isfinite(emax) || emax <= 0.0) {
        throw CanteraError("EEDFTwoTermApproximation::buildForcedMaxwellianGrid",
            "Invalid Maxwellian grid maximum energy.");
    }
    vector<double> edges(n + 1, 0.0);
    for (size_t j = 1; j <= n; ++j) {
        const double u = static_cast<double>(j) / static_cast<double>(n);
        if (m_forcedMaxwellianGridType == "Linear") {
            edges[j] = emax * u;
        } else if (m_forcedMaxwellianGridType == "Quadratic") {
            edges[j] = emax * static_cast<double>(j) * (j + 1.0) /
                (static_cast<double>(n) * (n + 1.0));
        } else {
            // Fixed-ratio exponential mapping, without an additional YAML parameter.
            const double a = std::log(1.01);
            edges[j] = emax * std::expm1(a * j) / std::expm1(a * n);
        }
    }
    edges.back() = emax;
    setCustomGrid(edges);
    m_kTeMax = emax;
}

void EEDFTwoTermApproximation::setMaxwellianEEDF(
    double kTe_eV)
{
    if (!std::isfinite(kTe_eV) ||
        kTe_eV <= 0.0) {

        throw CanteraError(
            "EEDFTwoTermApproximation::setMaxwellianEEDF",
            "Invalid kTe value for Maxwellian initialization.");
    }

    const double fFloor = 1e-300;

    for (size_t j = 0; j < m_points; j++) {
        const double arg =
            -m_gridCenter[j] / kTe_eV;

        if (arg < -700.0) {
            m_f0(j) = fFloor;
        } else {
            m_f0(j) =
                std::max(
                    fFloor,
                    2.0
                    * std::sqrt(1.0 / Pi)
                    * std::pow(kTe_eV, -1.5)
                    * std::exp(arg));
        }
    }

    const double fnorm =
        norm(m_f0, m_gridCenter);

    if (!std::isfinite(fnorm) ||
        fnorm <= 0.0) {

        throw CanteraError(
            "EEDFTwoTermApproximation::setMaxwellianEEDF",
            "Invalid norm for Maxwellian initialization.");
    }

    m_f0 /= fnorm;
}

EEDFTwoTermApproximation::AutomaticTailFit
EEDFTwoTermApproximation::automaticTailFit() const
{
    AutomaticTailFit fit;
    if (m_points < 6) {
        return fit;
    }

    constexpr double usableFloor = 1e-280;
    constexpr size_t minPoints = 6;
    constexpr std::array<int, 10> percentages{{
        5, 7, 9, 10, 11, 13, 15, 17, 19, 20}};
    vector<size_t> valid;
    valid.reserve(m_points);

    for (size_t i = 0; i < m_points; ++i) {
        const double f = m_f0(i);
        if (std::isfinite(f) && f > usableFloor
            && std::isfinite(m_gridCenter[i])) {
            valid.push_back(i);
        }
    }
    fit.validPoints = valid.size();
    if (valid.size() < minPoints) {
        return fit;
    }

    for (size_t j = 0; j < percentages.size(); ++j) {
        const size_t count = std::min(valid.size(),
            std::max(minPoints,
                static_cast<size_t>(std::ceil(
                    percentages[j] * 0.01 * valid.size()))));
        fit.counts[j] = count;

        double weightSum = 0.0;
        double weightedX = 0.0;
        double weightedY = 0.0;
        const size_t start = valid.size() - count;
        for (size_t k = start; k < valid.size(); ++k) {
            const size_t i = valid[k];
            const double w = m_gridEdge[i + 1] - m_gridEdge[i];
            if (!(w > 0.0) || !std::isfinite(w)) {
                return fit;
            }
            weightSum += w;
            weightedX += w * m_gridCenter[i];
            weightedY += w * std::log(m_f0(i));
        }
        const double xMean = weightedX / weightSum;
        const double yMean = weightedY / weightSum;
        double num = 0.0;
        double den = 0.0;
        for (size_t k = start; k < valid.size(); ++k) {
            const size_t i = valid[k];
            const double w = m_gridEdge[i + 1] - m_gridEdge[i];
            const double dx = m_gridCenter[i] - xMean;
            num += w * dx * (std::log(m_f0(i)) - yMean);
            den += w * dx * dx;
        }
        if (den > 0.0) {
            const double beta = -num / den;
            if (std::isfinite(beta) && beta > 1e-12) {
                fit.slopes[j] = beta;
            }
        }
    }

    // Only the four *core* windows (5, 7, 9, 10%) authorize
    // exponential extrapolation. The wider 11..20% windows are diagnostic
    // and can supply an exploratory extension span during bootstrap.
    vector<double> positive;
    for (size_t j = 0; j < 4; ++j) {
        const double beta = fit.slopes[j];
        if (std::isfinite(beta) && beta > 0.0) {
            positive.push_back(beta);
        }
    }
    if (!positive.empty()) {
        std::sort(positive.begin(), positive.end());
        fit.localSlope = positive[positive.size() / 2];
        fit.conservativeSlope = positive.front();
    }
    if (valid.back() != m_points - 1) {
        return fit; // floored tail cannot certify extrapolation
    }

    // Find a coherent triplet among the extended percentages, for a
    // *bootstrap step size only*. It cannot validate extrapolation.
    for (size_t j = 4; j + 2 < percentages.size(); ++j) {
        if (fit.counts[j] == fit.counts[j + 1]
            || fit.counts[j + 1] == fit.counts[j + 2]) {
            continue;
        }
        const double a = fit.slopes[j];
        const double b = fit.slopes[j + 1];
        const double c = fit.slopes[j + 2];
        if (!(a > 0.0 && b > 0.0 && c > 0.0)) {
            continue;
        }
        const double lo = std::min({a, b, c});
        const double hi = std::max({a, b, c});
        if ((hi - lo) / b <= 0.10) {
            const size_t first = valid[valid.size() - fit.counts[j + 2]];
            fit.extendedStableSpanEv =
                m_gridCenter[m_points - 1] - m_gridCenter[first];
            fit.extendedStablePercent = percentages[j + 2];
        }
    }

    if (positive.size() != 4) {
        return fit;
    }
    for (size_t j = 1; j < 4; ++j) {
        if (fit.counts[j] == fit.counts[j - 1]) {
            return fit;
        }
    }
    const double median = 0.5 * (positive[1] + positive[2]);
    fit.spreadPercent =
        100.0 * (positive.back() - positive.front()) / median;
    fit.stable = std::isfinite(fit.spreadPercent)
        && fit.spreadPercent <= 10.0;
    return fit;
}


double EEDFTwoTermApproximation::automaticDecayPreservingMaxEnergy(
    double targetDecayDecades) const
{
    const double currentMax = m_gridEdge.back();
    if (m_points < 2 || !std::isfinite(targetDecayDecades)
        || targetDecayDecades <= 0.0) {
        return currentMax;
    }

    constexpr double usableFloor = 1e-280;
    double maxF = 0.0;
    for (size_t i = 0; i < m_points; ++i) {
        const double f = m_f0(i);
        if (std::isfinite(f) && f > usableFloor) {
            maxF = std::max(maxF, f);
        }
    }
    if (!(maxF > 0.0)) {
        return currentMax;
    }

    const double targetLog = std::log(maxF)
        - targetDecayDecades * std::log(10.0);

    // Search right-to-left for a crossing actually resolved by Boltzmann.
    // Never infer a crossing from a numerical zero or floored endpoint.
    for (size_t i = m_points - 1; i > 0; --i) {
        const double leftF = m_f0(i - 1);
        const double rightF = m_f0(i);
        if (!std::isfinite(leftF) || !std::isfinite(rightF)
            || leftF <= usableFloor || rightF <= usableFloor) {
            continue;
        }
        const double yl = std::log(leftF);
        const double yr = std::log(rightF);
        if (yl > targetLog && yr <= targetLog && yr < yl) {
            const double fraction = (targetLog - yl) / (yr - yl);
            const double crossing = m_gridCenter[i - 1]
                + fraction * (m_gridCenter[i] - m_gridCenter[i - 1]);
            const double halfLastCell = currentMax
                - m_gridCenter[m_points - 1];
            return std::min(currentMax,
                std::max(1e-8, crossing + halfLastCell));
        }
    }
    return currentMax;
}

double EEDFTwoTermApproximation::automaticPredictorDecayDecades() const
{
    if (m_points == 0) {
        return 0.0;
    }
    constexpr double usableFloor = 1e-280;
    double maxF = 0.0;
    double rightF = 0.0;
    for (size_t i = 0; i < m_points; ++i) {
        const double f = m_f0(i);
        if (std::isfinite(f) && f > usableFloor) {
            maxF = std::max(maxF, f);
            rightF = f; // last actually resolved value above the floor
        }
    }
    if (!(rightF > 0.0) || !(maxF > 0.0)) {
        return 0.0;
    }
    return std::log10(maxF) - std::log10(rightF);
}

double EEDFTwoTermApproximation::automaticEedfAtEnergy(
    double energy,
    double tailSlope) const
{
    if (m_points == 0 ||
        !std::isfinite(energy) ||
        energy < 0.0) {

        return 0.0;
    }

    const double fFloor = 1e-300;

    if (energy <= m_gridCenter[0]) {
        return std::max(
            m_f0(0),
            fFloor);
    }

    const double lastEnergy =
        m_gridCenter[m_points - 1];

    const double lastValue =
        std::max(
            m_f0(m_points - 1),
            fFloor);

    if (energy >= lastEnergy) {

        if (!std::isfinite(tailSlope) ||
            tailSlope <= 0.0) {
            return fFloor;
        }

        const double logValue =
            std::log(lastValue)
            - tailSlope
            * (energy - lastEnergy);

        if (logValue < -745.0) {
            return 0.0;
        }

        return std::exp(logValue);
    }

    const double* begin =
        m_gridCenter.data();

    const double* end =
        m_gridCenter.data()
        + m_gridCenter.size();

    const double* upper =
        std::upper_bound(
            begin,
            end,
            energy);

    const size_t i1 =
        static_cast<size_t>(
            upper - begin);

    const size_t i0 = i1 - 1;

    const double x0 =
        m_gridCenter[i0];

    const double x1 =
        m_gridCenter[i1];

    const double y0 =
        std::log(
            std::max(m_f0(i0), fFloor));

    const double y1 =
        std::log(
            std::max(m_f0(i1), fFloor));

    const double t =
        (energy - x0)
        / (x1 - x0);

    const double logValue =
        y0 + t * (y1 - y0);

    return std::exp(logValue);
}

double EEDFTwoTermApproximation::automaticExtrapolatedEedf(
    double energy, double lastEnergy, double lastValue,
    const AutomaticTailFit& fit) const
{
    constexpr double floor = 1e-300;
    if (!(std::isfinite(energy) && std::isfinite(lastEnergy)
        && std::isfinite(lastValue) && lastValue > 0.0)) {
        return floor;
    }
    const double fLast = std::max(lastValue, floor);
    const double delta = std::max(0.0, energy - lastEnergy);
    if (delta == 0.0) {
        return fLast;
    }

    // Stable fit: retain the original slowest positive core slope.
    if (fit.stable && std::isfinite(fit.conservativeSlope)
        && fit.conservativeSlope > 0.0) {
        return std::max(floor, fLast *
            std::exp(-fit.conservativeSlope * delta));
    }

    // Unstable fit: arithmetic mean of extrapolated EEDFs, not mean slope.
    // Only distinct, valid sample-window lengths contribute.
    double sum = 0.0;
    size_t count = 0;
    size_t lastWindowCount = 0;
    for (size_t k = 0; k < fit.slopes.size(); ++k) {
        const double beta = fit.slopes[k];
        if (!(std::isfinite(beta) && beta > 0.0)
            || fit.counts[k] < 6
            || fit.counts[k] == lastWindowCount) {
            continue;
        }
        sum += std::exp(-beta * delta);
        ++count;
        lastWindowCount = fit.counts[k];
    }
    if (count == 0) {
        return floor; // No defensible extrapolating slope.
    }
    return std::max(floor, fLast * (sum / count));
}


double EEDFTwoTermApproximation::automaticEedfAtEnergy(
    double energy, const AutomaticTailFit& fit) const
{
    if (m_points == 0 || !std::isfinite(energy) || energy < 0.0) {
        return 0.0;
    }
    const double lastEnergy = m_gridCenter[m_points - 1];
    if (energy >= lastEnergy) {
        return automaticExtrapolatedEedf(
            energy, lastEnergy, m_f0(m_points - 1), fit);
    }
    // Inside the resolved domain: preserve the original log interpolation.
    // The scalar-slope variant is retained exclusively for J_tail.
    return automaticEedfAtEnergy(energy, NAN);
}


vector<double>
EEDFTwoTermApproximation::automaticDiagnosticGrid(
    double maxEnergy,
    size_t scaffoldPoints) const
{
    vector<double> x;

    scaffoldPoints =
        std::max<size_t>(
            scaffoldPoints,
            16);

    x.reserve(
        scaffoldPoints
        + m_gridEdge.size()
        + 64);

    x.push_back(0.0);
    x.push_back(maxEnergy);

    // Smooth stretched background grid.
    const double Es =
        m_automaticBootstrapStretchEnergy;

    const double logarithmicSpan =
        std::log1p(maxEnergy / Es);

    for (size_t i = 0;
         i <= scaffoldPoints;
         i++) {

        const double u =
            static_cast<double>(i)
            / static_cast<double>(
                scaffoldPoints);

        const double eps =
            Es * std::expm1(
                u * logarithmicSpan);

        if (eps > 0.0 &&
            eps < maxEnergy) {
            x.push_back(eps);
        }
    }

    // Current EEDF grid.
    for (double eps : m_gridEdge) {
        if (eps > 0.0 &&
            eps < maxEnergy) {
            x.push_back(eps);
        }
    }

    for (Eigen::Index i = 0;
         i < m_gridCenter.size();
         i++) {

        const double eps =
            m_gridCenter[i];

        if (eps > 0.0 &&
            eps < maxEnergy) {
            x.push_back(eps);
        }
    }

    // Native nodes of every collision table.
    //
    // They enrich the diagnostic integration only. They are NOT forced to
    // become EEDF-grid points.
    for (size_t k = 0;
         k < m_phase->nCollisions();
         k++) {

        const auto rate =
            m_phase->collisionRate(k);

        const auto levels =
            rate->energyLevels();

        for (double eps : levels) {
            if (eps > 0.0 &&
                eps < maxEnergy) {
                x.push_back(eps);
            }
        }

        const double threshold = rate->threshold();
        auto addGuardSamples = [&](double center) {
            if (!(center > 0.0 && center < maxEnergy)) return;
            const double width=std::max(0.02,0.05*center);
            for (double z : {-3.0,-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0,3.0}) {
                const double e=center+z*width;
                if(e>0.0 && e<maxEnergy) x.push_back(e);
            }
        };
        if (rate->kind() != "elastic" && rate->kind() != "effective") {
            addGuardSamples(threshold);
            const auto sigmas=rate->crossSections();
            if (levels.size() == sigmas.size()) {
                for(size_t j=0;j<levels.size();++j) {
                    if(sigmas[j]>0.0) {
                        addGuardSamples(j>0 && sigmas[j-1]<=0.0
                            ? levels[j-1] : levels[j]);
                        break;
                    }
                }
            }
        } else if (threshold > 0.0 && threshold < maxEnergy) {
            x.push_back(threshold);
        }
    }

    std::sort(
        x.begin(),
        x.end());

    const double tolerance =
        1e-12
        * std::max(1.0, maxEnergy);

    vector<double> filtered;
    filtered.reserve(x.size());

    for (double value : x) {
        if (filtered.empty() ||
            value
                > filtered.back()
                + tolerance) {

            filtered.push_back(value);
        }
    }

    if (filtered.empty() ||
        filtered.front() > 0.0) {
        filtered.insert(
            filtered.begin(),
            0.0);
    } else {
        filtered.front() = 0.0;
    }

    if (maxEnergy
        - filtered.back()
        > tolerance) {

        filtered.push_back(maxEnergy);

    } else {
        filtered.back() = maxEnergy;
    }

    return filtered;
}

double EEDFTwoTermApproximation::
automaticElectronTemperatureEv() const
{
    if (m_points < 2) {
        throw CanteraError(
            "EEDFTwoTermApproximation::automaticElectronTemperatureEv",
            "Automatic Te evaluation requires at least two cells.");
    }

    vector<double> centerEnergy(
        m_gridCenter.data(),
        m_gridCenter.data()
        + m_gridCenter.size());

    vector<double> centerF(
        m_f0.data(),
        m_f0.data()
        + m_f0.size());

    const size_t nEdges =
        m_gridEdge.size();

    Eigen::VectorXd edgeF(nEdges);
    Eigen::VectorXd eps32(nEdges);
    Eigen::VectorXd eps52(nEdges);

    for (size_t i = 0;
         i < nEdges;
         i++) {

        const double eps =
            m_gridEdge[i];

        edgeF[i] =
            linearInterp(
                eps,
                centerEnergy,
                centerF);

        eps32[i] =
            std::pow(eps, 1.5);

        eps52[i] =
            std::pow(eps, 2.5);
    }

    double normalization =
        2.0 / 3.0
        * numericalQuadrature(
            "simpson",
            edgeF,
            eps32);

    double energyMoment =
        2.0 / 5.0
        * numericalQuadrature(
            "simpson",
            edgeF,
            eps52);

    if (!std::isfinite(normalization) ||
        normalization <= 0.0 ||
        !std::isfinite(energyMoment) ||
        energyMoment <= 0.0) {

        normalization =
            2.0 / 3.0
            * numericalQuadrature(
                "trapezoidal",
                edgeF,
                eps32);

        energyMoment =
            2.0 / 5.0
            * numericalQuadrature(
                "trapezoidal",
                edgeF,
                eps52);
    }

    if (!std::isfinite(normalization) ||
        normalization <= 0.0 ||
        !std::isfinite(energyMoment) ||
        energyMoment <= 0.0) {

        throw CanteraError(
            "EEDFTwoTermApproximation::automaticElectronTemperatureEv",
            "Unable to compute a valid electron temperature.");
    }

    const double meanEnergy =
        energyMoment / normalization;

    // k_B Te / e expressed in eV.
    return
        2.0 / 3.0
        * meanEnergy;
}

bool EEDFTwoTermApproximation::automaticGridCollisionObservableActive(
    size_t k) const
{
    // Always use RAW phase mole fractions, not normalized target fractions:
    // 0.001 means 0.1 mol% of the physical mixture.
    return m_phase->moleFraction(m_phase->targetIndex(k))
        >= m_automaticMinimumObservableTargetMoleFraction;
}

EEDFTwoTermApproximation::AutomaticGridSnapshot
EEDFTwoTermApproximation::automaticGridSnapshot()
{
    AutomaticGridSnapshot snapshot;

    snapshot.electronTemperatureEv =
        automaticElectronTemperatureEv();

    snapshot.collisionRates.assign(
        m_phase->nCollisions(),
        0.0);

    for (size_t k = 0;
         k < m_phase->nCollisions();
         k++) {

        if (!automaticGridCollisionObservableActive(k)) {
            continue;
        }
        snapshot.collisionRates[k] =
            forwardRateCoefficientFromEEDF(
                m_f0,
                k);
    }

    return snapshot;
}

double EEDFTwoTermApproximation::
automaticGridConvergenceObjective(
    const AutomaticGridSnapshot& previous,
    const AutomaticGridSnapshot& current,
    size_t& worstCollision,
    double& electronTemperatureError) const
{
    worstCollision = npos;

    const double TeScale =
        m_automaticElectronTemperatureAtolEv
        + m_automaticElectronTemperatureRtol
        * std::max(
            std::abs(previous.electronTemperatureEv),
            std::abs(current.electronTemperatureEv));

    electronTemperatureError =
        std::abs(
            current.electronTemperatureEv
            - previous.electronTemperatureEv)
        / TeScale;

    double objective =
        electronTemperatureError;

    if (previous.collisionRates.size()
        != current.collisionRates.size()) {

        return INFINITY;
    }

    for (size_t k = 0;
         k < current.collisionRates.size();
         k++) {

        if (!automaticGridCollisionObservableActive(k)) {
            continue;
        }

        const double oldRate =
            previous.collisionRates[k];

        const double newRate =
            current.collisionRates[k];

        if (!std::isfinite(oldRate) ||
            !std::isfinite(newRate)) {
            return INFINITY;
        }

        const double scale =
            m_automaticCollisionRateRtol
            * (
                m_automaticCollisionRateFloorM3s
                + std::max(
                    std::abs(oldRate),
                    std::abs(newRate))
            );

        const double error =
            std::abs(newRate - oldRate)
            / scale;

        if (error > objective) {
            objective = error;
            worstCollision = k;
        }
    }

    return objective;
}

double EEDFTwoTermApproximation::automaticTailObjective(
    double cutoffEnergy,
    const AutomaticGridSnapshot& snapshot,
    double tailSlope)
{
    if (!std::isfinite(tailSlope) ||
        tailSlope <= 0.0) {
        return INFINITY;
    }

    const double lastCenter =
        m_gridCenter[m_points - 1];

    // Continue the fitted exponential by the user-requested number
    // of additional decades.
    const double extrapolationLength =
        m_automaticTailExtrapolationDecades
        * std::log(10.0)
        / tailSlope;

    const double integrationEnd =
        std::max(
            cutoffEnergy,
            lastCenter
            + extrapolationLength);

    vector<double> x =
        automaticDiagnosticGrid(
            integrationEnd,
            std::max<size_t>(
                4 * m_automaticFinalCells,
                200));

    x.push_back(cutoffEnergy);

    std::sort(
        x.begin(),
        x.end());

    x.erase(
        std::unique(
            x.begin(),
            x.end()),
        x.end());

    const double I1 =
        norm(m_f0, m_gridCenter);

    if (!std::isfinite(I1) ||
        I1 <= 0.0) {
        return INFINITY;
    }

    const double meanEnergy =
        1.5
        * snapshot.electronTemperatureEv;

    const double TeScale =
        m_automaticElectronTemperatureAtolEv
        + m_automaticElectronTemperatureRtol
        * std::abs(
            snapshot.electronTemperatureEv);

    vector<double> rateTail(
        m_phase->nCollisions(),
        0.0);

    double temperatureTail = 0.0;

    bool havePrevious = false;

    double previousEnergy = 0.0;
    double previousTemperatureKernel = 0.0;

    vector<double> previousRateKernel(
        m_phase->nCollisions(),
        0.0);

    for (double eps : x) {

        if (eps < cutoffEnergy) {
            continue;
        }

        const double f =
            automaticEedfAtEnergy(
                eps,
                tailSlope);

        const double sqrtE =
            std::sqrt(
                std::max(eps, 0.0));

        const double temperatureSensitivity =
            (2.0 / 3.0)
            / I1
            * sqrtE
            * (eps - meanEnergy);

        const double temperatureKernel =
            std::abs(
                temperatureSensitivity)
            * f;

        vector<double> currentRateKernel(
            m_phase->nCollisions(),
            0.0);

        for (size_t k = 0;
             k < m_phase->nCollisions();
             k++) {

            if (!automaticGridCollisionObservableActive(k)) {
                continue;
            }

            const auto rate =
                m_phase->collisionRate(k);

            const auto energy =
                rate->energyLevels();

            const auto sigma =
                rate->crossSections();

            const double crossSection =
                linearInterpCrossSectionZeroOutside(
                    eps,
                    energy,
                    sigma);

            const double K =
                snapshot.collisionRates[k];

            const double sensitivity =
                (
                    m_gamma
                    * eps
                    * crossSection
                    - K
                    * sqrtE
                ) / I1;

            currentRateKernel[k] =
                std::abs(sensitivity)
                * f;
        }

        if (havePrevious) {

            const double dE =
                eps - previousEnergy;

            temperatureTail +=
                0.5
                * dE
                * (
                    previousTemperatureKernel
                    + temperatureKernel
                );

            for (size_t k = 0;
                 k < rateTail.size();
                 k++) {

                rateTail[k] +=
                    0.5
                    * dE
                    * (
                        previousRateKernel[k]
                        + currentRateKernel[k]
                    );
            }
        }

        previousEnergy = eps;

        previousTemperatureKernel =
            temperatureKernel;

        previousRateKernel.swap(
            currentRateKernel);

        havePrevious = true;
    }

    double objective =
        temperatureTail
        / TeScale;

    for (size_t k = 0;
         k < rateTail.size();
         k++) {

        if (!automaticGridCollisionObservableActive(k)) {
            continue;
        }

        const double scale =
            m_automaticCollisionRateRtol
            * (
                m_automaticCollisionRateFloorM3s
                + std::abs(
                    snapshot.collisionRates[k])
            );

        objective =
            std::max(
                objective,
                rateTail[k] / scale);
    }

    return objective;
}

double EEDFTwoTermApproximation::chooseAutomaticMaxEnergy()
{
    const double currentMax = m_gridEdge.back();
    const double measuredDecay = automaticPredictorDecayDecades();
    const double requiredDecay = m_automaticMinimumPredictorDecayDecades;

    if (measuredDecay < requiredDecay) {
        // This is the ONLY situation in which decay drives domain growth.
        // A stable exponential slope is a predictor, never a tail-error proof.
        const AutomaticTailFit fit = automaticTailFit();
        if (fit.stable && std::isfinite(fit.conservativeSlope)
            && fit.conservativeSlope > 0.0) {
            const double missing = requiredDecay - measuredDecay;
            const double predictedGrowth = 1.15 * missing
                * std::log(10.0) / fit.conservativeSlope;
            if (std::isfinite(predictedGrowth) && predictedGrowth > 0.0) {
                return std::min(currentMax * m_automaticMaxEmaxGrowthFactor,
                    currentMax + std::max(0.10 * currentMax, predictedGrowth));
            }
        }
        // No usable extrapolation: geometric growth, bounded by the
        // existing per-iteration maximum-growth configuration.
        return currentMax * std::min(2.0, m_automaticMaxEmaxGrowthFactor);
    }

    // A soft efficiency target, NOT a further acceptance condition.
    // With D_min = 8, aim for approximately 9 resolved decades when
    // shrinking. Never extend an already-acceptable domain to reach 9.
    const double targetDecay = requiredDecay + 1.0;
    if (measuredDecay < targetDecay) {
        return currentMax;
    }

    // Shrinking can only use a crossing already measured on the old grid.
    // Recheck D >= D_min on the corrected Boltzmann solution.
    const double measuredCutoff =
        automaticDecayPreservingMaxEnergy(targetDecay);
    return std::min(currentMax,
        std::max(currentMax * m_automaticMinEmaxShrinkFactor,
            measuredCutoff));
}

vector<double>
EEDFTwoTermApproximation::buildAutomaticGrid(
    double maxEnergy,
    const AutomaticGridSnapshot& snapshot,
    const AutomaticTailFit& tailFit)
{
    // Scalar only used by the legacy debug CSV column, never by J_tail.
    const double tailSlope = tailFit.stable
        ? tailFit.conservativeSlope : NAN;
    const size_t N =
        m_automaticFinalCells;

    vector<double> support =
        automaticDiagnosticGrid(
            maxEnergy,
            std::max<size_t>(
                4 * N,
                300));

    const size_t n =
        support.size();

    const char* diagEnv = std::getenv("AUTO_MONITOR_DIAG");
    const bool monitorDiag = diagEnv && std::string(diagEnv) == "1";


    if (n < 3) {
        return
            makeAutomaticBootstrapGrid(
                maxEnergy,
                N);
    }

    // ------------------------------------------------------------------
    // Temporary monitor diagnostics.
    // PlasmaPhase::reducedElectricField() is expressed in V m^2.
    // 1 Td = 1e-21 V m^2.
    // ------------------------------------------------------------------

    DEBUG_AUTO_GRID_CURRENT_BUILD_ID =
        0;

    const double debugEnTd =
        1.0e21
        * m_phase->reducedElectricField();

    const bool debugThisBuild =
        debugAutomaticGridTarget(
            debugEnTd,
            N);

    size_t debugBuildId =
        0;

    if (debugThisBuild) {

        debugBuildId =
            ++DEBUG_AUTO_GRID_BUILD_COUNTER;

        DEBUG_AUTO_GRID_CURRENT_BUILD_ID =
            debugBuildId;
    }

    const size_t debugPredictorCells =
        m_points;

    const double debugPredictorEmax =
        m_gridEdge.empty()
        ? NAN
        : m_gridEdge.back();

    vector<AutomaticGridDebugPoint>
        debugPoints;

    if (debugThisBuild) {
        debugPoints.resize(n);
    }
    // END OF TEMPORARY DEBUG SCRIPT

    const double I1 =
        norm(m_f0, m_gridCenter);

    if (!std::isfinite(I1) ||
        I1 <= 0.0) {

        throw CanteraError(
            "EEDFTwoTermApproximation::buildAutomaticGrid",
            "Invalid predictor normalization.");
    }

    const double meanEnergy =
        1.5
        * snapshot.electronTemperatureEv;

    const double TeScale =
        m_automaticElectronTemperatureAtolEv
        + m_automaticElectronTemperatureRtol
        * std::abs(
            snapshot.electronTemperatureEv);

    // ------------------------------------------------------------------
    // Curvature of ln(f) on the current EEDF centers.
    // ------------------------------------------------------------------

    vector<double> centerEnergy(
        m_gridCenter.data(),
        m_gridCenter.data()
        + m_gridCenter.size());

    vector<double> curvature(
        m_points,
        0.0);

    const double fFloor =
        1e-300;

    if (m_points >= 3) {

        for (size_t i = 1;
             i + 1 < m_points;
             i++) {

            if (m_f0(i) <= 1e-280) {
                curvature[i] = 0.0;
                continue;
            }

            const double leftSlope =
                (
                    std::log(
                        std::max(
                            m_f0(i),
                            fFloor))
                    -
                    std::log(
                        std::max(
                            m_f0(i - 1),
                            fFloor))
                )
                /
                (
                    m_gridCenter[i]
                    - m_gridCenter[i - 1]
                );

            const double rightSlope =
                (
                    std::log(
                        std::max(
                            m_f0(i + 1),
                            fFloor))
                    -
                    std::log(
                        std::max(
                            m_f0(i),
                            fFloor))
                )
                /
                (
                    m_gridCenter[i + 1]
                    - m_gridCenter[i]
                );

            curvature[i] =
                2.0
                * std::abs(
                    rightSlope
                    - leftSlope)
                /
                (
                    m_gridCenter[i + 1]
                    - m_gridCenter[i - 1]
                );
        }

        curvature.front() =
            curvature[1];

        curvature.back() =
            curvature[m_points - 2];
    }

    // Collision guard sites: declared thresholds and the leftmost boundary
    // of actual positive support of the piecewise-linear table.  When sigma
    // goes from zero at x[j-1] to >0 at x[j], positivity starts right of
    // x[j-1]; do NOT confuse it with x[j].  If first sigma is >0, use x[0].
    struct GuardSite { double energy; double width; };
    vector<GuardSite> guardSites;
    auto pushSite = [&](double center) {
        if (!std::isfinite(center) || center <= 0.0 || center >= maxEnergy) {
            return;
        }
        const double tol = 1e-10 * std::max(1.0, maxEnergy);
        for (const auto& site : guardSites) {
            if (std::abs(center - site.energy) <= tol) {
                return;
            }
        }
        guardSites.push_back({center, std::max(0.02, 0.05 * center)});
    };
    for (size_t k = 0; k < m_phase->nCollisions(); ++k) {
        if (!automaticGridCollisionObservableActive(k)) {
            continue;
        }
        const auto rate = m_phase->collisionRate(k);
        if (rate->kind() == "elastic" || rate->kind() == "effective") {
            continue;
        }
        pushSite(rate->threshold());
        const auto levels = rate->energyLevels();
        const auto sigma = rate->crossSections();
        if (levels.size() != sigma.size()) {
            throw CanteraError("EEDFTwoTermApproximation::buildAutomaticGrid",
                "Collision cross-section table length mismatch.");
        }
        for (size_t j = 0; j < levels.size(); ++j) {
            if (sigma[j] > 0.0) {
                pushSite(j == 0 ? levels[0] :
                    (sigma[j-1] <= 0.0 ? levels[j-1] : levels[j]));
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Observable monitor.
    // ------------------------------------------------------------------

    vector<double> monitor(n, 0.0);
    vector<double> adaptiveShape(n, 0.0);
    vector<double> safeFloor(n, 0.0);
    vector<double> importanceCurvature(n, 0.0);

    const double baseDensity =
        static_cast<double>(N)
        / maxEnergy;

    for (size_t i = 0;
         i < n;
         i++) {

        const double eps =
            support[i];

        const double f =
            automaticEedfAtEnergy(
                eps,
                tailFit);

        const double sqrtE =
            std::sqrt(
                std::max(eps, 0.0));

        const double localCurvature =
            linearInterpBounded(
                eps,
                centerEnergy,
                curvature,
                curvature.front(),
                0.0);

        // NORMAL CODE WITHOUT DEBUG
        // // Te sensitivity.
        // const double temperatureSensitivity =
        //     (2.0 / 3.0)
        //     / I1
        //     * sqrtE
        //     * (eps - meanEnergy);

        // double importance =
        //     std::abs(
        //         temperatureSensitivity)
        //     * f
        //     / TeScale;

        // // Every collision enters independently.
        // for (size_t k = 0;
        //      k < m_phase->nCollisions();
        //      k++) {

        //     const auto rate =
        //         m_phase->collisionRate(k);

        //     const double sigma =
        //         linearInterpCrossSectionZeroOutside(
        //             eps,
        //             rate->energyLevels(),
        //             rate->crossSections());

        //     const double K =
        //         snapshot.collisionRates[k];

        //     const double sensitivity =
        //         (
        //             m_gamma
        //             * eps
        //             * sigma
        //             - K
        //             * sqrtE
        //         ) / I1;

        //     const double scale =
        //         m_automaticCollisionRateRtol
        //         * (
        //             m_automaticCollisionRateFloorM3s
        //             + std::abs(K)
        //         );

        //     const double collisionImportance =
        //         std::abs(sensitivity)
        //         * f
        //         / scale;

        //     importance =
        //         std::max(
        //             importance,
        //             collisionImportance);
        // }

        // code snippet with debug
        // Te sensitivity.
        const double temperatureSensitivity =
            (2.0 / 3.0)
            / I1
            * sqrtE
            * (eps - meanEnergy);

        const double temperatureImportance =
            std::abs(
                temperatureSensitivity)
            * f
            / TeScale;

        double importance =
            temperatureImportance;


        // Diagnostic information about which collision controls
        // the local monitor.
        double maxCollisionImportance =
            0.0;

        size_t maxCollisionIndex =
            npos;

        string maxCollisionName;


        // Selected difficult collision processes.
        double n2v1resImportance = 0.0;
        double n2v1resSigma = 0.0;

        double o2v1resImportance = 0.0;
        double o2v1resSigma = 0.0;

        double o2v2resImportance = 0.0;
        double o2v2resSigma = 0.0;

        double o2v1Importance = 0.0;
        double o2v1Sigma = 0.0;


        // Only sufficiently present collision targets drive the mesh.
        for (size_t k = 0;
            k < m_phase->nCollisions();
            k++) {

            if (!automaticGridCollisionObservableActive(k)) {
                continue;
            }

            const auto rate =
                m_phase->collisionRate(k);

            const double sigma =
                linearInterpCrossSectionZeroOutside(
                    eps,
                    rate->energyLevels(),
                    rate->crossSections());

            const double K =
                snapshot.collisionRates[k];

            const double sensitivity =
                (
                    m_gamma
                    * eps
                    * sigma
                    - K
                    * sqrtE
                ) / I1;

            const double scale =
                m_automaticCollisionRateRtol
                * (
                    m_automaticCollisionRateFloorM3s
                    + std::abs(K)
                );

            const double collisionImportance =
                std::abs(sensitivity)
                * f
                / scale;


            if (debugThisBuild) {

                string collisionName =
                    rate->collisionName();

                if (collisionName.empty()) {
                    collisionName =
                        m_phase
                        ->collision(k)
                        ->equation();
                }

                if (collisionImportance
                        > maxCollisionImportance) {

                    maxCollisionImportance =
                        collisionImportance;

                    maxCollisionIndex =
                        k;

                    maxCollisionName =
                        collisionName;
                }


                if (collisionName.find(
                        "N2(v1res)")
                        != string::npos) {

                    n2v1resImportance =
                        collisionImportance;

                    n2v1resSigma =
                        sigma;
                }


                if (collisionName.find(
                        "O2(v1res)")
                        != string::npos) {

                    o2v1resImportance =
                        collisionImportance;

                    o2v1resSigma =
                        sigma;
                }


                if (collisionName.find(
                        "O2(v2res)")
                        != string::npos) {

                    o2v2resImportance =
                        collisionImportance;

                    o2v2resSigma =
                        sigma;
                }


                // "O2(v1)" does NOT match "O2(v1res)"
                // because the closing parenthesis is part of the token.
                if (collisionName.find(
                        "O2(v1)")
                        != string::npos) {

                    o2v1Importance =
                        collisionImportance;

                    o2v1Sigma =
                        sigma;
                }
            }


            importance =
                std::max(
                    importance,
                    collisionImportance);
        }


        if (debugThisBuild) {

            AutomaticGridDebugPoint& dbg =
                debugPoints[i];

            dbg.f =
                f;

            dbg.curvature =
                localCurvature;

            dbg.temperatureImportance =
                temperatureImportance;

            dbg.maxCollisionImportance =
                maxCollisionImportance;

            dbg.maxCollisionName =
                maxCollisionName;

            if (maxCollisionImportance
                    > temperatureImportance) {

                dbg.controllerName =
                    maxCollisionName;

            } else {

                dbg.controllerName =
                    "electron temperature";
            }

            dbg.n2v1resImportance =
                n2v1resImportance;

            dbg.n2v1resSigma =
                n2v1resSigma;

            dbg.o2v1resImportance =
                o2v1resImportance;

            dbg.o2v1resSigma =
                o2v1resSigma;

            dbg.o2v2resImportance =
                o2v2resImportance;

            dbg.o2v2resSigma =
                o2v2resSigma;

            dbg.o2v1Importance =
                o2v1Importance;

            dbg.o2v1Sigma =
                o2v1Sigma;

            dbg.totalImportance =
                importance;
        }
        // end of debug code snippet 

        // Local error model:
        //
        // delta O_cell ~ importance * |(ln f)''| * h^3
        //
        // hence the equidistribution density:
        //
        // rho ~ [ importance * curvature ]^(1/3).
        const double adaptiveDensity =
            std::cbrt(
                std::max(
                    0.0,
                    importance
                    * localCurvature));

        // A = I*kappa, and a = A^(1/3).  Density is constructed later
        // by constrained optimization, NOT by summing ad hoc densities.
        const double baselineDensity =
            m_automaticBaselineMetricFraction * baseDensity;
        double thresholdGuard = 0.0;
        for (const auto& site : guardSites) {
            const double z = (eps - site.energy) / site.width;
            thresholdGuard = std::max(thresholdGuard,
                                      std::exp(-0.5 * z * z));
        }
        const double thresholdGuardDensity =
            m_automaticThresholdGuardFraction * baseDensity * thresholdGuard;
        const double safeDensity = baselineDensity + thresholdGuardDensity;
        adaptiveShape[i] = adaptiveDensity;
        safeFloor[i] = safeDensity;
        monitor[i] = adaptiveDensity; // provisional, before KKT normalization
        if (monitorDiag) {
            importanceCurvature[i] = std::max(0.0,importance*localCurvature);
        }

        if (debugThisBuild) {

            AutomaticGridDebugPoint& dbg =
                debugPoints[i];

            dbg.adaptiveDensity =
                adaptiveDensity;

            dbg.baselineDensity =
                baselineDensity;

            dbg.thresholdGuard =
                thresholdGuard;

            dbg.thresholdGuardDensity =
                thresholdGuardDensity;

            dbg.rhoBeforeSmoothing =
                adaptiveDensity + safeDensity;
        }

        // END DEBUG
    }

    // Constrained KKT solution rho=max(C*a, floor), with integral rho=N.
    // This replaces the uncalibrated additive densities.
    auto trapz = [&](const vector<double>& values) {
        double mass = 0.0;
        for (size_t i = 1; i < n; ++i) {
            mass += 0.5 * (values[i-1] + values[i])
                  * (support[i] - support[i-1]);
        }
        return mass;
    };
    const double floorMass = trapz(safeFloor);
    if (!(floorMass < static_cast<double>(N)) || !std::isfinite(floorMass)) {
        throw CanteraError("EEDFTwoTermApproximation::buildAutomaticGrid",
            "Constrained guard is infeasible: integral(floor)={} >= N={}.",
            floorMass, N);
    }
    auto projectConstrained = [&](const vector<double>& shape) {
        vector<double> trial(n);
        auto massAt = [&](double c) {
            for (size_t i = 0; i < n; ++i) {
                trial[i] = std::max(safeFloor[i], c*shape[i]);
            }
            return trapz(trial);
        };
        double high=1.0;
        while (massAt(high) < static_cast<double>(N)) {
            if (high > 1e250) {
                throw CanteraError("EEDFTwoTermApproximation::buildAutomaticGrid",
                    "Cannot normalize adaptive monitor (zero or tiny shape).");
            }
            high *= 2.0;
        }
        double low=0.0;
        for (size_t iter=0; iter<75; ++iter) {
            const double mid=0.5*(low+high);
            if (massAt(mid) < static_cast<double>(N)) {
                low=mid;
            } else {
                high=mid;
            }
        }
        massAt(high);
        return trial;
    };
    const double shapeIntegral=trapz(adaptiveShape);
    if (shapeIntegral <= 0.0) {
        // Degenerate A=0 everywhere: uniform density satisfies the floor
        // when f_base+f_guard<1. No meaningful unique KKT optimizer.
        monitor.assign(n, static_cast<double>(N)/maxEnergy);
        for (size_t i=0; i<n; ++i) {
            monitor[i]=std::max(monitor[i], safeFloor[i]);
        }
        const double integral=trapz(monitor);
        for (double& value : monitor) value *= static_cast<double>(N)/integral;
    } else {
        monitor=projectConstrained(adaptiveShape);
    }
    const vector<double> optimalMonitor=monitor;

    // Legacy smoother is optional. Retain its exact coefficients for an
    // A/B test, then re-project onto the constraints and the N-cell budget.
    if (m_automaticMonitorSmoothing) {
        for (size_t pass=0; pass<2; ++pass) {
            vector<double> next=monitor;
            for (size_t i=1; i+1<n; ++i) {
                next[i]=0.25*monitor[i-1]+0.50*monitor[i]
                       +0.25*monitor[i+1];
            }
            monitor.swap(next);
        }
        // Re-projection onto rho>=safeFloor, integral rho=N.  A global
        // rescaling of max(smoothed,floor) is NOT safe: it can violate floor.
        const vector<double> smoothed=monitor;
        const double freeMass=static_cast<double>(N)-floorMass;
        double low=0.0, high=1.0;
        auto massShift = [&](double t) {
            for (size_t i=0; i<n; ++i)
                monitor[i]=safeFloor[i]+t*std::max(0.0,smoothed[i]-safeFloor[i]);
            return trapz(monitor);
        };
        while(massShift(high)<static_cast<double>(N)) high*=2.0;
        for(size_t j=0;j<75;++j) {
            double mid=(low+high)*0.5;
            if(massShift(mid)<static_cast<double>(N)) low=mid;
            else high=mid;
        }
        massShift(high);
        (void)freeMass;
    }

    if (monitorDiag) {
        // Exact modeled objective and pointwise relative deformation.
        auto objective = [&](const vector<double>& dens) {
            vector<double> integrand(n);
            for(size_t i=0;i<n;++i)
                integrand[i]=importanceCurvature[i]/(dens[i]*dens[i]);
            return trapz(integrand);
        };
        const double objOptimal=objective(optimalMonitor);
        const double objUsed=objective(monitor);
        double minSafeRatio=std::numeric_limits<double>::infinity();
        double maxRelChange=0.0;
        for(size_t i=0;i<n;++i) {
            minSafeRatio=std::min(minSafeRatio,monitor[i]/safeFloor[i]);
            maxRelChange=std::max(maxRelChange,
                std::abs(monitor[i]-optimalMonitor[i])/optimalMonitor[i]);
        }
        std::ofstream detail("automatic_monitor_v7.csv",std::ios::app);
        if(detail.tellp()==std::streampos(0)) {
            detail << "build,N,Emax,smooth,energy_eV,A,adaptive_shape,safe_floor,optimal,final\n";
        }
        static thread_local size_t v7BuildCounter=0;
        ++v7BuildCounter;
        detail<<std::setprecision(17);
        for(size_t i=0;i<n;++i) {
            detail<<v7BuildCounter<<","<<N<<","<<maxEnergy<<","
                <<m_automaticMonitorSmoothing<<","<<support[i]<<","
                <<importanceCurvature[i]<<","<<adaptiveShape[i]<<","
                <<safeFloor[i]<<","<<optimalMonitor[i]<<","<<monitor[i]<<"\n";
        }
        std::ofstream summary("automatic_monitor_v7_summary.csv",std::ios::app);
        if(summary.tellp()==std::streampos(0)) {
            summary<<"build,N,Emax,smooth,guard_sites,floor_mass,optimal_mass,final_mass,objective_optimal,objective_final,objective_ratio,min_floor_ratio,max_rel_change\n";
        }
        summary<<std::setprecision(17)<<v7BuildCounter<<","<<N<<","<<maxEnergy<<","
               <<m_automaticMonitorSmoothing<<","<<guardSites.size()<<","<<floorMass<<","
               <<trapz(optimalMonitor)<<","<<trapz(monitor)<<","<<objOptimal<<","<<objUsed<<","
               <<(objOptimal>0?objUsed/objOptimal:1.0)<<","<<minSafeRatio<<","<<maxRelChange<<"\n";
    }

    // ------------------------------------------------------------------
    // Cumulative monitor integral.
    // ------------------------------------------------------------------

    vector<double> cumulative(
        n,
        0.0);

    for (size_t i = 1;
         i < n;
         i++) {

        cumulative[i] =
            cumulative[i - 1]
            + 0.5
            * (
                monitor[i - 1]
                + monitor[i]
            )
            * (
                support[i]
                - support[i - 1]
            );
    }

    const double total =
        cumulative.back();

    if (!std::isfinite(total) ||
        total <= 0.0) {

        return
            makeAutomaticBootstrapGrid(
                maxEnergy,
                N);
    }

    // ------------------------------------------------------------------
    // Equidistribution.
    // ------------------------------------------------------------------

    vector<double> edges(
        N + 1,
        0.0);

    edges.front() = 0.0;
    edges.back() = maxEnergy;

    for (size_t j = 1;
         j < N;
         j++) {

        const double target =
            total
            * static_cast<double>(j)
            / static_cast<double>(N);

        auto upper =
            std::lower_bound(
                cumulative.begin(),
                cumulative.end(),
                target);

        size_t i1 =
            static_cast<size_t>(
                upper
                - cumulative.begin());

        if (i1 == 0) {
            edges[j] =
                support.front();
            continue;
        }

        if (i1 >= n) {
            edges[j] =
                support.back();
            continue;
        }

        const size_t i0 =
            i1 - 1;

        const double c0 =
            cumulative[i0];

        const double c1 =
            cumulative[i1];

        const double fraction =
            c1 > c0
            ? (target - c0)
                / (c1 - c0)
            : 0.0;

        edges[j] =
            support[i0]
            + fraction
            * (
                support[i1]
                - support[i0]
            );
    }

    // Strict monotonicity safety.
    for (size_t j = 1;
         j <= N;
         j++) {

        if (!(edges[j] >
              edges[j - 1])) {

            throw CanteraError(
                "EEDFTwoTermApproximation::buildAutomaticGrid",
                "Automatic grid construction produced non-increasing "
                "edge {}: {} <= {}.",
                j,
                edges[j],
                edges[j - 1]);
        }
    }


    // BEGIN DEBUG

    // ------------------------------------------------------------------
    // Temporary Automatic-grid diagnostics.
    //
    // monitor[i] is now the monitor AFTER the two smoothing passes.
    // debugPoints[i].rhoBeforeSmoothing stores the raw monitor.
    // ------------------------------------------------------------------

    if (debugThisBuild) {

        // --------------------------------------------------------------
        // Detailed monitor.
        // --------------------------------------------------------------

        std::ofstream monitorFile(
            AUTO_GRID_MONITOR_DEBUG_FILE,
            std::ios::out | std::ios::app);

        if (monitorFile) {

            if (monitorFile.tellp()
                    == std::streampos(0)) {

                monitorFile
                    << "build_id,"
                    << "pc_iteration,"
                    << "en_Td,"
                    << "predictor_cells,"
                    << "predictor_Emax_eV,"
                    << "final_cells,"
                    << "proposed_Emax_eV,"
                    << "Te_eV,"
                    << "I1,"
                    << "tail_slope,"
                    << "support_index,"
                    << "energy_eV,"
                    << "eedf_predictor,"
                    << "logf_curvature,"
                    << "Te_importance,"
                    << "max_collision_importance,"
                    << "max_collision_name,"
                    << "controller_name,"
                    << "N2_v1res_importance,"
                    << "N2_v1res_sigma,"
                    << "O2_v1res_importance,"
                    << "O2_v1res_sigma,"
                    << "O2_v2res_importance,"
                    << "O2_v2res_sigma,"
                    << "O2_v1_importance,"
                    << "O2_v1_sigma,"
                    << "total_importance,"
                    << "adaptive_density,"
                    << "baseline_density,"
                    << "threshold_guard,"
                    << "threshold_guard_density,"
                    << "rho_before_smoothing,"
                    << "rho_after_smoothing,"
                    << "cumulative_fraction\n";
            }

            monitorFile
                << std::setprecision(17);

            for (size_t i = 0;
                i < n;
                i++) {

                const AutomaticGridDebugPoint& dbg =
                    debugPoints[i];

                const double cumulativeFraction =
                    total > 0.0
                    ? cumulative[i] / total
                    : 0.0;

                monitorFile
                    << debugBuildId << ","
                    << DEBUG_AUTO_GRID_PC_ITERATION << ","
                    << debugEnTd << ","
                    << debugPredictorCells << ","
                    << debugPredictorEmax << ","
                    << N << ","
                    << maxEnergy << ","
                    << snapshot.electronTemperatureEv << ","
                    << I1 << ","
                    << tailSlope << ","
                    << i << ","
                    << support[i] << ","
                    << dbg.f << ","
                    << dbg.curvature << ","
                    << dbg.temperatureImportance << ","
                    << dbg.maxCollisionImportance << ","
                    << debugAutomaticGridCsvSafe(
                        dbg.maxCollisionName) << ","
                    << debugAutomaticGridCsvSafe(
                        dbg.controllerName) << ","
                    << dbg.n2v1resImportance << ","
                    << dbg.n2v1resSigma << ","
                    << dbg.o2v1resImportance << ","
                    << dbg.o2v1resSigma << ","
                    << dbg.o2v2resImportance << ","
                    << dbg.o2v2resSigma << ","
                    << dbg.o2v1Importance << ","
                    << dbg.o2v1Sigma << ","
                    << dbg.totalImportance << ","
                    << dbg.adaptiveDensity << ","
                    << dbg.baselineDensity << ","
                    << dbg.thresholdGuard << ","
                    << dbg.thresholdGuardDensity << ","
                    << dbg.rhoBeforeSmoothing << ","
                    << monitor[i] << ","
                    << cumulativeFraction
                    << "\n";
            }
        }


        // --------------------------------------------------------------
        // Raw fully-equidistributed grid proposed by this monitor.
        // The actually applied relaxed grid will be written later in
        // solveWithAutomaticGrid().
        // --------------------------------------------------------------

        std::ofstream edgeFile(
            AUTO_GRID_EDGES_DEBUG_FILE,
            std::ios::out | std::ios::app);

        if (edgeFile) {

            if (edgeFile.tellp()
                    == std::streampos(0)) {

                edgeFile
                    << "build_id,"
                    << "pc_iteration,"
                    << "en_Td,"
                    << "final_cells,"
                    << "stage,"
                    << "alpha,"
                    << "Emax_eV,"
                    << "edge_index,"
                    << "energy_eV\n";
            }

            edgeFile
                << std::setprecision(17);

            for (size_t j = 0;
                j < edges.size();
                j++) {

                edgeFile
                    << debugBuildId << ","
                    << DEBUG_AUTO_GRID_PC_ITERATION << ","
                    << debugEnTd << ","
                    << N << ","
                    << "proposed,"
                    << 1.0 << ","
                    << maxEnergy << ","
                    << j << ","
                    << edges[j]
                    << "\n";
            }
        }
    }

    // END DEBUG

    if (monitorDiag) {
        double minH=std::numeric_limits<double>::infinity();
        double maxH=0.0, maxNeighborRatio=1.0;
        for(size_t j=0;j<N;++j) {
            const double h=edges[j+1]-edges[j];
            minH=std::min(minH,h);
            maxH=std::max(maxH,h);
            if(j>0) {
                const double prev=edges[j]-edges[j-1];
                maxNeighborRatio=std::max(maxNeighborRatio,
                    std::max(h/prev,prev/h));
            }
        }
        std::ofstream edgeSummary("automatic_edges_v7_summary.csv",std::ios::app);
        if(edgeSummary.tellp()==std::streampos(0))
            edgeSummary<<"N,Emax,smooth,hmin,hmax,Rh\n";
        edgeSummary<<std::setprecision(17)<<N<<","<<maxEnergy<<","
            <<m_automaticMonitorSmoothing<<","<<minH<<","<<maxH<<","<<maxNeighborRatio<<"\n";
        std::ofstream edgeOut("automatic_edges_v7.csv",std::ios::app);
        if(edgeOut.tellp()==std::streampos(0))
            edgeOut<<"N,Emax,smooth,edge_index,energy_eV\n";
        edgeOut<<std::setprecision(17);
        for(size_t j=0;j<=N;++j)
            edgeOut<<N<<","<<maxEnergy<<","<<m_automaticMonitorSmoothing
                   <<","<<j<<","<<edges[j]<<"\n";
    }
    return edges;
}

void EEDFTwoTermApproximation::
projectPreviousEEDFOnAutomaticGrid(
    const Eigen::VectorXd& oldGridCenter,
    const Eigen::VectorXd& oldF0,
    const AutomaticTailFit& tailFit)
{
    if (oldGridCenter.size()
            != oldF0.size() ||
        oldGridCenter.size() < 2) {

        throw CanteraError(
            "EEDFTwoTermApproximation::projectPreviousEEDFOnAutomaticGrid",
            "Invalid previous EEDF.");
    }

    const double fFloor =
        1e-300;

    for (size_t j = 0;
         j < m_points;
         j++) {

        const double eps =
            m_gridCenter[j];

        double value = fFloor;

        if (eps <= oldGridCenter[0]) {

            value =
                std::max(
                    oldF0(0),
                    fFloor);

        } else if (
            eps >=
            oldGridCenter[
                oldGridCenter.size() - 1]) {

            const double oldLastEnergy =
                oldGridCenter[
                    oldGridCenter.size() - 1];

            const double oldLastValue =
                std::max(
                    oldF0(
                        oldF0.size() - 1),
                    fFloor);

            value = automaticExtrapolatedEedf(
                eps, oldLastEnergy, oldLastValue, tailFit);

        } else {

            const double* begin =
                oldGridCenter.data();

            const double* end =
                oldGridCenter.data()
                + oldGridCenter.size();

            const double* upper =
                std::upper_bound(
                    begin,
                    end,
                    eps);

            const size_t i1 =
                static_cast<size_t>(
                    upper - begin);

            const size_t i0 =
                i1 - 1;

            const double x0 =
                oldGridCenter[i0];

            const double x1 =
                oldGridCenter[i1];

            const double y0 =
                std::log(
                    std::max(
                        oldF0(i0),
                        fFloor));

            const double y1 =
                std::log(
                    std::max(
                        oldF0(i1),
                        fFloor));

            const double t =
                (eps - x0)
                / (x1 - x0);

            value =
                std::exp(
                    y0
                    + t
                    * (y1 - y0));
        }

        m_f0(j) =
            std::max(
                value,
                fFloor);
    }

    const double fnorm =
        norm(m_f0, m_gridCenter);

    if (!std::isfinite(fnorm) ||
        fnorm <= 0.0) {

        throw CanteraError(
            "EEDFTwoTermApproximation::projectPreviousEEDFOnAutomaticGrid",
            "Invalid EEDF normalization after automatic remeshing.");
    }

    m_f0 /= fnorm;
}

void EEDFTwoTermApproximation::solveWithAutomaticGrid()
{
    bool predictorReady = false;

    // ------------------------------------------------------------------
    // Prefer the previously converged EEDF during a transient calculation.
    // ------------------------------------------------------------------
    if (m_reusePreviousEEDFAsPredictor && m_has_EEDF &&
        !m_regimeChangedThisCall && !m_forceMaxwellianRestart) {

        try {
            updateCrossSections();

            solveBoltzmannWithSuperElastic(
                m_f0);

            predictorReady = true;

        } catch (const CanteraError& err) {

            if (m_automaticVerbose) {
                writelog(
                    "Automatic EEDF grid: predictor on previous grid failed; "
                    "falling back to bootstrap. Reason: {}\n",
                    err.what());
            }
        }
    }

    // ------------------------------------------------------------------
    // Fresh bootstrap: require a measured decay in [D_min, D_max].
    // A failed solve provides no decay measurement: backtrack the energy
    // domain between the last resolved under-decayed grid and the first
    // failed grid, rather than extrapolating Emax without bound.
    // ------------------------------------------------------------------
    const double maximumDecay =
        m_automaticMaximumPredictorDecayDecades;

    size_t activeCollisionCount = 0;
    for (size_t k = 0; k < m_phase->nCollisions(); ++k) {
        activeCollisionCount += automaticGridCollisionObservableActive(k);
    }
    if (m_automaticVerbose) {
        writelog("[AUTO][TARGET_FILTER] X_min={} (mol_fraction), "
            "selected={}/{} collision observables; Te always included; "
            "D_window=[{},{}]\n",
            m_automaticMinimumObservableTargetMoleFraction,
            activeCollisionCount, m_phase->nCollisions(),
            m_automaticMinimumPredictorDecayDecades, maximumDecay);
    }

    if (!predictorReady) {
        double bootstrapMax = m_automaticBootstrapMaxEnergy;
        double lastUnderDecayedMax = NAN;
        double firstFailedMax = INFINITY;

        for (size_t attempt = 0;
             attempt < m_automaticBootstrapMaxAttempts; ++attempt) {
            try {
                const vector<double> bootstrap =
                    makeAutomaticBootstrapGrid(
                        bootstrapMax, m_automaticBootstrapCells);
                setCustomGrid(bootstrap);
                m_kTeMax = bootstrapMax;
                m_has_EEDF = false;
                setMaxwellianEEDF(m_init_kTe);
                updateCrossSections();
                solveBoltzmannWithSuperElastic(m_f0);

                const double decades = automaticPredictorDecayDecades();
                const AutomaticTailFit fit = automaticTailFit();
                if (m_automaticVerbose) {
                    writelog(
                        "[AUTO][BOOT {}/{}] Emax={} eV decay={} decades "
                        "(min={}) beta_5={} beta_7={} beta_9={} "
                        "beta_10={} spread={} % stable={}\n",
                        attempt + 1, m_automaticBootstrapMaxAttempts,
                        bootstrapMax, decades,
                        m_automaticMinimumPredictorDecayDecades,
                        fit.slopes[0], fit.slopes[1], fit.slopes[2],
                        fit.slopes[3], fit.spreadPercent, fit.stable);
                }
                if (m_automaticVerbose) {
                    writelog("[AUTO][BOOT] extended slopes: "
                        "11%={} 13%={} 15%={} 17%={} 19%={} 20%={} "
                        "stable_extended_up_to={} %\n",
                        fit.slopes[4], fit.slopes[5], fit.slopes[6],
                        fit.slopes[7], fit.slopes[8], fit.slopes[9],
                        fit.extendedStablePercent);
                }
                if (m_automaticVerbose) {
                    writelog("[AUTO][BOOT] fit window points: "
                        "5%={} 7%={} 9%={} 10%={} 11%={} 13%={} "
                        "15%={} 17%={} 19%={} 20%={}\n",
                        fit.counts[0], fit.counts[1], fit.counts[2],
                        fit.counts[3], fit.counts[4], fit.counts[5],
                        fit.counts[6], fit.counts[7], fit.counts[8],
                        fit.counts[9]);
                }

                if (decades >= m_automaticMinimumPredictorDecayDecades
                    && decades <= maximumDecay) {
                    predictorReady = true;
                    if (m_automaticVerbose) {
                        writelog("[AUTO][BOOT] ACCEPT: measured decay={} "
                            "in [{}, {}] decades; slope stable={}.\n",
                            decades, m_automaticMinimumPredictorDecayDecades,
                            maximumDecay, fit.stable);
                    }
                    break;
                }

                if (decades > maximumDecay) {
                    // Cut only at a crossing measured in this EEDF. Leave
                    // a one-decade margin below D_max to avoid remesh jitter.
                    const double targetDecay = maximumDecay - 1.0;
                    const double shortenedMax =
                        automaticDecayPreservingMaxEnergy(targetDecay);
                    if (!std::isfinite(shortenedMax)
                        || !(shortenedMax < bootstrapMax * (1.0 - 1e-10))) {
                        if (m_automaticVerbose) {
                            writelog("[AUTO][BOOT] Cannot shorten Emax={} "
                                "eV: no resolved D={} crossing.\n",
                                bootstrapMax, targetDecay);
                        }
                        break; // report failure; never accept D > D_max
                    }
                    if (m_automaticVerbose) {
                        writelog("[AUTO][BOOT] DECAY_HIGH: D={} > {}; "
                            "shorten Emax {} -> {} eV (target D={}).\n",
                            decades, maximumDecay, bootstrapMax,
                            shortenedMax, targetDecay);
                    }
                    bootstrapMax = shortenedMax;
                    lastUnderDecayedMax = NAN;
                    firstFailedMax = INFINITY;
                    continue; // recompute EEDF; never accept extrapolation
                }

                // Measured D < D_min: this grid has a valid EEDF and is
                // the new lower bound for an energy-domain bracket.
                lastUnderDecayedMax = bootstrapMax;
                if (std::isfinite(firstFailedMax)) {
                    bootstrapMax = 0.5 * (lastUnderDecayedMax
                        + firstFailedMax);
                    if (m_automaticVerbose) {
                        writelog("[AUTO][BOOT] UNDER_DECAY: D={} < {}; "
                            "bisect resolved/failure bracket [{}, {}] "
                            "-> {} eV.\n",
                            decades, m_automaticMinimumPredictorDecayDecades,
                            lastUnderDecayedMax, firstFailedMax, bootstrapMax);
                    }
                } else if (fit.stable && fit.conservativeSlope > 0.0) {
                    // Preserve the existing growth rule when no upper
                    // failure has yet been observed.
                    const double missing =
                        m_automaticMinimumPredictorDecayDecades - decades;
                    const double delta = 1.15 * missing
                        * std::log(10.0) / fit.conservativeSlope;
                    bootstrapMax += std::max(0.10 * bootstrapMax,
                        std::min(delta, bootstrapMax
                            * (m_automaticBootstrapExpansionFactor - 1.0)));
                } else {
                    bootstrapMax *= m_automaticBootstrapExpansionFactor;
                }
            } catch (const CanteraError& err) {
                const double failedAt = bootstrapMax;
                if (m_automaticVerbose) {
                    writelog("[AUTO][BOOT {}/{}] solve failed at Emax={} "
                        "eV: {}\n", attempt + 1,
                        m_automaticBootstrapMaxAttempts, failedAt, err.what());
                }
                if (std::isfinite(lastUnderDecayedMax)
                    && failedAt > lastUnderDecayedMax) {
                    firstFailedMax = std::min(firstFailedMax, failedAt);
                    bootstrapMax = 0.5 * (lastUnderDecayedMax
                        + firstFailedMax);
                    if (m_automaticVerbose) {
                        writelog("[AUTO][BOOT] BACKTRACK: resolved Emax={} "
                            "eV, failed Emax={} eV; retry {} eV.\n",
                            lastUnderDecayedMax, firstFailedMax, bootstrapMax);
                    }
                } else {
                    // No successfully resolved lower bound exists.
                    // Preserve original fallback for this distinct case.
                    bootstrapMax *= m_automaticBootstrapExpansionFactor;
                }
            }
            // Avoid exhausting attempts on indistinguishable endpoints.
            if (!std::isfinite(bootstrapMax) || bootstrapMax <= 0.0
                || (std::isfinite(firstFailedMax)
                    && std::isfinite(lastUnderDecayedMax)
                    && (firstFailedMax - lastUnderDecayedMax)
                        <= 1e-8 * std::max(1.0, lastUnderDecayedMax))) {
                if (m_automaticVerbose) {
                    writelog("[AUTO][BOOT] STOP: Emax bracket exhausted "
                        "or invalid; no acceptable resolved EEDF.\n");
                }
                break;
            }
        }
        if (!predictorReady) {
            throw CanteraError(
                "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                "Bootstrap did not provide a resolved EEDF with at least "
                "{} decades in {} attempts.",
                m_automaticMinimumPredictorDecayDecades,
                m_automaticBootstrapMaxAttempts);
        }

        m_forceMaxwellianRestart = false;
    }

    if (!predictorReady) {
        throw CanteraError(
            "EEDFTwoTermApproximation::solveWithAutomaticGrid",
            "Unable to obtain an initial predictor EEDF.");
    }

    AutomaticGridSnapshot previous =
        automaticGridSnapshot();

    bool accepted = false;

    double lastObjective =
        INFINITY;

    // Optional debug model estimate only; never an acceptance variable.
    double lastTailObjective = NAN;

    for (size_t iteration = 0;
         iteration
            < m_automaticPredictorCorrectorIterations;
         iteration++) {

        const AutomaticTailFit predictorFit = automaticTailFit();
        const double predictorDecay = automaticPredictorDecayDecades();
        if (m_automaticVerbose) {
            writelog("[AUTO][PC {}] predictor decay={} beta5={} beta7={} "
                "beta9={} beta10={} spread={} % stable={}\n",
                iteration + 1, predictorDecay, predictorFit.slopes[0],
                predictorFit.slopes[1], predictorFit.slopes[2],
                predictorFit.slopes[3], predictorFit.spreadPercent,
                predictorFit.stable);
        }

        // Reserved for the existing iteration CSV schema. Not estimated
        // and never consulted for acceptance or domain selection.
        const double predictedTailObjective = NAN;

        const double newMaxEnergy = chooseAutomaticMaxEnergy();

        // ----------------------------------------------------------------------
        // Fully equidistributed grid proposed by the current monitor.
        // ----------------------------------------------------------------------

        // Build the new grid from the resolved EEDF and the numerical
        // continuation of its tail (no J_tail acceptance test).
        DEBUG_AUTO_GRID_PC_ITERATION =
            iteration + 1;

        const vector<double> proposedGrid =
            buildAutomaticGrid(
                newMaxEnergy,
                previous,
                predictorFit);

        // Preserve the old grid and EEDF before remeshing.
        const Eigen::VectorXd oldGridCenter =
            m_gridCenter;

        const vector<double> oldGridEdge =
            m_gridEdge;

        const Eigen::VectorXd oldF0 =
            m_f0;

        // ----------------------------------------------------------------------
        // Adaptive relaxation of the INTERIOR grid positions.
        //
        // Relaxation is performed in normalized energy coordinates:
        //
        //     xi = epsilon / Emax
        //
        // The energy-domain endpoint newMaxEnergy itself is never relaxed.
        //
        // First Automatic remesh:
        //     alpha = 1
        //
        // Subsequent remeshes:
        //
        //     s = max_j |xi_proposed_j - xi_old_j|
        //
        //     alpha = alpha_min
        //             + (1 - alpha_min)
        //             * min(s / s_full, 1)
        //
        // Thus:
        //
        //     large remesh  -> alpha = 1
        //     small remesh  -> alpha -> alpha_min
        //
        // This lets the grid move rapidly when far from the fixed point while
        // damping oscillations once the grid geometry is already close.
        // ----------------------------------------------------------------------

        vector<double> newGrid =
            proposedGrid;

        double appliedGridRelaxation =
            1.0;

        double maxProposedNormalizedGridShift =
            0.0;

        double maxAppliedNormalizedGridShift =
            0.0;

        const bool compatibleGridSizes =
            oldGridEdge.size() ==
            proposedGrid.size()
            && proposedGrid.size() >= 2;

        const bool canRelaxGrid =
            iteration > 0
            && compatibleGridSizes;

        if (canRelaxGrid) {

            const double oldMaxEnergy =
                oldGridEdge.back();

            if (!std::isfinite(oldMaxEnergy) ||
                oldMaxEnergy <= 0.0 ||
                !std::isfinite(newMaxEnergy) ||
                newMaxEnergy <= 0.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                    "Invalid Automatic-grid energy range during relaxation: "
                    "old Emax={}, proposed Emax={}.",
                    oldMaxEnergy,
                    newMaxEnergy);
            }

            // --------------------------------------------------------------
            // First pass:
            // determine how far the proposed grid lies from the current grid
            // in normalized coordinates.
            // --------------------------------------------------------------

            for (size_t j = 1;
                j + 1 < proposedGrid.size();
                j++) {

                const double oldXi =
                    oldGridEdge[j]
                    / oldMaxEnergy;

                const double proposedXi =
                    proposedGrid[j]
                    / newMaxEnergy;

                if (!std::isfinite(oldXi) ||
                    !std::isfinite(proposedXi)) {

                    throw CanteraError(
                        "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                        "Non-finite normalized grid coordinate at edge {}: "
                        "old xi={}, proposed xi={}.",
                        j,
                        oldXi,
                        proposedXi);
                }

                maxProposedNormalizedGridShift =
                    std::max(
                        maxProposedNormalizedGridShift,
                        std::abs(
                            proposedXi
                            - oldXi));
            }

            // --------------------------------------------------------------
            // Adaptive relaxation factor.
            //
            // alpha = 0.5 at zero displacement,
            // alpha = 1.0 for shifts >= 0.08.
            // --------------------------------------------------------------

            const double normalizedShiftRatio =
                std::min(
                    1.0,
                    maxProposedNormalizedGridShift
                    / AUTO_GRID_RELAXATION_FULL_SHIFT);

            const double alpha =
                AUTO_GRID_RELAXATION_MIN_FACTOR
                + (1.0
                - AUTO_GRID_RELAXATION_MIN_FACTOR)
                * normalizedShiftRatio;

            if (!std::isfinite(alpha) ||
                alpha <= 0.0 ||
                alpha > 1.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                    "Invalid adaptive Automatic-grid relaxation factor: {}.",
                    alpha);
            }

            appliedGridRelaxation =
                alpha;

            // Exact domain endpoints.
            newGrid.front() =
                0.0;

            newGrid.back() =
                newMaxEnergy;

            // --------------------------------------------------------------
            // Second pass:
            // apply the relaxed normalized-grid displacement.
            // --------------------------------------------------------------

            for (size_t j = 1;
                j + 1 < proposedGrid.size();
                j++) {

                const double oldXi =
                    oldGridEdge[j]
                    / oldMaxEnergy;

                const double proposedXi =
                    proposedGrid[j]
                    / newMaxEnergy;

                const double relaxedXi =
                    (1.0 - alpha)
                    * oldXi
                    + alpha
                    * proposedXi;

                if (!std::isfinite(relaxedXi)) {

                    throw CanteraError(
                        "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                        "Non-finite relaxed normalized grid coordinate "
                        "at edge {}.",
                        j);
                }

                newGrid[j] =
                    newMaxEnergy
                    * relaxedXi;

                maxAppliedNormalizedGridShift =
                    std::max(
                        maxAppliedNormalizedGridShift,
                        std::abs(
                            relaxedXi
                            - oldXi));
            }

            // --------------------------------------------------------------
            // Strict monotonicity / finite-value safety.
            // --------------------------------------------------------------

            for (size_t j = 1;
                j < newGrid.size();
                j++) {

                if (!std::isfinite(newGrid[j]) ||
                    !(newGrid[j] > newGrid[j - 1])) {

                    throw CanteraError(
                        "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                        "Relaxed Automatic grid is invalid at edge {}: "
                        "{} <= {}.",
                        j,
                        newGrid[j],
                        newGrid[j - 1]);
                }
            }

        } else {

            // --------------------------------------------------------------
            // First Automatic remesh, or incompatible grid sizes.
            //
            // Do not relax. The bootstrap/grid-size transition must remain free
            // to move directly to the newly proposed grid.
            // --------------------------------------------------------------

            appliedGridRelaxation =
                1.0;

            if (compatibleGridSizes &&
                oldGridEdge.back() > 0.0 &&
                newMaxEnergy > 0.0) {

                const double oldMaxEnergy =
                    oldGridEdge.back();

                for (size_t j = 1;
                    j + 1 < proposedGrid.size();
                    j++) {

                    const double oldXi =
                        oldGridEdge[j]
                        / oldMaxEnergy;

                    const double proposedXi =
                        proposedGrid[j]
                        / newMaxEnergy;

                    if (!std::isfinite(oldXi) ||
                        !std::isfinite(proposedXi)) {

                        throw CanteraError(
                            "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                            "Non-finite normalized grid coordinate "
                            "at edge {}.",
                            j);
                    }

                    maxProposedNormalizedGridShift =
                        std::max(
                            maxProposedNormalizedGridShift,
                            std::abs(
                                proposedXi
                                - oldXi));
                }

                // No relaxation was applied.
                maxAppliedNormalizedGridShift =
                    maxProposedNormalizedGridShift;
            }
        }

        // BEGIN DEBUG
        // ------------------------------------------------------------------
        // Temporary diagnostic: compare old / proposed / actually applied grid.
        // The "proposed" grid was already written by buildAutomaticGrid().
        // ------------------------------------------------------------------

        if (DEBUG_AUTO_GRID_CURRENT_BUILD_ID != 0) {

            const double debugEnTd =
                1.0e21
                * m_phase->reducedElectricField();

            std::ofstream edgeFile(
                AUTO_GRID_EDGES_DEBUG_FILE,
                std::ios::out | std::ios::app);

            if (edgeFile) {

                if (edgeFile.tellp()
                        == std::streampos(0)) {

                    edgeFile
                        << "build_id,"
                        << "pc_iteration,"
                        << "en_Td,"
                        << "final_cells,"
                        << "stage,"
                        << "alpha,"
                        << "Emax_eV,"
                        << "edge_index,"
                        << "energy_eV\n";
                }

                edgeFile
                    << std::setprecision(17);


                // Grid entering this predictor-corrector iteration.
                for (size_t j = 0;
                    j < oldGridEdge.size();
                    j++) {

                    edgeFile
                        << DEBUG_AUTO_GRID_CURRENT_BUILD_ID << ","
                        << iteration + 1 << ","
                        << debugEnTd << ","
                        << m_automaticFinalCells << ","
                        << "old,"
                        << -1.0 << ","
                        << oldGridEdge.back() << ","
                        << j << ","
                        << oldGridEdge[j]
                        << "\n";
                }


                // Grid actually sent to setCustomGrid().
                for (size_t j = 0;
                    j < newGrid.size();
                    j++) {

                    edgeFile
                        << DEBUG_AUTO_GRID_CURRENT_BUILD_ID << ","
                        << iteration + 1 << ","
                        << debugEnTd << ","
                        << m_automaticFinalCells << ","
                        << "applied,"
                        << appliedGridRelaxation << ","
                        << newMaxEnergy << ","
                        << j << ","
                        << newGrid[j]
                        << "\n";
                }
            }
        }
        // END DEBUG


        setCustomGrid(
            newGrid);

        m_kTeMax =
            newMaxEnergy;

        m_has_EEDF = false;

        projectPreviousEEDFOnAutomaticGrid(
            oldGridCenter,
            oldF0,
            predictorFit);

        updateCrossSections();

        solveBoltzmannWithSuperElastic(
            m_f0);

        const AutomaticGridSnapshot current =
            automaticGridSnapshot();

        size_t worstCollision =
            npos;

        double temperatureError =
            0.0;

        lastObjective =
            automaticGridConvergenceObjective(
                previous,
                current,
                worstCollision,
                temperatureError);

        const AutomaticTailFit correctedFit = automaticTailFit();
        const double correctedDecay = automaticPredictorDecayDecades();
        // Enforce the upper decay bound on the corrected EEDF as well.
        // chooseAutomaticMaxEnergy() already reduces the domain for
        // D >= D_min + 1 using a crossing measured on the preceding grid.
        const bool decayAccepted =
            correctedDecay >= m_automaticMinimumPredictorDecayDecades
            && correctedDecay <= maximumDecay;

        // Optional, explicitly NON-CERTIFYING diagnostic. Disabled by
        // default, and never involved in Emax or acceptance decisions.
        lastTailObjective = NAN;
        if (AUTO_TAIL_MODEL_DIAGNOSTIC_ENABLED
            && correctedFit.stable && decayAccepted) {
            lastTailObjective = automaticTailObjective(
                m_gridEdge.back(), current,
                correctedFit.conservativeSlope);
        }
        if (m_automaticVerbose) {
            writelog("[AUTO][PC {}] corrected decay={} (minimum={}) "
                "spread={} % stable={} "
                "tail_model_debug={} tail_model_estimate={}\n",
                iteration + 1, correctedDecay,
                m_automaticMinimumPredictorDecayDecades,
                correctedFit.spreadPercent, correctedFit.stable,
                AUTO_TAIL_MODEL_DIAGNOSTIC_ENABLED, lastTailObjective);
        }

        string worstObservable =
            "electron temperature";

        if (worstCollision != npos) {

            const auto rate =
                m_phase->collisionRate(
                    worstCollision);

            worstObservable =
                rate->collisionName();

            if (worstObservable.empty()) {
                worstObservable =
                    m_phase
                    ->collision(
                        worstCollision)
                    ->equation();
            }
        }

        if (m_automaticVerbose) {
            writelog(
                "Automatic EEDF grid iteration {}: "
                "N={}, Emax={} eV, J_obs={}, "
                "Te_error={}, worst='{}', "
                "decay={} decades, decay_accepted={}, "
                "grid_alpha={}, grid_shift_proposed={}, "
                "grid_shift_applied={}\n",
                iteration + 1,
                m_points,
                m_gridEdge.back(),
                lastObjective,
                temperatureError,
                worstObservable,
                correctedDecay,
                decayAccepted,
                appliedGridRelaxation,
                maxProposedNormalizedGridShift,
                maxAppliedNormalizedGridShift);
        }

        // BEGIN DEBUG

        if (DEBUG_AUTO_GRID_CURRENT_BUILD_ID != 0) {

            const double debugEnTd =
                1.0e21
                * m_phase->reducedElectricField();

            const bool debugAccepted =
                lastObjective <= 1.0 && decayAccepted;

            std::ofstream iterationFile(
                AUTO_GRID_ITERATION_DEBUG_FILE,
                std::ios::out | std::ios::app);

            if (iterationFile) {

                if (iterationFile.tellp()
                        == std::streampos(0)) {

                    iterationFile
                        << "build_id,"
                        << "pc_iteration,"
                        << "en_Td,"
                        << "n_cells,"
                        << "Emax_eV,"
                        << "alpha,"
                        << "grid_shift_proposed,"
                        << "grid_shift_applied,"
                        << "objective,"
                        << "Te_error,"
                        << "predicted_tail_objective,"
                        << "tail_objective,"
                        << "worst_observable,"
                        << "accepted\n";
                }

                iterationFile
                    << std::setprecision(17)
                    << DEBUG_AUTO_GRID_CURRENT_BUILD_ID << ","
                    << iteration + 1 << ","
                    << debugEnTd << ","
                    << m_points << ","
                    << m_gridEdge.back() << ","
                    << appliedGridRelaxation << ","
                    << maxProposedNormalizedGridShift << ","
                    << maxAppliedNormalizedGridShift << ","
                    << lastObjective << ","
                    << temperatureError << ","
                    << predictedTailObjective << ","
                    << lastTailObjective << ","
                    << debugAutomaticGridCsvSafe(
                        worstObservable) << ","
                    << (debugAccepted ? 1 : 0)
                    << "\n";
            }
        }

        // END DEBUG


        if (lastObjective <= 1.0 && decayAccepted) {
            if (m_automaticVerbose) {
                writelog("[AUTO][PC] ACCEPT: J_obs={} <= 1 and "
                    "resolved decay={} in [{}, {}] decades. "
                    "Extrapolated J_tail is NOT an acceptance criterion.\n",
                    lastObjective, correctedDecay,
                    m_automaticMinimumPredictorDecayDecades, maximumDecay);
            }
            accepted = true;
            break;
        }

        previous = current;
    }

    if (!accepted) {

        if (m_automaticRequireConvergence) {

            throw CanteraError(
                "EEDFTwoTermApproximation::solveWithAutomaticGrid",
                "Automatic grid did not satisfy J_obs <= 1 and "
                "resolved decay in [{}, {}] decades after {} "
                "predictor-corrector iterations. "
                "Last J_obs={}, last decay={} decades.",
                m_automaticMinimumPredictorDecayDecades,
                maximumDecay,
                m_automaticPredictorCorrectorIterations,
                lastObjective,
                automaticPredictorDecayDecades());
        }

        warn_user(
            "EEDFTwoTermApproximation::solveWithAutomaticGrid",
            "Automatic grid did not satisfy J_obs <= 1 and "
            "resolved decay in [{}, {}] decades after {} "
            "predictor-corrector iterations. "
            "Using the last grid: J_obs={}, decay={} decades.",
            m_automaticMinimumPredictorDecayDecades,
            maximumDecay,
            m_automaticPredictorCorrectorIterations,
            lastObjective,
            automaticPredictorDecayDecades());
    }
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

    const bool forcedMaxwellian = m_phase->reducedElectricField() <= EN_min;
    m_regimeChangedThisCall = m_has_EEDF &&
        (forcedMaxwellian != m_lastEEDFWasForcedMaxwellian);

    if (!m_forceMaxwellianRestart && !checkParamsVariation()) {
        if (DEBUG_SE) {
            writelog("EEDF-SE-DEBUG CALC_SKIP: EEDF cache accepted, no recomputation.\n");
        }
        return 0;
    }

    m_f0_computed_at_last_call = true;

    // On forced Maxwell: replace ANY inherited Boltzmann grid first.
    // Grid caches must then be rebuilt BEFORE cross sections and mobility.
    if (forcedMaxwellian) {
        const double gasKTeEv = Boltzmann * m_phase->temperature() / ElectronCharge;
        buildForcedMaxwellianGrid(gasKTeEv);
    } else if (m_regimeChangedThisCall && m_gridType != "Automatic") {
        // Return from the dedicated Maxwellian grid: start the legacy Boltzmann
        // solve from its configured grid, just like an independent fresh solve.
        // This applies with or without legacy Emax adaptation enabled.
        updateGrid(m_initialMaxEnergy);
    }

    checkSpeciesNoCrossSection();
    // Start from a cross-section state without super-elastic terms. If the
    // feature is enabled, solveBoltzmannWithSuperElastic() will first compute a
    // predictor EEDF, then activate super-elastic terms for the corrector solve.
    m_applySuperElasticCollisions = false;
    updateCrossSections();

    const double EN = m_phase->reducedElectricField();

    // At very low reduced electric field, keep the existing forced-Maxwellian
    // behavior.
    if (EN <= EN_min) {

        setMaxwellianEEDF(
            Boltzmann
            * m_phase->temperature()
            / ElectronCharge);

    } else if (m_gridType == "Automatic") {

        // New observable-driven grid path.
        solveWithAutomaticGrid();

    } else {

        // ------------------------------------------------------------------
        // Existing Linear / Quadratic / Geometric path.
        // ------------------------------------------------------------------

        if (!m_has_EEDF ||
            m_forceMaxwellianRestart) {

            if (m_firstguess == "maxwell") {

                setMaxwellianEEDF(
                    m_init_kTe);

                if (m_forceMaxwellianRestart) {
                    m_forceMaxwellianRestart = false;
                }

            } else {

                throw CanteraError(
                    "EEDFTwoTermApproximation::calculateDistributionFunction",
                    "Unknown EEDF first guess '{}'.",
                    m_firstguess);
            }
        }

        solveBoltzmannWithSuperElastic(
            m_f0);

        // ------------------------------------------------------------------
        // Existing legacy Emax adaptation.
        // ------------------------------------------------------------------
        if (m_adaptGrid) {

            const double fFloor =
                1e-300;

            for (size_t n = 0;
                n < m_maxGridAdaptIterations;
                n++) {

                const double fLeft =
                    std::max(
                        std::abs(m_f0(0)),
                        fFloor);

                const double fRight =
                    std::max(
                        std::abs(
                            m_f0(
                                m_points - 1)),
                        fFloor);

                const double decades =
                    std::log10(fLeft)
                    - std::log10(fRight);

                if (!std::isfinite(decades)) {

                    throw CanteraError(
                        "EEDFTwoTermApproximation::calculateDistributionFunction",
                        "Non-finite EEDF decay detected during grid adaptation.");
                }

                if (decades
                    < m_minEedfDecay) {

                    const double newMaxEnergy =
                        m_kTeMax
                        * (
                            1.0
                            + m_gridUpdateFactor
                        );

                    const Eigen::VectorXd oldGridCenter =
                        m_gridCenter;

                    const Eigen::VectorXd oldF0 =
                        m_f0;

                    updateGrid(
                        newMaxEnergy);

                    if (m_maxwellianReset) {

                        if (m_firstguess
                            == "maxwell") {

                            setMaxwellianEEDF(
                                m_init_kTe);

                        } else {

                            throw CanteraError(
                                "EEDFTwoTermApproximation::calculateDistributionFunction",
                                "Unknown EEDF first guess '{}'.",
                                m_firstguess);
                        }

                    } else {

                        projectPreviousEEDFOnCurrentGrid(
                            oldGridCenter,
                            oldF0);
                    }

                    updateCrossSections();

                    solveBoltzmannWithSuperElastic(
                        m_f0);

                } else if (
                    decades
                    > m_maxEedfDecay) {

                    const double newMaxEnergy =
                        m_kTeMax
                        /
                        (
                            1.0
                            + m_gridUpdateFactor
                        );

                    const Eigen::VectorXd oldGridCenter =
                        m_gridCenter;

                    const Eigen::VectorXd oldF0 =
                        m_f0;

                    updateGrid(
                        newMaxEnergy);

                    if (m_maxwellianReset) {

                        if (m_firstguess
                            == "maxwell") {

                            setMaxwellianEEDF(
                                m_init_kTe);

                        } else {

                            throw CanteraError(
                                "EEDFTwoTermApproximation::calculateDistributionFunction",
                                "Unknown EEDF first guess '{}'.",
                                m_firstguess);
                        }

                    } else {

                        projectPreviousEEDFOnCurrentGrid(
                            oldGridCenter,
                            oldF0);
                    }

                    updateCrossSections();

                    solveBoltzmannWithSuperElastic(
                        m_f0);

                } else {

                    break;
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
    m_lastEEDFWasForcedMaxwellian = forcedMaxwellian;
    m_regimeChangedThisCall = false;

    // Update electron mobility.
    m_electronMobility = computeElectronMobility(m_f0);

    // Store the state for which the current EEDF is valid.
    storeCurrentParamsForEEDF();

    debugEedfSummary("final_center_f0", m_gridCenter, m_f0);
    return 0;
}

// ancienne version de converge
// void EEDFTwoTermApproximation::converge(Eigen::VectorXd& f0)
// {
//     double err0 = 0.0;
//     double err1 = 0.0;
//     double delta = m_delta0;

//     if (m_maxn == 0) {
//         throw CanteraError("EEDFTwoTermApproximation::converge",
//                            "m_maxn is zero; no iterations will occur.");
//     }
//     if (m_points == 0) {
//         throw CanteraError("EEDFTwoTermApproximation::converge",
//                            "m_points is zero; the EEDF grid is empty.");
//     }
//     if (isnan(delta) || delta == 0.0) {
//         throw CanteraError("EEDFTwoTermApproximation::converge",
//                            "m_delta0 is NaN or zero; solver cannot update.");
//     }

//     if (DEBUG_SE) {
//         writelog(
//             "EEDF-SE-DEBUG CONVERGE_START: applySE={}, enableSE={}, "
//             "maxn={}, rtol={}, delta0={}, norm_initial={}\n",
//             m_applySuperElasticCollisions,
//             m_enableSuperElasticCollisions,
//             m_maxn,
//             m_rtol,
//             m_delta0,
//             norm(f0, m_gridCenter)
//         );
//     }

//     for (size_t n = 0; n < m_maxn; n++) {
//         if (0.0 < err1 && err1 < err0) {
//             delta *= log(m_factorM) / (log(err0) - log(err1));
//         }

//         Eigen::VectorXd f0_old = f0;
//         f0 = iterate(f0_old, delta);
//         checkFinite("EEDFTwoTermApproximation::converge: f0", asSpan(f0));

//         // // The physical EEDF is non-negative. Sparse solves can nevertheless
//         // // produce tiny negative components at roundoff level. Clip only when
//         // // the negative excursion is negligible relative to the EEDF scale;
//         // // otherwise keep a hard failure with a useful diagnostic.
//         // Eigen::Index minIndex = 0;
//         // const double fMin = f0.minCoeff(&minIndex);
//         // const double fScale = f0.cwiseAbs().maxCoeff();

//         // if (!std::isfinite(fScale) || fScale <= 0.0) {
//         //     throw CanteraError(
//         //         "EEDFTwoTermApproximation::converge",
//         //         "EEDF has invalid magnitude after an iteration: max|f0|={}.",
//         //         fScale);
//         // }

//         // if (fMin < 0.0) {
//         //     const double relativeNegativeAmplitude =
//         //         -fMin / fScale;

//         //     if (relativeNegativeAmplitude > EEDF_NEGATIVE_REL_TOL) {
//         //         throw CanteraError(
//         //             "EEDFTwoTermApproximation::converge",
//         //             "Significantly negative EEDF after an iteration: "
//         //             "f_min={} at index {} (energy={} eV), max|f0|={}, "
//         //             "|f_min|/max|f0|={}, allowed relative amplitude={}.",
//         //             fMin,
//         //             static_cast<size_t>(minIndex),
//         //             m_gridCenter[static_cast<size_t>(minIndex)],
//         //             fScale,
//         //             relativeNegativeAmplitude,
//         //             EEDF_NEGATIVE_REL_TOL);
//         //     }

//         //     for (Eigen::Index i = 0; i < f0.size(); i++) {
//         //         if (f0[i] < 0.0) {
//         //             f0[i] = 0.0;
//         //         }
//         //     }

//         //     const double clippedNorm =
//         //         norm(f0, m_gridCenter);

//         //     if (!std::isfinite(clippedNorm) || clippedNorm <= 0.0) {
//         //         throw CanteraError(
//         //             "EEDFTwoTermApproximation::converge",
//         //             "Invalid EEDF norm after clipping roundoff-level "
//         //             "negative values: {}.",
//         //             clippedNorm);
//         //     }

//         //     f0 /= clippedNorm;

//         //     if (DEBUG_SE) {
//         //         writelog(
//         //             "EEDF-SE-DEBUG CLIP_NEGATIVE: iter={}, f_min={}, "
//         //             "relative_negative={}\n",
//         //             n,
//         //             fMin,
//         //             relativeNegativeAmplitude);
//         //     }
//         // }

//         err0 = err1;
//         Eigen::VectorXd Df0 = (f0_old - f0).cwiseAbs();
//         err1 = norm(Df0, m_gridCenter);
//         if (DEBUG_SE && (n < 5 || err1 < 10.0 * m_rtol || n + 1 == m_maxn)) {
//             writelog(
//                 "EEDF-SE-DEBUG CONVERGE_ITER: applySE={}, iter={}, delta={}, "
//                 "err={}, norm_f0={}, f0_first={}, f0_last={}\n",
//                 m_applySuperElasticCollisions,
//                 n,
//                 delta,
//                 err1,
//                 norm(f0, m_gridCenter),
//                 f0.size() > 0 ? f0[0] : -1.0,
//                 f0.size() > 0 ? f0[f0.size() - 1] : -1.0
//             );
//         }
//         const size_t minIterations = m_applySuperElasticCollisions ? m_minSuperElasticIterations : 0;
//         if (err1 < m_rtol && n >= minIterations) {
//             if (DEBUG_SE) {
//                 writelog(
//                     "EEDF-SE-DEBUG CONVERGE_DONE: applySE={}, iter={}, err={}, norm_f0={}\n",
//                     m_applySuperElasticCollisions,
//                     n,
//                     err1,
//                     norm(f0, m_gridCenter)
//                 );
//             }
//             break;
//         } else if (n == m_maxn - 1) {
//             throw CanteraError("EEDFTwoTermApproximation::converge", "Convergence failed");
//         }
//         if (DEBUG_SE && m_applySuperElasticCollisions) {
//             debugEedfSummary("corrector_iter_withSE", m_gridCenter, f0);
//         }
//     }
// }

// nouvelle version de converge adaptée à la grille automatique et aux conditions de positivité
void EEDFTwoTermApproximation::converge(Eigen::VectorXd& f0)
{
    double err0 = 0.0;
    double err1 = 0.0;
    double delta = m_delta0;

    // If an iteration required backtracking, do not immediately enlarge
    // delta again at the beginning of the next nonlinear iteration.
    bool previousStepBacktracked = false;

    if (m_maxn == 0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::converge",
            "m_maxn is zero; no iterations will occur.");
    }

    if (m_points == 0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::converge",
            "m_points is zero; the EEDF grid is empty.");
    }

    if (!std::isfinite(delta) || delta <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::converge",
            "Invalid initial pseudo-time step m_delta0={}.",
            delta);
    }

    // ------------------------------------------------------------------
    // Ensure that the initial iterate is inside the admissible
    // non-negative cone, apart from roundoff-level negative values.
    // ------------------------------------------------------------------
    {
        Eigen::Index minIndex = 0;

        const double fMin =
            f0.minCoeff(&minIndex);

        const double fScale =
            f0.cwiseAbs().maxCoeff();

        if (!std::isfinite(fMin) ||
            !std::isfinite(fScale) ||
            fScale <= 0.0) {

            throw CanteraError(
                "EEDFTwoTermApproximation::converge",
                "Invalid initial EEDF: f_min={}, max|f0|={}.",
                fMin,
                fScale);
        }

        if (fMin < 0.0) {

            const double relativeNegativeAmplitude =
                -fMin / fScale;

            if (relativeNegativeAmplitude
                > EEDF_NEGATIVE_REL_TOL) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Initial EEDF is significantly negative: "
                    "f_min={} at index {} (energy={} eV), "
                    "max|f0|={}, |f_min|/max|f0|={}, "
                    "allowed relative amplitude={}.",
                    fMin,
                    static_cast<size_t>(minIndex),
                    m_gridCenter[
                        static_cast<size_t>(minIndex)],
                    fScale,
                    relativeNegativeAmplitude,
                    EEDF_NEGATIVE_REL_TOL);
            }

            // Roundoff only: clip and renormalize.
            for (Eigen::Index i = 0;
                 i < f0.size();
                 i++) {

                if (f0[i] < 0.0) {
                    f0[i] = 0.0;
                }
            }

            const double fNorm =
                norm(f0, m_gridCenter);

            if (!std::isfinite(fNorm) ||
                fNorm <= 0.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Invalid initial EEDF norm after "
                    "roundoff clipping: {}.",
                    fNorm);
            }

            f0 /= fNorm;
        }
    }

    if (DEBUG_SE) {
        writelog(
            "EEDF-SE-DEBUG CONVERGE_START: "
            "applySE={}, enableSE={}, "
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

        // --------------------------------------------------------------
        // Existing adaptive-delta logic.
        //
        // If the previous accepted step had to be backtracked, keep the
        // reduced delta for one additional nonlinear iteration instead of
        // immediately jumping back to a much larger value.
        // --------------------------------------------------------------
        if (!previousStepBacktracked &&
            0.0 < err1 &&
            err1 < err0) {

            const double denominator =
                std::log(err0) -
                std::log(err1);

            if (std::isfinite(denominator) &&
                denominator > 0.0) {

                const double multiplier =
                    std::log(m_factorM)
                    / denominator;

                if (std::isfinite(multiplier) &&
                    multiplier > 0.0) {

                    delta *= multiplier;
                }
            }
        }

        if (!std::isfinite(delta) ||
            delta <= 0.0) {

            throw CanteraError(
                "EEDFTwoTermApproximation::converge",
                "Invalid pseudo-time step during convergence: "
                "delta={}.",
                delta);
        }

        Eigen::VectorXd f0_old = f0;

        // --------------------------------------------------------------
        // Positivity-preserving backtracking.
        // --------------------------------------------------------------
        Eigen::VectorXd candidate;

        double trialDelta =
            delta;

        size_t backtrackCount =
            0;

        double candidateMin =
            0.0;

        double candidateScale =
            0.0;

        double relativeNegativeAmplitude =
            0.0;

        Eigen::Index candidateMinIndex =
            0;

        while (true) {

            candidate =
                iterate(
                    f0_old,
                    trialDelta);

            checkFinite(
                "EEDFTwoTermApproximation::converge: candidate",
                asSpan(candidate));

            candidateMin =
                candidate.minCoeff(
                    &candidateMinIndex);

            candidateScale =
                candidate.cwiseAbs().maxCoeff();

            if (!std::isfinite(candidateMin) ||
                !std::isfinite(candidateScale) ||
                candidateScale <= 0.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Invalid trial EEDF during convergence: "
                    "f_min={}, max|f|={}, delta={}.",
                    candidateMin,
                    candidateScale,
                    trialDelta);
            }

            relativeNegativeAmplitude =
                candidateMin < 0.0
                ? -candidateMin / candidateScale
                : 0.0;

            // Candidate is physically admissible, apart from roundoff.
            if (relativeNegativeAmplitude
                <= EEDF_NEGATIVE_REL_TOL) {

                break;
            }

            // Candidate left the physical non-negative cone.
            // Retry the SAME nonlinear iteration from f0_old.
            if (backtrackCount >=
                EEDF_DELTA_BACKTRACK_MAX_RETRIES) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Unable to obtain a non-negative EEDF iterate "
                    "after {} pseudo-time backtracks. "
                    "Last f_min={} at index {} "
                    "(energy={} eV), max|f|={}, "
                    "|f_min|/max|f|={}, last delta={}.",
                    EEDF_DELTA_BACKTRACK_MAX_RETRIES,
                    candidateMin,
                    static_cast<size_t>(
                        candidateMinIndex),
                    m_gridCenter[
                        static_cast<size_t>(
                            candidateMinIndex)],
                    candidateScale,
                    relativeNegativeAmplitude,
                    trialDelta);
            }

            trialDelta *=
                EEDF_DELTA_BACKTRACK_FACTOR;

            backtrackCount++;

            if (!std::isfinite(trialDelta) ||
                trialDelta <= 0.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Pseudo-time step became invalid during "
                    "backtracking: delta={}.",
                    trialDelta);
            }
        }

        // --------------------------------------------------------------
        // Clip only roundoff-level negative values in the ACCEPTED trial.
        //
        // This also guarantees that the next nonlinear iteration starts
        // from a non-negative EEDF.
        // --------------------------------------------------------------
        if (candidateMin < 0.0) {

            for (Eigen::Index i = 0;
                 i < candidate.size();
                 i++) {

                if (candidate[i] < 0.0) {
                    candidate[i] = 0.0;
                }
            }

            const double candidateNorm =
                norm(
                    candidate,
                    m_gridCenter);

            if (!std::isfinite(candidateNorm) ||
                candidateNorm <= 0.0) {

                throw CanteraError(
                    "EEDFTwoTermApproximation::converge",
                    "Invalid EEDF norm after clipping "
                    "roundoff-level negative values: {}.",
                    candidateNorm);
            }

            candidate /=
                candidateNorm;
        }

        // Accept the trial delta and the corresponding EEDF.
        delta =
            trialDelta;

        previousStepBacktracked =
            backtrackCount > 0;

        f0 =
            candidate;

        err0 =
            err1;

        Eigen::VectorXd Df0 =
            (f0_old - f0).cwiseAbs();

        err1 =
            norm(
                Df0,
                m_gridCenter);

        if (DEBUG_SE &&
            backtrackCount > 0) {

            writelog(
                "EEDF-SE-DEBUG DELTA_BACKTRACK: "
                "applySE={}, iter={}, "
                "backtracks={}, accepted_delta={}, "
                "relative_negative={}\n",
                m_applySuperElasticCollisions,
                n,
                backtrackCount,
                delta,
                relativeNegativeAmplitude);
        }

        if (DEBUG_SE &&
            (n < 5 ||
             err1 < 10.0 * m_rtol ||
             n + 1 == m_maxn)) {

            writelog(
                "EEDF-SE-DEBUG CONVERGE_ITER: "
                "applySE={}, iter={}, "
                "delta={}, err={}, norm_f0={}, "
                "f0_first={}, f0_last={}, "
                "backtracks={}\n",
                m_applySuperElasticCollisions,
                n,
                delta,
                err1,
                norm(f0, m_gridCenter),
                f0.size() > 0
                    ? f0[0]
                    : -1.0,
                f0.size() > 0
                    ? f0[f0.size() - 1]
                    : -1.0,
                backtrackCount
            );
        }

        const size_t minIterations =
            m_applySuperElasticCollisions
            ? m_minSuperElasticIterations
            : 0;

        if (err1 < m_rtol &&
            n >= minIterations) {

            // Every accepted nonlinear iterate is now already
            // non-negative to roundoff, so no separate large-negative
            // final state should be possible.

            if (DEBUG_SE) {
                writelog(
                    "EEDF-SE-DEBUG CONVERGE_DONE: "
                    "applySE={}, iter={}, err={}, "
                    "delta={}, norm_f0={}\n",
                    m_applySuperElasticCollisions,
                    n,
                    err1,
                    delta,
                    norm(f0, m_gridCenter)
                );
            }

            break;

        } else if (n == m_maxn - 1) {

            throw CanteraError(
                "EEDFTwoTermApproximation::converge",
                "Convergence failed");
        }

        if (DEBUG_SE &&
            m_applySuperElasticCollisions) {

            debugEedfSummary(
                "corrector_iter_withSE",
                m_gridCenter,
                f0);
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
    g[0] = -log(f1 / f0_) / (m_gridCenter[1] - m_gridCenter[0]);

    // Handle last point (i = N)
    size_t N = m_points - 1;
    double fN = std::max(f0(N), f_min);
    double fNm1 = std::max(f0(N - 1), f_min);
    g[N] = -log(fN / fNm1) / (m_gridCenter[N] - m_gridCenter[N - 1]);

    // Handle interior points
    for (size_t i = 1; i < N; ++i) {
        double f_up   = std::max(f0(i + 1), f_min);
        double f_down = std::max(f0(i - 1), f_min);
        g[i] = -log(f_up / f_down) / (m_gridCenter[i + 1] - m_gridCenter[i - 1]);
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

// double EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF(
//     const Eigen::VectorXd& f0, size_t k)
// {
//     auto rate = m_phase->collisionRate(k);
//     auto x = rate->energyLevels();
//     auto y = rate->crossSections();

//     vector<double> integrand(m_points, 0.0);

//     for (size_t i = 0; i < m_points; i++) {
//         const double eps = m_gridCenter[i];
//         const double sigma =
//             linearInterpCrossSectionZeroOutside(eps, x, y);
//         integrand[i] = eps * sigma * f0(i);
//     }

//     auto f = ConstMappedVector(integrand.data(), integrand.size());
//     auto epsGrid = ConstMappedVector(m_gridCenter.data(), m_gridCenter.size());

//     return m_gamma * simpson(f, epsGrid);
// }

double EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF(
    const Eigen::VectorXd& f0, size_t k)
{
    if (k >= m_phase->nCollisions()) {
        throw CanteraError(
            "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
            "Collision index {} is out of range. Number of collisions: {}.",
            k,
            m_phase->nCollisions());
    }

    if (static_cast<size_t>(f0.size()) != m_points) {
        throw CanteraError(
            "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
            "EEDF size mismatch: got {}, expected {}.",
            f0.size(),
            m_points);
    }

    if (k >= m_eps.size() ||
        k >= m_sigma.size() ||
        k >= m_j.size()) {
        throw CanteraError(
            "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
            "Collision integration cache is not initialized for collision {}.",
            k);
    }

    if (m_eps[k].size() != m_sigma[k].size() ||
        m_eps[k].size() != m_j[k].size()) {
        throw CanteraError(
            "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
            "Inconsistent collision integration cache sizes for collision {}: "
            "eps={}, sigma={}, sourceCells={}.",
            k,
            m_eps[k].size(),
            m_sigma[k].size(),
            m_j[k].size());
    }

    // Local logarithmic slopes used by the Boltzmann collision operator:
    //
    //     f(eps) ~= f_j * exp[g_j * (eps_j - eps)]
    //
    // inside source cell j.
    const vector<double> g = vector_g(f0);

    double K = 0.0;
    double absoluteContributionSum = 0.0;

    // m_eps[k] already contains a refined partition including:
    //   - EEDF-grid-related breakpoints,
    //   - shifted collision breakpoints,
    //   - native cross-section tabulation energies.
    //
    // Therefore narrow structures in sigma(eps) are preserved even when
    // they fall between two EEDF cell centers.
    for (size_t n = 0; n < m_eps[k].size(); n++) {

        if (m_eps[k][n].size() != 2 ||
            m_sigma[k][n].size() != 2) {
            throw CanteraError(
                "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
                "Malformed collision integration segment {} for collision {}.",
                n,
                k);
        }

        const double eps_a = m_eps[k][n][0];
        const double eps_b = m_eps[k][n][1];

        const double sigma_a = m_sigma[k][n][0];
        const double sigma_b = m_sigma[k][n][1];

        const size_t j = m_j[k][n];

        if (j >= m_points) {
            throw CanteraError(
                "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
                "Invalid source-cell index {} for collision {} segment {}.",
                j,
                k,
                n);
        }

        // integralPQ evaluates
        //
        //   integral_a^b eps * sigma(eps)
        //       * exp[g_j * (eps_j - eps)] d eps
        //
        // with sigma linearly interpolated between sigma_a and sigma_b.
        //
        // Multiplication by f0(j) reconstructs the local EEDF.
        const double r = integralPQ(
            eps_a,
            eps_b,
            sigma_a,
            sigma_b,
            g[j],
            m_gridCenter[j]);

        const double contribution =
            m_gamma * r * f0(j);

        if (!std::isfinite(contribution)) {
            throw CanteraError(
                "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
                "Non-finite rate contribution for collision {} segment {}.",
                k,
                n);
        }

        K += contribution;
        absoluteContributionSum += std::abs(contribution);
    }

    if (!std::isfinite(K)) {
        throw CanteraError(
            "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
            "Non-finite forward rate coefficient for collision '{}': {}.",
            m_phase->collisionRate(k)->collisionName(),
            K);
    }

    // A physical electron-collision rate coefficient is non-negative.
    // Analytical segment integration can still leave a tiny negative residual
    // after cancellation. Accept only roundoff-scale negatives; keep a hard
    // error for any physically significant negative result.
    const double roundoffTolerance =
        RATE_ROUNDOFF_FACTOR
        * std::numeric_limits<double>::epsilon()
        * std::max(absoluteContributionSum, 1.0e-300);

    const double negativeTolerance =
        std::max(
            RATE_NEGATIVE_ABS_TOL,
            roundoffTolerance);

    if (K < 0.0) {
        if (-K <= negativeTolerance) {
            K = 0.0;
        } else {
            throw CanteraError(
                "EEDFTwoTermApproximation::forwardRateCoefficientFromEEDF",
                "Negative forward rate coefficient for collision '{}': {}. "
                "Absolute contribution sum={}, allowed negative tolerance={}.",
                m_phase->collisionRate(k)->collisionName(),
                K,
                absoluteContributionSum,
                negativeTolerance);
        }
    }

    return K;
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

    // ancienne version du code au 16/08/2026
    //     for (size_t i = 0; i < m_points; i++) {
    //         m_totalCrossSectionCenter[i] += X_target *
    //                                         linearInterp(m_gridCenter[i], x, y);
    //     }

    //     for (size_t i = 0; i < m_points + 1; i++) {
    //         m_totalCrossSectionEdge[i] += X_target *
    //                                       linearInterp(m_gridEdge[i], x, y);
    //     }
    // }

    // nouvelle version avec l'interpolation plus poussée
        for (size_t i = 0; i < m_points; i++) {
            m_totalCrossSectionCenter[i] += X_target *
                                            linearInterpCrossSectionZeroOutside(m_gridCenter[i], x, y);
        }

        for (size_t i = 0; i < m_points + 1; i++) {
            m_totalCrossSectionEdge[i] += X_target *
                                          linearInterpCrossSectionZeroOutside(m_gridEdge[i], x, y);
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

// // new implementation of the function proposed by nicolas
// // implementation 1 faite par moi, avec une potentielle reconstruction fausse
// // A cause d'une mauvaise interpolation
// void EEDFTwoTermApproximation::calculateTotalElasticCrossSection()
// {
//     m_sigmaElastic.clear();
//     m_sigmaElastic.resize(m_points, 0.0);
//     for (size_t k : m_phase->kElastic()) {

//         // writelog("Enter with an elastic cs\n");
//         auto x = m_phase->collisionRate(k)->energyLevels();
//         auto y = m_phase->collisionRate(k)->crossSections();
//         vector<double> y_elastic(y.data(), y.data() + y.size());

//         // Note:
//         // moleFraction(m_kTargets[k]) <=> m_X_targets[m_klocTargets[k]]
//         double mass_ratio = ElectronMass / (m_phase->molecularWeight(m_kTargets[k]) / Avogadro);

//         if (m_phase->collisionRate(k)->kind()=="effective") {

//             // writelog("Enter with an elastic cs that is actually an effective\n");

//             for (size_t ki : m_phase->kInelastic())
//             {
//                 if(m_phase->targetIndex(ki) == m_phase->targetIndex(k)){
//                     // writelog("loop over inelastic processes: process {}\n", ki);
//                     auto xi = m_phase->collisionRate(ki)->energyLevels();
//                     auto yi = m_phase->collisionRate(ki)->crossSections();
//                     for (size_t i = 0; i < x.size(); i++)
//                     {
//                         y_elastic[i] -= linearInterpCrossSectionZeroOutside(x[i], xi, yi);
//                     }
//                 }
//             }
//             // check that the reconstructed elastic cross section is non-negative.
//             for (size_t i = 0; i < y_elastic.size(); i++) {
//                 if (y_elastic[i] < 0.0) {
//                     if (y_elastic[i] > -1e-30) {
//                         // y_elastic[i] = 0.0;
//                     } else {
//                         const std::string& effectiveKind = m_phase->collisionRate(k)->kind();
//                         std::string effectiveTarget = m_phase->speciesName(m_phase->targetIndex(k));
//                         if (m_warning_negative_CS){
//                             writelog("Warning: reconstructed elastic cross section is negative "
//                                     "for collision {} kind {} target {} at energy {}: {}\n",
//                                     k, effectiveKind, effectiveTarget, x[i], y_elastic[i]);
//                         }
//                         // y_elastic[i] = 0.0;
//                     }
//                 }
//             }
//         }

//         for (size_t i = 0; i < m_points; i++) {
//             m_sigmaElastic[i] += 2.0 * mass_ratio * m_X_targets[m_klocTargets[k]] *
//                                  linearInterp(m_gridEdge[i], x, y_elastic);
//         }
//     }
//     m_warning_negative_CS = false;
// }

// Nouvelle interpolation en théorie plus juste
void EEDFTwoTermApproximation::calculateTotalElasticCrossSection()
{
    m_sigmaElastic.clear();
    m_sigmaElastic.resize(m_points, 0.0);

    for (size_t k : m_phase->kElastic()) {

        auto rate = m_phase->collisionRate(k);

        auto x = rate->energyLevels();
        auto y = rate->crossSections();

        const size_t target = m_phase->targetIndex(k);

        // Electron / target mass ratio
        double mass_ratio =
            ElectronMass
            / (m_phase->molecularWeight(m_kTargets[k]) / Avogadro);

        const double prefactor =
            2.0 * mass_ratio * m_X_targets[m_klocTargets[k]];

        // Reconstruct directly on the EEDF grid.
        for (size_t i = 0; i < m_points; i++) {

            const double eps = m_gridEdge[i];

            // Use the original elastic/effective CS directly at the
            // energy where the solver needs it.
            //
            // Keep linearInterp() or linearInterpCrossSectionZeroOutside() here so that the interpolation of the
            // effective CS is identical to that used in
            // calculateTotalCrossSection().
            double sigma_elastic = linearInterpCrossSectionZeroOutside(eps, x, y);

            if (rate->kind() == "effective") {

                // effective = elastic momentum-transfer
                //           + sum of forward inelastic cross sections
                //
                // Therefore:
                //
                // elastic = effective - sum(inelastic)
                //
                // IMPORTANT:
                // Every inelastic cross section is interpolated directly
                // at eps. We do NOT first project the inelastic CS onto
                // the effective-CS grid.
                for (size_t ki : m_phase->kInelastic()) {

                    if (m_phase->targetIndex(ki) != target) {
                        continue;
                    }

                    auto rate_i = m_phase->collisionRate(ki);

                    auto xi = rate_i->energyLevels();
                    auto yi = rate_i->crossSections();

                    sigma_elastic -=
                        linearInterpCrossSectionZeroOutside(
                            eps, xi, yi);
                }

                // Diagnostic only.
                // Do NOT clip negative reconstructed cross sections.
                if (sigma_elastic < -1e-30 && m_warning_negative_CS) {

                    const std::string& effectiveKind = rate->kind();

                    std::string effectiveTarget =
                        m_phase->speciesName(target);

                    writelog(
                        "Warning: reconstructed elastic cross section is negative "
                        "for collision {} kind {} target {} at energy {}: {}\n",
                        k,
                        effectiveKind,
                        effectiveTarget,
                        eps,
                        sigma_elastic);
                }
            }

            m_sigmaElastic[i] += prefactor * sigma_elastic;
        }
    }

    m_warning_negative_CS = false;
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

// void EEDFTwoTermApproximation::setGridType(const string& gridType)
// {
//     if (gridType != "Linear" &&
//         gridType != "Quadratic" &&
//         gridType != "Geometric") {
//         throw CanteraError("EEDFTwoTermApproximation::setGridType",
//             "Unknown energy grid type '{}'. Expected Linear, Quadratic or Geometric.",
//             gridType);
//     }

//     m_gridType = gridType;
// }

void EEDFTwoTermApproximation::setGridType(const string& gridType)
{
    if (gridType != "Linear" &&
        gridType != "Quadratic" &&
        gridType != "Geometric" &&
        gridType != "Automatic") {

        throw CanteraError(
            "EEDFTwoTermApproximation::setGridType",
            "Unknown energy grid type '{}'. Expected Linear, Quadratic, "
            "Geometric or Automatic.",
            gridType);
    }

    m_gridType = gridType;

    // The legacy Emax adaptation and the observable-driven Automatic grid
    // are two independent algorithms and must never control the same grid.
    if (m_gridType == "Automatic") {
        m_adaptGrid = false;
    }

    m_f0_ok = false;
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
    m_initialMaxEnergy = initialMaxEnergy;
    m_initialGridCells = nGridCells;
}

void EEDFTwoTermApproximation::setAutomaticMonitorSmoothing(bool enabled)
{
    m_automaticMonitorSmoothing = enabled;
}

void EEDFTwoTermApproximation::setAutomaticGridParameters(
    size_t finalCells,
    size_t bootstrapCells,
    double bootstrapMaxEnergy,
    double bootstrapStretchEnergy,
    size_t bootstrapMaxAttempts,
    double bootstrapExpansionFactor,
    size_t predictorCorrectorIterations,
    bool requireConvergence,
    double electronTemperatureRtol,
    double electronTemperatureAtolEv,
    double collisionRateRtol,
    double collisionRateFloorM3s,
    double tailBudgetFraction,
    double tailExtrapolationDecades,
    double minimumPredictorDecayDecades,
    double maxEmaxGrowthFactor,
    double minEmaxShrinkFactor,
    double baselineMetricFraction,
    double thresholdGuardFraction,
    double maximumPredictorDecayDecades,
    double minimumObservableTargetMoleFraction)
{
    if (finalCells < 3) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "final-number-of-cells must be at least 3.");
    }

    if (bootstrapCells < 3) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "bootstrap-number-of-cells must be at least 3.");
    }

    if (!std::isfinite(bootstrapMaxEnergy) ||
        bootstrapMaxEnergy <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "bootstrap-max-energy-level must be finite and positive.");
    }

    if (!std::isfinite(bootstrapStretchEnergy) ||
        bootstrapStretchEnergy <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "bootstrap-stretch-energy must be finite and positive.");
    }

    if (bootstrapMaxAttempts == 0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "bootstrap-max-attempts must be greater than zero.");
    }

    if (!std::isfinite(bootstrapExpansionFactor) ||
        bootstrapExpansionFactor <= 1.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "bootstrap-expansion-factor must be greater than one.");
    }

    if (predictorCorrectorIterations == 0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "predictor-corrector-iterations must be greater than zero.");
    }

    if (!std::isfinite(electronTemperatureRtol) ||
        electronTemperatureRtol <= 0.0 ||
        !std::isfinite(electronTemperatureAtolEv) ||
        electronTemperatureAtolEv < 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "Invalid electron-temperature tolerances.");
    }

    if (!std::isfinite(collisionRateRtol) ||
        collisionRateRtol <= 0.0 ||
        !std::isfinite(collisionRateFloorM3s) ||
        collisionRateFloorM3s <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "collision-rate-rtol and collision-rate-floor-m3-s "
            "must be finite and positive.");
    }

    if (!std::isfinite(tailBudgetFraction) ||
        tailBudgetFraction <= 0.0 ||
        tailBudgetFraction > 1.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "tail-budget-fraction must satisfy 0 < value <= 1.");
    }

    if (!std::isfinite(tailExtrapolationDecades) ||
        tailExtrapolationDecades <= 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "tail-extrapolation-decades must be finite and positive.");
    }

    if (!std::isfinite(minimumPredictorDecayDecades) ||
        minimumPredictorDecayDecades < 8.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "minimum-predictor-decay-decades must be >= 8 for Automatic.");
    }

    if (!std::isfinite(maximumPredictorDecayDecades) ||
        maximumPredictorDecayDecades <= minimumPredictorDecayDecades) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "maximum-predictor-decay-decades must be finite and strictly "
            "greater than minimum-predictor-decay-decades.");
    }

    if (!std::isfinite(minimumObservableTargetMoleFraction) ||
        minimumObservableTargetMoleFraction < 0.0 ||
        minimumObservableTargetMoleFraction > 1.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "minimum-observable-target-mole-fraction must be in [0, 1].");
    }

    if (!std::isfinite(maxEmaxGrowthFactor) ||
        maxEmaxGrowthFactor <= 1.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "max-emax-growth-factor must be greater than one.");
    }

    if (!std::isfinite(minEmaxShrinkFactor) ||
        minEmaxShrinkFactor <= 0.0 ||
        minEmaxShrinkFactor >= 1.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "min-emax-shrink-factor must satisfy 0 < value < 1.");
    }

    if (!std::isfinite(baselineMetricFraction) ||
        baselineMetricFraction <= 0.0 ||
        !std::isfinite(thresholdGuardFraction) ||
        thresholdGuardFraction < 0.0) {
        throw CanteraError(
            "EEDFTwoTermApproximation::setAutomaticGridParameters",
            "Invalid automatic-grid monitor parameters.");
    }

    m_automaticFinalCells = finalCells;

    m_automaticBootstrapCells = bootstrapCells;
    m_automaticBootstrapMaxEnergy = bootstrapMaxEnergy;
    m_automaticBootstrapStretchEnergy = bootstrapStretchEnergy;
    m_automaticBootstrapMaxAttempts = bootstrapMaxAttempts;
    m_automaticBootstrapExpansionFactor = bootstrapExpansionFactor;

    m_automaticPredictorCorrectorIterations =
        predictorCorrectorIterations;

    m_automaticRequireConvergence = requireConvergence;

    m_automaticElectronTemperatureRtol =
        electronTemperatureRtol;

    m_automaticElectronTemperatureAtolEv =
        electronTemperatureAtolEv;

    m_automaticCollisionRateRtol =
        collisionRateRtol;

    m_automaticCollisionRateFloorM3s =
        collisionRateFloorM3s;

    // Legacy YAML values are kept only for the optional disabled debug
    // estimator. Neither influences the Automatic-grid algorithm.
    m_automaticTailBudgetFraction =
        tailBudgetFraction;

    m_automaticTailExtrapolationDecades =
        tailExtrapolationDecades;

    m_automaticMinimumPredictorDecayDecades =
        minimumPredictorDecayDecades;

    m_automaticMaximumPredictorDecayDecades =
        maximumPredictorDecayDecades;

    m_automaticMinimumObservableTargetMoleFraction =
        minimumObservableTargetMoleFraction;

    m_automaticMaxEmaxGrowthFactor =
        maxEmaxGrowthFactor;

    m_automaticMinEmaxShrinkFactor =
        minEmaxShrinkFactor;

    m_automaticBaselineMetricFraction =
        baselineMetricFraction;

    m_automaticThresholdGuardFraction =
        thresholdGuardFraction;

    m_gridType = "Automatic";
    m_adaptGrid = false;

    // Install a valid bootstrap grid immediately. At YAML parsing time the
    // complete collision catalogue may not yet be loaded; that is fine because
    // this grid depends only on bootstrap parameters.
    vector<double> bootstrap =
        makeAutomaticBootstrapGrid(
            m_automaticBootstrapMaxEnergy,
            m_automaticBootstrapCells);

    setCustomGrid(bootstrap);

    m_kTeMax = m_automaticBootstrapMaxEnergy;
    m_initialGridCells = m_automaticBootstrapCells;

    m_has_EEDF = false;
    m_f0_ok = false;
}

void EEDFTwoTermApproximation::enableGridAdaptation(bool enabled)
{
    if (enabled && m_gridType == "Automatic") {
        throw CanteraError(
            "EEDFTwoTermApproximation::enableGridAdaptation",
            "Legacy energy-grid adaptation cannot be enabled when "
            "energy_levels_distribution is Automatic.");
    }

    m_adaptGrid = enabled;
}

void EEDFTwoTermApproximation::setGridAdaptationParameters(bool enabled,
                                                           double minDecayDecades,
                                                           double maxDecayDecades,
                                                           double updateFactor,
                                                           size_t maxIterations, 
                                                           bool maxwellian_reset)
{
    if (enabled && m_gridType == "Automatic") {
        throw CanteraError(
            "EEDFTwoTermApproximation::setGridAdaptationParameters",
            "Legacy energy-grid adaptation cannot be enabled when "
            "energy_levels_distribution is Automatic.");
    }

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
        setGeometricGrid(m_kTeMax, m_initialGridCells, m_geometricGridRatio);
    } else if (m_gridType == "Automatic") {
        throw CanteraError(
            "EEDFTwoTermApproximation::updateGrid",
            "updateGrid() belongs to the legacy grid-adaptation algorithm "
            "and cannot be used with the Automatic grid.");
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

    // When electron-electron collisions are disabled, the EEDF does not
    // depend on either the ionization degree or the electron density.
    // Reactor integrators may temporarily propose tiny negative electron
    // concentrations when the physical concentration approaches zero.
    if (!m_enableEeCollisions) {
        m_eeCol = false;
        m_ionDegree = 0.0;
        m_nElectron = 0.0;
        m_ionDegree_tmp = 0.0;
        m_nElectron_tmp = 0.0;
        return;
    }

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
