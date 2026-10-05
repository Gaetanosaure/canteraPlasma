//! @file ElectronCollisionPlasmaRate.cpp

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/kinetics/ElectronCollisionPlasmaRate.h"
#include "cantera/kinetics/Reaction.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/thermo/PlasmaPhase.h"
#include "cantera/numerics/funcs.h"

namespace Cantera
{

// namespace
// {

// double interpolateCrossSection(
//     double energy,
//     const vector<double>& energyLevels,
//     const vector<double>& crossSections)
// {
//     if (energyLevels.empty() || crossSections.empty()) {
//         throw CanteraError("interpolateCrossSection",
//             "Cross-section data cannot be empty.");
//     }

//     if (energy < energyLevels.front() ||
//         energy > energyLevels.back()) {
//         return 0.0;
//     }

//     return linearInterp(energy, energyLevels, crossSections);
// }

// }

namespace
{

double interpolateCrossSection(
    double energy,
    const vector<double>& energyLevels,
    const vector<double>& crossSections)
{
    if (energyLevels.empty() || crossSections.empty()) {
        throw CanteraError(
            "interpolateCrossSection",
            "Cross-section data cannot be empty.");
    }

    if (energyLevels.size() != crossSections.size()) {
        throw CanteraError(
            "interpolateCrossSection",
            "Cross-section energy and value arrays must have identical sizes.");
    }

    // Cross sections are defined to be zero outside their tabulated domain.
    if (energy < energyLevels.front() ||
        energy > energyLevels.back()) {
        return 0.0;
    }

    return linearInterp(energy, energyLevels, crossSections);
}


// Build a rate-integration grid containing every breakpoint of both:
//   1. the EEDF piecewise-linear representation;
//   2. the cross-section piecewise-linear representation.
//
// sigmaArgumentShift defines which cross section is sampled:
//
//     sigma(energy + sigmaArgumentShift)
//
// Forward collision:
//     sigmaArgumentShift = 0
//
// Super-elastic collision:
//     sigmaArgumentShift = U
//
// In the latter case, native cross-section points x_sigma therefore create
// integration breakpoints at:
//     energy = x_sigma - U
void buildMergedRateIntegrationGrid(
    const vector<double>& eedfEnergyLevels,
    const vector<double>& crossSectionEnergyLevels,
    const vector<double>& crossSections,
    double sigmaArgumentShift,
    vector<double>& integrationEnergy,
    vector<double>& integrationCrossSection)
{
    if (eedfEnergyLevels.size() < 2) {
        throw CanteraError(
            "buildMergedRateIntegrationGrid",
            "EEDF energy grid must contain at least two points.");
    }

    if (crossSectionEnergyLevels.size() < 2 ||
        crossSectionEnergyLevels.size() != crossSections.size()) {
        throw CanteraError(
            "buildMergedRateIntegrationGrid",
            "Invalid cross-section tabulation.");
    }

    integrationEnergy.clear();
    integrationCrossSection.clear();

    // Only integrate where both the EEDF and the shifted cross section exist.
    //
    // sigma(eps + shift) is tabulated when
    //
    //   sigmaEmin <= eps + shift <= sigmaEmax
    //
    // therefore
    //
    //   sigmaEmin - shift <= eps <= sigmaEmax - shift.
    const double lower = std::max(
        eedfEnergyLevels.front(),
        crossSectionEnergyLevels.front() - sigmaArgumentShift);

    const double upper = std::min(
        eedfEnergyLevels.back(),
        crossSectionEnergyLevels.back() - sigmaArgumentShift);

    if (!(upper > lower)) {
        // No overlap between the EEDF domain and cross-section domain.
        return;
    }

    integrationEnergy.reserve(
        eedfEnergyLevels.size() + crossSectionEnergyLevels.size() + 2);

    integrationEnergy.push_back(lower);
    integrationEnergy.push_back(upper);

    // Add every EEDF breakpoint inside the overlap.
    for (double eps : eedfEnergyLevels) {
        if (eps > lower && eps < upper) {
            integrationEnergy.push_back(eps);
        }
    }

    // Add every native cross-section breakpoint, shifted into source-energy
    // coordinates.
    for (double epsSigma : crossSectionEnergyLevels) {
        const double eps = epsSigma - sigmaArgumentShift;

        if (eps > lower && eps < upper) {
            integrationEnergy.push_back(eps);
        }
    }

    std::sort(integrationEnergy.begin(), integrationEnergy.end());

    auto last = std::unique(
        integrationEnergy.begin(),
        integrationEnergy.end());

    integrationEnergy.resize(
        std::distance(integrationEnergy.begin(), last));

    integrationCrossSection.resize(integrationEnergy.size());

    // Cache the cross section on the merged integration grid.
    for (size_t i = 0; i < integrationEnergy.size(); i++) {
        integrationCrossSection[i] =
            interpolateCrossSection(
                integrationEnergy[i] + sigmaArgumentShift,
                crossSectionEnergyLevels,
                crossSections);
    }
}


// Integrate
//
//     integral (eps + energyFactorShift) F(eps) sigma(eps) d eps
//
// on a grid containing every breakpoint of the piecewise-linear EEDF
// and piecewise-linear cross section.
//
// Inside each integration segment:
//
//     F(eps)       = linear
//     sigma(eps)   = linear
//     eps + shift  = linear
//
// so their product is a polynomial of degree <= 3.
//
// Two-point Gauss-Legendre quadrature is therefore exact on each segment
// for these discrete piecewise-linear representations.
double integrateRateOnMergedGrid(
    const vector<double>& integrationEnergy,
    const vector<double>& integrationCrossSection,
    const vector<double>& eedfEnergyLevels,
    const vector<double>& eedfDistribution,
    double energyFactorShift)
{
    if (integrationEnergy.size() < 2) {
        return 0.0;
    }

    if (integrationEnergy.size() != integrationCrossSection.size()) {
        throw CanteraError(
            "integrateRateOnMergedGrid",
            "Integration energy and cross-section arrays have different sizes.");
    }

    if (eedfEnergyLevels.size() != eedfDistribution.size() ||
        eedfEnergyLevels.size() < 2) {
        throw CanteraError(
            "integrateRateOnMergedGrid",
            "Invalid EEDF arrays.");
    }

    // Gauss-Legendre nodes mapped from [-1, 1] to [0, 1].
    constexpr double invSqrt3 =
        0.577350269189625764509148780501957456;

    constexpr double t1 = 0.5 * (1.0 - invSqrt3);
    constexpr double t2 = 0.5 * (1.0 + invSqrt3);

    double integral = 0.0;

    // EEDF value at the first integration node.
    double fLeft = linearInterp(
        integrationEnergy[0],
        eedfEnergyLevels,
        eedfDistribution);

    for (size_t i = 0; i + 1 < integrationEnergy.size(); i++) {
        const double a = integrationEnergy[i];
        const double b = integrationEnergy[i + 1];
        const double h = b - a;

        if (!(h > 0.0)) {
            throw CanteraError(
                "integrateRateOnMergedGrid",
                "Integration grid is not strictly increasing.");
        }

        const double sigmaLeft =
            integrationCrossSection[i];

        const double sigmaRight =
            integrationCrossSection[i + 1];

        const double fRight = linearInterp(
            b,
            eedfEnergyLevels,
            eedfDistribution);

        // Values at the two Gauss points.
        //
        // Because every EEDF and cross-section breakpoint is in the merged
        // grid, both quantities are linear throughout [a,b].
        const double f1 =
            fLeft + t1 * (fRight - fLeft);

        const double f2 =
            fLeft + t2 * (fRight - fLeft);

        const double sigma1 =
            sigmaLeft + t1 * (sigmaRight - sigmaLeft);

        const double sigma2 =
            sigmaLeft + t2 * (sigmaRight - sigmaLeft);

        const double eps1 = a + t1 * h;
        const double eps2 = a + t2 * h;

        const double integrand1 =
            (eps1 + energyFactorShift) * f1 * sigma1;

        const double integrand2 =
            (eps2 + energyFactorShift) * f2 * sigma2;

        // Two-point Gauss-Legendre on [a,b].
        integral += 0.5 * h * (integrand1 + integrand2);

        fLeft = fRight;
    }

    if (!std::isfinite(integral) || integral < 0.0) {
        throw CanteraError(
            "integrateRateOnMergedGrid",
            "Invalid rate integral: {}.",
            integral);
    }

    return integral;
}

}

ElectronCollisionPlasmaData::ElectronCollisionPlasmaData()
{
    energyLevels.assign(1, 0.0);
    distribution.assign(1, 0.0);
}

bool ElectronCollisionPlasmaData::update(
    const ThermoPhase& phase,
    const Kinetics& kin)
{
    auto& pp = const_cast<PlasmaPhase&>(
        dynamic_cast<const PlasmaPhase&>(phase));

    if (pp.electronEnergyDistributionType() == "Boltzmann-two-term") {
        pp.updateElectronEnergyDistribution();
    }

    // Total number density in molecules/m^3
    const double currentNumberDensity =
        pp.molarDensity() * Avogadro;

    if (!std::isfinite(currentNumberDensity)
        || currentNumberDensity <= 0.0) {
        throw CanteraError(
            "ElectronCollisionPlasmaData::update",
            "Total gas number density must be finite and positive. "
            "Current value is {} molecules/m^3.",
            currentNumberDensity);
    }

    // A density-dependent electron-collision rate must be recalculated even
    // when the EEDF itself has not changed.
    const bool numberDensityChanged =
        currentNumberDensity != numberDensity;

    numberDensity = currentNumberDensity;

    const bool distributionChanged =
        pp.distributionNumber() != m_dist_number;

    if (!distributionChanged) {
        return numberDensityChanged;
    }

    m_dist_number = pp.distributionNumber();

    distribution.resize(pp.nElectronEnergyLevels());
    pp.getElectronEnergyDistribution(distribution);

    if (pp.levelNumber() != levelNumber || energyLevels.empty()) {
        levelNumber = pp.levelNumber();
        energyLevels.resize(pp.nElectronEnergyLevels());
        pp.getElectronEnergyLevels(energyLevels);
    }

    return true;
}

void ElectronCollisionPlasmaRate::setParameters(const AnyMap& node,
                                                const UnitStack& rate_units)
{

    ReactionRate::setParameters(node, rate_units);

    if (node.hasKey("collision")) {
        m_collisionName = node["collision"].asString();
    }

    bool hasInlineCrossSectionData =
        node.hasKey("energy-levels") && node.hasKey("cross-sections");

    bool isNamedElectronCollision = node.hasKey("name");
    bool isCollisionReference = node.hasKey("collision");

    if (hasInlineCrossSectionData) {
        if (!isNamedElectronCollision && !isCollisionReference) {
            writelog("CAREFUL! Inline electron-collision cross-section data without a "
                    "'name' entry are deprecated. Please move these data to a named "
                    "'electron-collisions' entry and reference it using 'collision'.\n");
        }

        applyCollisionData(node);
    }

}

void ElectronCollisionPlasmaRate::getParameters(AnyMap& node) const
{
    node["type"] = type();

    node["energy-levels"] = m_energyLevels;
    node["cross-sections"] = m_crossSections;

    if (m_threshold != 0.0) {
        node["threshold"] = m_threshold;
    }

    if (!m_kind.empty()) {
        node["kind"] = m_kind;
    }

    if (!m_target.empty()) {
        node["target"] = m_target;
    }

    if (!m_product.empty()) {
        node["product"] = m_product;
    }
    if (!m_correspondingSpecies.empty()) {
        node["corresponding-species"] = m_correspondingSpecies;
    }

    if (m_superElasticDegeneracyRatio != 1.0) {
        node["super-elastic-degeneracy-ratio"] = m_superElasticDegeneracyRatio;
    }
}

void ElectronCollisionPlasmaRate::updateInterpolatedCrossSection(
    span<const double> sharedLevels)
{
    m_crossSectionsInterpolated.clear();
    m_crossSectionsInterpolated.reserve(sharedLevels.size());
    for (double level : sharedLevels) {
        m_crossSectionsInterpolated.emplace_back(
        interpolateCrossSection(level, m_energyLevels, m_crossSections));
    }
}

// double ElectronCollisionPlasmaRate::evalFromStruct(
//     const ElectronCollisionPlasmaData& shared_data)
// {
//     // Interpolate cross sections when the electron-energy grid changes.
//     if (m_levelNumber != shared_data.levelNumber) {
//         m_crossSectionsInterpolated.clear();
//         m_crossSectionsInterpolated.reserve(
//             shared_data.energyLevels.size());

//         for (double level : shared_data.energyLevels) {
//             m_crossSectionsInterpolated.push_back(
//                 linearInterp(
//                     level,
//                     m_energyLevels,
//                     m_crossSections));
//         }

//         m_levelNumber = shared_data.levelNumber;
//     }

//     AssertThrowMsg(
//         m_crossSectionsInterpolated.size()
//             == shared_data.distribution.size(),
//         "ElectronCollisionPlasmaRate::evalFromStruct",
//         "Size mismatch: len(interp) = {}, len(distribution) = {}",
//         m_crossSectionsInterpolated.size(),
//         shared_data.distribution.size());

//     auto crossSections = Eigen::Map<const Eigen::ArrayXd>(
//         m_crossSectionsInterpolated.data(),
//         m_crossSectionsInterpolated.size());

//     auto energy = Eigen::Map<const Eigen::ArrayXd>(
//         shared_data.energyLevels.data(),
//         shared_data.energyLevels.size());

//     auto distribution = Eigen::Map<const Eigen::ArrayXd>(
//         shared_data.distribution.data(),
//         shared_data.distribution.size());

//     // Standard binary electron-collision coefficient [m^3/kmol/s]
//     double kf =
//         std::sqrt(2.0 * ElectronCharge / ElectronMass)
//         * Avogadro
//         * simpson(
//             energy.cwiseProduct(
//                 distribution.cwiseProduct(crossSections)),
//             energy);

//     // Special density-dependent attachment:
//     //
//     //     Electron + O2 + M -> O2- + M
//     //
//     // The cross-section table is understood to contain the generalized
//     // three-body attachment data. The third body is folded into the effective
//     // bimolecular coefficient using the total density in molecules/cm^3.
//     const bool isO2ThreeBodyAttachment =
//         m_kind == "attachment"
//         && m_target == "O2"
//         && (m_product == "O2^-" || m_product == "O2-");

//     if (isO2ThreeBodyAttachment) {
//         const double numberDensityCm3 =
//             shared_data.numberDensity * 1.0e-6;

//         kf *= numberDensityCm3;
//     }

//     if (!std::isfinite(kf) || kf < 0.0) {
//         throw CanteraError(
//             "ElectronCollisionPlasmaRate::evalFromStruct",
//             "Invalid electron-collision rate coefficient for collision '{}': {}.",
//             m_collisionName,
//             kf);
//     }

//     return kf;
// }

double ElectronCollisionPlasmaRate::evalFromStruct(
    const ElectronCollisionPlasmaData& shared_data)
{
    // Keep the cross section interpolated on the PlasmaPhase EEDF grid
    // synchronized for callers that explicitly use crossSectionInterpolated().
    //
    // IMPORTANT:
    // use interpolateCrossSection(), not bare linearInterp(), so the behavior
    // outside the tabulated cross-section domain is consistently zero.
    if (m_levelNumber != shared_data.levelNumber) {
        m_crossSectionsInterpolated.clear();
        m_crossSectionsInterpolated.reserve(
            shared_data.energyLevels.size());

        for (double level : shared_data.energyLevels) {
            m_crossSectionsInterpolated.push_back(
                interpolateCrossSection(
                    level,
                    m_energyLevels,
                    m_crossSections));
        }

        m_levelNumber = shared_data.levelNumber;
    }

    // Rebuild the accurate forward-rate integration grid only when the EEDF
    // energy grid itself changes. Changes in F0 do not require rebuilding
    // this geometrical cache.
    if (m_rateIntegrationLevelNumber != shared_data.levelNumber) {
        buildMergedRateIntegrationGrid(
            shared_data.energyLevels,
            m_energyLevels,
            m_crossSections,
            0.0, // forward rate samples sigma(eps)
            m_rateIntegrationEnergy,
            m_rateIntegrationCrossSection);

        m_rateIntegrationLevelNumber =
            shared_data.levelNumber;
    }

    // Compute
    //
    //     I = integral eps * F0(eps) * sigma(eps) d eps
    //
    // using a grid containing all EEDF and cross-section breakpoints.
    const double integral =
        integrateRateOnMergedGrid(
            m_rateIntegrationEnergy,
            m_rateIntegrationCrossSection,
            shared_data.energyLevels,
            shared_data.distribution,
            0.0);

    // Standard binary electron-collision coefficient [m^3/kmol/s]
    double kf =
        std::sqrt(2.0 * ElectronCharge / ElectronMass)
        * Avogadro
        * integral;

    // Special density-dependent attachment:
    //
    //     Electron + O2 + M -> O2- + M
    //
    // The cross-section table is understood to contain the generalized
    // three-body attachment data. The third body is folded into the effective
    // bimolecular coefficient using the total density in molecules/cm^3.
    const bool isO2ThreeBodyAttachment =
        m_kind == "attachment"
        && m_target == "O2"
        && (m_product == "O2^-" || m_product == "O2-");

    if (isO2ThreeBodyAttachment) {
        const double numberDensityCm3 =
            shared_data.numberDensity * 1.0e-6;

        kf *= numberDensityCm3;
    }

    if (!std::isfinite(kf) || kf < 0.0) {
        throw CanteraError(
            "ElectronCollisionPlasmaRate::evalFromStruct",
            "Invalid electron-collision rate coefficient for collision '{}': {}.",
            m_collisionName,
            kf);
    }

    return kf;
}

// void ElectronCollisionPlasmaRate::modifyRateConstants(
//     const ElectronCollisionPlasmaData& shared_data, double& kf, double& kr)
// {
//     if (kr == 0.0) {
//         // The reverse rate constant is only for reversible reactions
//         // kr = 0.0 indicates that the reaction is irreversible
//         return;
//     }

//     // Defensive guard: reverse super-elastic rates are physically meaningful
//     // only for excitation/de-excitation channels.
//     if (m_kind != "excitation") {
//         kr = 0.0;
//         return;
//     }

//     // Interpolate cross-sections data to the energy levels of
//     // the electron energy distribution function
//     if (m_levelNumberSuperelastic != shared_data.levelNumber) {
//         // super elastic collision energy levels and cross-sections
//         vector<double> superElasticEnergyLevels{0.0};
//         m_crossSectionsOffset.resize(shared_data.energyLevels.size());
//         for (size_t i = 1; i < m_energyLevels.size(); i++) {
//             // The energy levels are offset by the first energy level (threshold)
//             superElasticEnergyLevels.push_back(m_energyLevels[i] - m_threshold);
//         }
//         for (size_t i = 0; i < shared_data.energyLevels.size(); i++) {
//             // The interpolated super-elastic cross section is evaluated
//             // at the shared energy grid
//             m_crossSectionsOffset[i] = interpolateCrossSection(
//                                         shared_data.energyLevels[i],
//                                         superElasticEnergyLevels,
//                                         m_crossSections);
//         }
//         m_levelNumberSuperelastic = shared_data.levelNumber;
//     }

//     // Map energyLevels in Eigen::ArrayXd
//     auto eps = Eigen::Map<const Eigen::ArrayXd>(
//         shared_data.energyLevels.data(), shared_data.energyLevels.size()
//     );

//     // Map the electron energy distribution to Eigen::ArrayXd.
//     auto distribution = Eigen::Map<const Eigen::ArrayXd>(
//         shared_data.distribution.data(), shared_data.distribution.size()
//     );

//     // unit in kmol/m3/s
//     kr = m_superElasticDegeneracyRatio* pow(2.0 * ElectronCharge / ElectronMass, 0.5) * Avogadro *
//          simpson((eps + m_threshold).cwiseProduct(
//          distribution.cwiseProduct(m_crossSectionsOffset)), eps);

//     // writelog(
//     //     "EEDF-SE-DEBUG RATE_REVERSE_KR: collision='{}', kind='{}', "
//     //     "threshold={}, degeneracyRatio={}, kf={}, kr={}, nGrid={}, "
//     //     "gridFirst={}, gridLast={}\n",
//     //     m_collisionName,
//     //     m_kind,
//     //     m_threshold,
//     //     m_superElasticDegeneracyRatio,
//     //     kf,
//     //     kr,
//     //     shared_data.energyLevels.size(),
//     //     shared_data.energyLevels.empty() ? -1.0 : shared_data.energyLevels.front(),
//     //     shared_data.energyLevels.empty() ? -1.0 : shared_data.energyLevels.back()
//     // );
// }

void ElectronCollisionPlasmaRate::modifyRateConstants(
    const ElectronCollisionPlasmaData& shared_data,
    double& kf,
    double& kr)
{
    if (kr == 0.0) {
        // The reverse rate constant is only for reversible reactions.
        // kr = 0.0 indicates that the reaction is irreversible.
        return;
    }

    // Reverse super-elastic rates are physically meaningful only for
    // excitation/de-excitation channels.
    if (m_kind != "excitation") {
        kr = 0.0;
        return;
    }

    const double U = m_threshold;

    // Rebuild the super-elastic integration geometry only when the EEDF
    // energy grid changes.
    //
    // The reverse integral uses the forward excitation cross section at:
    //
    //     sigma_exc(eps + U)
    //
    // Therefore native cross-section breakpoints eps_sigma become source
    // energy breakpoints:
    //
    //     eps = eps_sigma - U.
    if (m_superElasticRateIntegrationLevelNumber
        != shared_data.levelNumber) {

        buildMergedRateIntegrationGrid(
            shared_data.energyLevels,
            m_energyLevels,
            m_crossSections,
            U, // samples sigma_exc(eps + U)
            m_superElasticRateIntegrationEnergy,
            m_superElasticRateIntegrationCrossSection);

        m_superElasticRateIntegrationLevelNumber =
            shared_data.levelNumber;
    }

    // From detailed balance:
    //
    //   sigma_se(eps)
    //       = ratio * (eps + U) / eps * sigma_exc(eps + U)
    //
    // and therefore:
    //
    //   eps * sigma_se(eps)
    //       = ratio * (eps + U) * sigma_exc(eps + U).
    //
    // Thus we integrate:
    //
    //   (eps + U) F0(eps) sigma_exc(eps + U)
    //
    // on the merged source-energy grid.
    const double integral =
        integrateRateOnMergedGrid(
            m_superElasticRateIntegrationEnergy,
            m_superElasticRateIntegrationCrossSection,
            shared_data.energyLevels,
            shared_data.distribution,
            U);

    kr =
        m_superElasticDegeneracyRatio
        * std::sqrt(2.0 * ElectronCharge / ElectronMass)
        * Avogadro
        * integral;

    if (!std::isfinite(kr) || kr < 0.0) {
        throw CanteraError(
            "ElectronCollisionPlasmaRate::modifyRateConstants",
            "Invalid reverse electron-collision rate coefficient for collision '{}': {}.",
            m_collisionName,
            kr);
    }
}

void ElectronCollisionPlasmaRate::setContext(const Reaction& rxn, const Kinetics& kin)
{

    const ThermoPhase& thermo = kin.thermo();
    // get electron species name
    string electronName;
    if (thermo.type() == "plasma") {
        electronName = dynamic_cast<const PlasmaPhase&>(thermo).electronSpeciesName();
    } else {
        throw CanteraError("ElectronCollisionPlasmaRate::setContext",
                           "ElectronCollisionPlasmaRate requires plasma phase");
    }

    // Number of reactants needs to be two
    if (rxn.reactants.size() != 2) {
        throw InputFileError("ElectronCollisionPlasmaRate::setContext", rxn.input,
            "ElectronCollisionPlasmaRate requires exactly two reactants");
    }

    // Must have only one electron
    // @todo add electron-electron collision rate
    if (rxn.reactants.at(electronName) != 1) {
        throw InputFileError("ElectronCollisionPlasmaRate::setContext", rxn.input,
            "ElectronCollisionPlasmaRate requires one and only one electron");
    }

    // Determine the "kind" of collision if not specified explicitly
    if (m_kind.empty()) {
        m_kind = "excitation"; // default
        if (rxn.reactants == rxn.products) {
            m_kind = "effective";
        } else {
            for (const auto& [p, stoich] : rxn.products) {
                if (p == electronName) {
                    continue;
                }
                double q = thermo.charge(thermo.speciesIndex(p, true));
                if (q > 0) {
                    m_kind = "ionization";
                } else if (q < 0) {
                    m_kind = "attachment";
                }
            }
        }
    }

    setDefaultThreshold();

    // writelog(
    //     "EEDF-SE-DEBUG RATE_CONTEXT: equation='{}', reversible={}, "
    //     "collision='{}', inferredOrStoredKind='{}', threshold={}, "
    //     "target='{}', product='{}', correspondingSpecies='{}', degeneracyRatio={}\n",
    //     rxn.equation(),
    //     rxn.reversible,
    //     m_collisionName,
    //     m_kind,
    //     m_threshold,
    //     m_target,
    //     m_product,
    //     m_correspondingSpecies,
    //     m_superElasticDegeneracyRatio
    // );

    if (!rxn.reversible) {
        return; // end checking of forward reaction
    }

    // Only excitation-like collisions may have a reverse super-elastic rate.
    // Ionization, attachment, effective and elastic collisions must never be
    // interpreted as super-elastic channels.
    if (m_kind != "excitation") {
        throw InputFileError("ElectronCollisionPlasmaRate::setContext", rxn.input,
            "Only electron-impact excitation reactions can be reversible "
            "for super-elastic de-excitation. Reaction '{}' was classified as '{}'.",
            rxn.equation(), m_kind);
    }

    // For super-elastic collisions
    if (rxn.products.size() != 2) {
        throw InputFileError("ElectronCollisionPlasmaRate::setContext", rxn.input,
            "ElectronCollisionPlasmaRate requires exactly two products"
            " if the reaction is reversible (super-elastic collisions)");
    }

    // Must have only one electron
    if (rxn.products.at(electronName) != 1) {
        throw InputFileError("ElectronCollisionPlasmaRate::setContext", rxn.input,
            "ElectronCollisionPlasmaRate requires one and only one electron in products"
            " if the reaction is reversible (super-elastic collisions)");
    }
}

void ElectronCollisionPlasmaRate::applyCollisionData(const AnyMap& node)
{
    if (node.hasKey("kind")) {
        string collisionKind = node["kind"].asString();

        if (!m_kind.empty() && m_kind != collisionKind) {
            string collisionName = node.hasKey("name") ? node["name"].asString() : m_collisionName;

            bool allowCollapsedInelastic =
                (m_kind == "effective" || m_kind == "elastic") &&
                collisionKind == "excitation";

            if (!allowCollapsedInelastic) {
                throw InputFileError("applyCollisionData", node,
                    "Electron collision '{}' has kind '{}', but the reaction was inferred as '{}'.",
                    collisionName, collisionKind, m_kind);
            }

            warn_user("ElectronCollisionPlasmaRate::applyCollisionData",
                "Electron collision '{}' has kind '{}', but the reaction was inferred "
                "as '{}'. Treating this as an intentionally collapsed inelastic "
                "electron-collision channel. No species source term will be generated "
                "for the unresolved product.",
                collisionName, collisionKind, m_kind);
        }

        // Important: keep the collision classified using the data-file kind.
        // For collapsed channels such as N2(rot), this ensures that the
        // Boltzmann solver treats the channel as inelastic.
        m_kind = collisionKind;
    }

    if (node.hasKey("target")) {
        m_target = node["target"].asString();
    }

    if (node.hasKey("product")) {
        m_product = node["product"].asString();
    }

    if (node.hasKey("corresponding-species")) {
        m_correspondingSpecies = node["corresponding-species"].asString();
        if (m_correspondingSpecies.empty()) {
            throw InputFileError("applyCollisionData", node,
                "'corresponding-species' cannot be empty.");
        }
    }

    m_superElasticDegeneracyRatio =
        node.getDouble("super-elastic-degeneracy-ratio", 1.0);

    if (!std::isfinite(m_superElasticDegeneracyRatio) ||
        m_superElasticDegeneracyRatio <= 0.0) {
        throw InputFileError("applyCollisionData", node,
            "'super-elastic-degeneracy-ratio' must be finite and positive.");
    }

    if (!node.hasKey("energy-levels")) {
        throw InputFileError("applyCollisionData", node, "Missing 'energy-levels'");
    }

    if (!node.hasKey("cross-sections")) {
        throw InputFileError("applyCollisionData", node, "Missing 'cross-sections'");
    }

    m_energyLevels = node["energy-levels"].asVector<double>();
    m_crossSections = node["cross-sections"].asVector<double>(m_energyLevels.size());
    m_threshold = node.getDouble("threshold", 0.0);

    setDefaultThreshold();

    validateCollisionData(node);
    m_hasCrossSectionData = true;

    // Collision data changed: invalidate every interpolation / integration cache.
    m_crossSectionsInterpolated.clear();

    m_rateIntegrationEnergy.clear();
    m_rateIntegrationCrossSection.clear();

    m_superElasticRateIntegrationEnergy.clear();
    m_superElasticRateIntegrationCrossSection.clear();

    m_levelNumber = -3;
    m_levelNumberSuperelastic = -2;

    m_rateIntegrationLevelNumber = -4;
    m_superElasticRateIntegrationLevelNumber = -5;

    // writelog(
    //     "EEDF-SE-DEBUG RATE_APPLY_DATA: collision='{}', kind='{}', "
    //     "target='{}', product='{}', correspondingSpecies='{}', "
    //     "threshold={}, degeneracyRatio={}, nLevels={}, "
    //     "firstLevel={}, firstSigma={}, lastLevel={}, lastSigma={}\n",
    //     m_collisionName,
    //     m_kind,
    //     m_target,
    //     m_product,
    //     m_correspondingSpecies,
    //     m_threshold,
    //     m_superElasticDegeneracyRatio,
    //     m_energyLevels.size(),
    //     m_energyLevels.empty() ? -1.0 : m_energyLevels.front(),
    //     m_crossSections.empty() ? -1.0 : m_crossSections.front(),
    //     m_energyLevels.empty() ? -1.0 : m_energyLevels.back(),
    //     m_crossSections.empty() ? -1.0 : m_crossSections.back()
    // );
}

void ElectronCollisionPlasmaRate::validateCollisionData(const AnyMap& node) const
{
    if (m_energyLevels.size() < 2) {
        throw InputFileError("validateCollisionData" , node, "Need at least two energy levels.");
    }

    if (m_energyLevels.size() != m_crossSections.size()) {
        throw InputFileError("validateCollisionData" , node, "energy-levels and cross-sections size mismatch.");
    }

    for (size_t i = 0; i < m_energyLevels.size(); i++) {
        if (!std::isfinite(m_energyLevels[i]) || m_energyLevels[i] < 0.0) {
            throw InputFileError("validateCollisionData" , node, "Inifnite or negative energy level value");
        }
        if (!std::isfinite(m_crossSections[i]) || m_crossSections[i] < 0.0) {
            throw InputFileError("validateCollisionData" , node, "Inifnite or negative cross-section value");
        }
        if (i > 0 && m_energyLevels[i] <= m_energyLevels[i - 1]) {
            throw InputFileError("validateCollisionData" , node, "energy-levels must be strictly increasing.");
        }
    }

    // if (!std::isfinite(m_threshold) || m_threshold < 0.0) {
    //     throw InputFileError("validateCollisionData" , node, "Inifnite or negative threshold value");
    // }

    if (!std::isfinite(m_threshold)) {
        throw InputFileError("validateCollisionData", node,
            "Infinite or non-finite threshold value.");
    }

    if (m_threshold < 0.0 && m_kind != "excitation") {
        throw InputFileError("validateCollisionData", node,
            "Negative threshold values are only allowed for diagnostic "
            "manual super-elastic excitation-like collisions.");
    }

    if (m_threshold < 0.0 && m_kind == "excitation") {
        warn_user("ElectronCollisionPlasmaRate::validateCollisionData",
            "Negative threshold value detected for an excitation-like collision."
            "This will be treated as a super-elastic collisions.");
    }
}

void ElectronCollisionPlasmaRate::setDefaultThreshold()
{
    if (m_threshold != 0.0 || m_energyLevels.empty()) {
        return;
    }

    if (m_kind != "excitation" && m_kind != "ionization" && m_kind != "attachment") {
        return;
    }

    for (double level : m_energyLevels) {
        if (level > 0.0) {
            m_threshold = level;
            break;
        }
    }
}

const string& ElectronCollisionPlasmaRate::collisionName() const
{
    return m_collisionName;
}

bool ElectronCollisionPlasmaRate::hasCrossSectionData() const
{
    return m_hasCrossSectionData;
}


}
