//! @file PlasmaReactor.h

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_PLASMAREACTOR_H
#define CT_PLASMAREACTOR_H

#include "IdealGasReactor.h"
#include "cantera/thermo/PlasmaPhase.h"

namespace Cantera
{

/**
 * @class PlasmaReactor
 * @brief Stirred reactor (IdealGasReactor-based) with plasma power deposition
 *        and an optional vibrational-energy model.
 *
 * PlasmaReactor extends IdealGasReactor with:
 * - **Volumetric plasma power deposition** added to the energy equation.
 * - An **optional vibrational-energy model** tracking one scalar vibrational
 *   energy per selected species instead of solving all vibrational levels.
 *
 * This reduces the number of transport equations for vibrational DOFs to one
 * per vibrational species while retaining their coupling to the thermal energy
 * via vibration–translation (V–T) relaxation.
 *
 * \section equations Governing equations
 *
 * \subsection mass Mass conservation
 * \f[
 * \frac{dm}{dt} = \dot{m}_{in} - \dot{m}_{out}
 * \f]
 *
 * \subsection species Species conservation (for species @f$k@f$)
 * \f[
 * m \, \frac{dY_k}{dt} =
 *   V W_k \, \dot{\omega}_k +
 *   \dot{Y}_{k,in} - \dot{Y}_{k,out}
 * \f]
 *
 * \subsection energy Energy equation
 * \f[
 * m c_v \frac{dT}{dt} =
 *   - P \frac{dV}{dt}
 *   + V \, \dot{\omega}_T
 *   + \dot{E}^p
 *   - \dot{E}_{vib}^p
 *   + \dot{R}_{VT}
 *   - \dot{E}_{out}
 *   + \dot{E}_{in}
 * \f]
 *
 * \subsection vib Vibrational-energy equations
 * Per vibrational species @f$j@f$:
 * \f[
 * \frac{dE_{vib}^{(p,j)}}{dt} =
 *   \dot{E}_{vib,j}^p
 *   - \dot{R}_{VT}^j
 *   - \dot{E}_{vib,j,out}^p
 *   + \dot{E}_{vib,j,in}^p
 * \f]
 *
 * Total plasma vibrational power and total V–T relaxation power:
 * \f[
 * \dot{E}_{vib}^p = \sum_{j \in \text{vib species}} \dot{E}_{vib,j}^p,
 * \qquad
 * \dot{R}_{VT}   = \sum_{j \in \text{vib species}} \dot{R}_{VT}^j
 * \f]
 *
 * V–T relaxation closure:
 * \f[
 * \dot{R}_{VT}^j = \frac{\dot{E}_{vib,j}^p}{\tau_j},
 * \qquad
 * \frac{1}{\tau_j} = \sum_{c \in \text{collision partners}} \frac{1}{\tau_{jc}},
 * \qquad
 * \tau_{jc} = \frac{1}{[c]\,k_{jc}}
 * \f]
 *
 * \section params Plasma-related symbols and units
 * \li @f$\dot{E}^p@f$ — **Plasma power** supplied to the mixture \[W\].
 * \li @f$\dot{E}_{vib}^p@f$ — **Plasma power exciting vibrational DOFs** (all species) \[W\].
 * \li @f$\dot{R}_{VT}@f$ — **V–T relaxation power** returned from vibrational to translational energy (total) \[W\].
 * \li @f$E_{vib}^j@f$ — **Energy stored in vibrational DOFs of species @f$j@f$** \[J\].
 * \li @f$\dot{E}_{vib,j}^p@f$ — **Plasma power exciting vibrational DOFs of species @f$j@f$** \[W\].
 * \li @f$\dot{R}_{VT}^j@f$ — **V–T relaxation power of species @f$j@f$** \[W\].
 * \li @f$\tau_j@f$ — **Characteristic V–T relaxation time of species @f$j@f$ in the mixture** \[s\].
 * \li @f$\tau_{jc}@f$ — **Characteristic V–T relaxation time of species @f$j@f$ with collision partner @f$c@f$** \[s\].
 * \li @f$k_{jc}@f$ — **Reaction constant for V–T relaxation between species @f$j@f$ and @f$c@f$** \[kmol·m@sup{-3}·s@sup{-1}\].
 *
 * \section models Relaxation-time models
 * The characteristic time @f$\tau_j@f$ can be:
 * - **Constant model**: user-specified constant per species.
 * - **Correlation-based models** (literature):
 *   Millikan–White (1963), Castela et al. (2016), Starikovskiy & Aleksandrov (2013).
 *   In Millikan–White and Castela, correlations give @f$\tau_{jc}@f$ directly;
 *   in Starikovskiy–Aleksandrov, correlations give @f$k_{jc}@f$ and
 *   we use first-order kinetics @f$\tau_{jc}=1/([c]\,k_{jc})@f$.
 *
 * \section refs References
 * - S. C. Millikan & D. R. White, *Systematics of Vibrational Relaxation*, J. Chem. Phys., 1963.
 * - M. Castela et al., *Combustion and Flame*, 2016.
 * - A. Y. Starikovskiy & N. B. Aleksandrov, *Progress in Energy and Combustion Science*, 2013.
 * 
 *@warning  This class is an experimental part of %Cantera and may be
 *           changed or removed without notice.
 * @see IdealGasReactor
 * @see compute_TauRelax()
 * @ingroup reactorGroup
 */

class PlasmaReactor : public IdealGasReactor
{
public:
    using IdealGasReactor::IdealGasReactor; // inherit constructors
    using IdealGasReactor::setThermo;

    string type() const override {
        return "PlasmaReactor";
    }

    void getState(double* y) override;

    void initialize(double t0=0.0) override;

    void updateState(double* y) override;

    void eval(double t, double* LHS, double* RHS) override;

    //! Set/Get discharge volume
    void setDisVol(double dis_vol) {
        m_dis_vol = dis_vol;
    }
    double disVol() const {
        return m_dis_vol;
    }

    //! Get discharge volumetric power
    //CQM may not be up to date
    // 
    double disVPower() const{
        return m_disVPower;
    }

    size_t componentIndex(const string& nm) const override;
    string componentName(size_t k) override;

    void compute_disVPower();

    void compute_disVibVPower();

    void compute_RvtVPower();

    void recoverVibSpecies();

    double compute_TauRelax(size_t n);

    double  tau_millikan_white(string spec_name);

    double  tau_castela(string spec_name);

    double  tau_starikovskiy(size_t n);

    std::vector<double> get_disVibVPower();

    std::vector<double> get_RvtVPower();

    std::vector<double> get_eVib();

    void setVibRelaxType(string relax_type_name);

    string getVibRelaxType();

    double getVibConstantModelTauRelax();

    void setVibConstantModelTauRelax(double tau_to_set);

    double Max(double a, double b);

    // a structure to store the relaxation time data for the starikovski model. The data will be read from a yaml file provided by the user.
    struct RelaxationEntry {
        std::string name;
        std::string target;
        double A, n, K, B, C, m, D, z;
    };

    double compute_k(const RelaxationEntry& entry, double T);

    string starikovskiy_yaml_path = "init";  // the path to the yaml file containing the relaxation data for the Starikovskiy model provided by the user

    void setStarikovskiyYamlPath(string path) {
        starikovskiy_yaml_path = path;
    }

    string getStarikovskiyYamlPath() {
        return starikovskiy_yaml_path;
}

    void readStarikovskiyRelaxYamlFile(string filename);

    void initializeStarikovskiyReading(){
        starikovskiy_read = false;
    }


protected:
    void setThermo(ThermoPhase& thermo) override;

    double m_dis_vol = 1; //!< Discharge volume

    double m_disVPower = 0; //!< Volumetric discharge power

    std::vector<double> disVibVPower; //!< Volumetric discharge power going into vibrational excitation

    std::vector<double> RvtVPower; // Vibrational energy relaxation into heat

    size_t m_nspevib = 0; //!< Number of species with vibrational excitation

    std::vector<std::string> vib_spec; // a vector to store the names of vibrational species

    PlasmaPhase* m_plasma = nullptr; // pointer to the plasma phase initialisation

    string relax_type = "Constant"; // relaxation type to be chosen by the user. It will be Castela, Starikovski, Constant or MillikanandWhite

    double tau_relax_constant_model = 1e-4; // relaxation time for the constant model

    std::vector<std::vector<RelaxationEntry>> m_data_starikovskiy; // relaxation data input from the relaxation yaml file provided by the user. Used by the Starikovskiy model

    bool starikovskiy_read = false; // boolean to check if the yaml file has been read or not, to avoid reading it several times


    Kinetics* m_kinetics;


};
}

#endif
