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
 * Class PlasmaReactor is a class for stirred reactors that ...
 *
 *
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

    Kinetics* m_kinetics;


};
}

#endif
