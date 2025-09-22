//! @file PlasmaReactor.cpp

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/zeroD/PlasmaReactor.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/PlasmaPhase.h"
#include "cantera/zeroD/FlowDevice.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/base/utilities.h"
#include "cantera/base/global.h"
#include "cantera/base/ct_defs.h"  // contient findInputFile()


#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace Cantera
{

void PlasmaReactor::setThermo(ThermoPhase& thermo)
{
    writelog("PlasmaReactor::setThermo(ThermoPhase&)\n");

    // Cast first so the error is precise
    auto* p = dynamic_cast<PlasmaPhase*>(&thermo);
    if (!p) {
        throw CanteraError("PlasmaReactor::setThermo(ThermoPhase&)",
                           "Thermo is not a PlasmaPhase; got '{}'.", thermo.type());
    }

    // Call the immediate base to keep hierarchy consistent
    IdealGasReactor::setThermo(thermo);

    m_plasma = p;

}
// old version before me trying to reinforce getState function. Keep it for now bc I am not sure the new version is that much better, but it is for sure more complex.
// void PlasmaReactor::getState(double* y)
// {
//     writelog("\nEntering getState function \n");
//     if (m_plasma == 0) {
//         throw CanteraError("IdealGasReactor::getState",
//                            "Error: reactor is empty.");
//     }
//     m_plasma->restoreState(m_state);
//     printf("\nHEY1\n");
//     // set the first component to the total mass
//     m_mass = m_plasma->density() * m_vol;
//     y[0] = m_mass;
//     printf("\nHEY2\n");
//     // set the second component to the total volume
//     y[1] = m_vol;

//     // Set the third component to the temperature
//     y[2] = m_plasma->temperature();
//     printf("\nHEY3\n");

//     // set components y+3 ... y+K+2 to the vibrational energy of each species
//     m_plasma->getVibrationalEnergies(y+3);
//     printf("\nHEY4\n");

//     // set components y+3+m_nspevib ... y+K+2+m_nspevib to the mass fractions of each species
//     m_plasma->getMassFractions(y+3+m_nspevib);
//     printf("\nHEY5\n");


//     // set the remaining components to the surface species
//     // coverages on the walls
//     getSurfaceInitialConditions(y + m_nsp + 3 + m_nspevib);

//     printf("\nExiting getState function\n");
// }

void PlasmaReactor::getState(double* y)
{
    //writelog("\nEntering getState function \n");
    if (!m_plasma) {
        throw CanteraError("PlasmaReactor::getState", "Error: reactor is empty.");
    }

    std::fill(y, y + m_nv, 0.0);    // safety

    m_plasma->restoreState(m_state);

    // base 3
    m_mass = m_plasma->density() * m_vol;
    y[0] = m_mass;
    y[1] = m_vol;
    y[2] = m_plasma->temperature();

    // vib block
    if (m_nspevib > 0) {
        m_plasma->getVibrationalEnergies(y + 3);
    }

    // species block
    m_plasma->getMassFractions(y + 3 + m_nspevib);

    // surfaces block. It is put here but normally we should not go through it as we have no surface reactions. 
    const size_t nsurf = (m_nv > (3 + m_nspevib + m_nsp))
        ? (m_nv - (3 + m_nspevib + m_nsp)) : 0;
    if (nsurf > 0) {
        getSurfaceInitialConditions(y + 3 + m_nspevib + m_nsp);
    }

    //printf("\nExiting getState function\n");
}

void PlasmaReactor::initialize(double t0) {
    writelog("PlasmaReactor::initialize\n");

    // If our override wasn’t reached during construction, recover here.
    if (!m_plasma) {
        if (!m_thermo) {
            throw CanteraError("PlasmaReactor::initialize",
                "No thermo attached to this reactor.");
        }
        if (auto* p = dynamic_cast<PlasmaPhase*>(m_thermo)) {
            m_plasma = p;
        } else {
            throw CanteraError("PlasmaReactor::initialize",
                "Attached thermo is not PlasmaPhase; got '{}'.", m_thermo->type());
        }

    }

    IdealGasReactor::initialize(t0);

    // Equations for vibrational energy density are initialised here.
    m_nspevib = m_plasma->nsp_evib();
    printf("\n Number of species with vibrational excitation seen in PlasmaReactor::initialize: %d \n", int(m_nspevib));
    
    m_nv += m_nspevib; // update the count of m_nv to send the right number of equations to ReactorNet::initialize.
    const auto expected = 3 + m_nspevib + m_nsp; // + nsurf if you have them
    if (expected > m_nv) {
        throw CanteraError("PlasmaReactor::initialize",
            "State layout exceeds m_nv: expected {} > m_nv {}", expected, m_nv);
    }
    
    disVibVPower.resize(m_nspevib);
    
    RvtVPower.resize(m_nspevib);
    
    recoverVibSpecies(); // get all the vibrationnal species declared in the plasma phase
    
    compute_disVPower();

    writelogf("PlasmaReactor::initialize done: m_nspevib=%zu, m_nsp=%zu, m_nv=%zu, expected=%zu\n",
        m_nspevib, m_nsp, m_nv, size_t(3 + m_nspevib + m_nsp));

    printf("\n Exiting initialise \n");
}

void PlasmaReactor::updateState(double* y)
{
    // The components of y are [0] the total mass, [1] the total volume,
    // [2] the temperature, [3...K+3] are the species vibrational energies,
    // [3+m_nspevib...K+3+m_nspevib] are the mass fractions of each species,
    // and [K+3+m_nspevib...] are the coverages of surface species on each wall.
    //printf("entering update_state function");
    m_mass = y[0];
    m_vol = y[1];
    m_plasma->setVibrationalEnergies(y+3);
    m_plasma->setMassFractions_NoNorm(y+3+m_nspevib);
    m_plasma->setState_TD(y[2], m_mass / m_vol);
    updateConnected(true);
    const size_t nsurf = (m_nv > (3 + m_nspevib + m_nsp))
    ? (m_nv - (3 + m_nspevib + m_nsp)) : 0;
    if (nsurf > 0) {
        updateSurfaceState(y + 3 + m_nspevib + m_nsp);
    }

}

void PlasmaReactor::eval(double time, double* LHS, double* RHS)
{
    //printf(" **************************** Entering eval function ******************************************\n");

    ////////////////////////////////////////// chatgpt safety addition proposal to reinforce the function //////////////////// comment: why not but doesn't seem to change anything

    std::fill(LHS, LHS + m_nv, 0.0);
    std::fill(RHS, RHS + m_nv, 0.0);

    // Mass and volume are ODEs
    LHS[0] = 1.0;                   // dm/dt
    LHS[1] = 1.0;                   // dV/dt
    RHS[1] = m_vdot;

    //////////////////////////////////////////////////////////////


    double& dmdt = RHS[0]; // dm/dt (gas phase)
    double& mcvdTdt = RHS[2]; // m * c_v * dT/dt
    double* devibdt = RHS + 3; // devib/dt
    double* mdYdt = RHS + 3 + m_nspevib; // mass * dY/dt

    evalWalls(time);
    m_plasma->restoreState(m_state);
    m_plasma->getPartialMolarIntEnergies(&m_uk[0]);
    const vector<double>& mw = m_plasma->molecularWeights();
    const double* Y = m_plasma->massFractions();

    if (m_chem) {
        m_kin->getNetProductionRates(&m_wdot[0]); // "omega dot"
    }

    evalSurfaces(LHS + m_nsp + m_nspevib + 3, RHS + m_nsp + m_nspevib + 3, m_sdot.data());
    double mdot_surf = dot(m_sdot.begin(), m_sdot.end(), mw.begin());
    dmdt += mdot_surf;

    // compression work and external heat transfer
    mcvdTdt += - m_pressure * m_vdot + m_Qdot;

    // gas heating from the discharge
    compute_disVPower();

    double tot_vib_power = 0;

    if (m_nspevib > 0) {
        compute_disVibVPower();
        for (size_t n = 0; n < m_nspevib; n++){
            tot_vib_power += disVibVPower[n];
        }
    }

    mcvdTdt += (m_disVPower - tot_vib_power) * m_vol; // RAW FAST GAS HEATING POWER (BEFORE APPLYING PLASMA CHEMICAL SOURCE TERMS)

    // gas heating from vibrational–translational relaxation
    double tot_relax_power = 0;
    if (m_nspevib > 0) {
        compute_RvtVPower();
        
        for (size_t n = 0; n < m_nspevib; n++){
            tot_relax_power += RvtVPower[n];
        }
    }

    mcvdTdt += tot_relax_power * m_vol; // SLOW GAS HEATING POWER - currently always equal to zero with the vibraitonal energy model since th model is not yet implemented in this version.
    

    // printf(" Entering species chemical for loop\n");
    for (size_t n = 0; n < m_nsp; n++) {
        
        // heat release from gas phase and surface reactions
        mcvdTdt -= m_wdot[n] * m_uk[n] * m_vol;
        mcvdTdt -= m_sdot[n] * m_uk[n];
        // production in gas phase and from surfaces
        mdYdt[n] = (m_wdot[n] * m_vol + m_sdot[n]) * mw[n];
        // dilution by net surface mass flux
        mdYdt[n] -= Y[n] * mdot_surf;
        //Assign left-hand side of dYdt ODE as total mass
        LHS[n+3+m_nspevib] = m_mass;
    }
    for (size_t n=0; n < m_nspevib; n++){
        devibdt[n] = disVibVPower[n] - RvtVPower[n];
    }
    
    // Assign left-hand side of devibdt as one
    for (size_t n = 0; n < m_nspevib; n++){
        LHS[3+n] = 1;
    }
    // add terms for outlets
    for (auto outlet : m_outlet) {
        double mdot = outlet->massFlowRate();
        dmdt -= mdot; // mass flow out of system
        mcvdTdt -= mdot * m_pressure * m_vol / m_mass; // flow work
    }

    // add terms for inlets
    for (auto inlet : m_inlet) {
        double mdot = inlet->massFlowRate();
        dmdt += mdot; // mass flow into system
        mcvdTdt += inlet->enthalpy_mass() * mdot;
        for (size_t n = 0; n < m_nsp; n++) {
            double mdot_spec = inlet->outletSpeciesMassFlowRate(n);
            // flow of species into system and dilution by other species
            mdYdt[n] += mdot_spec - mdot * Y[n];

            // In combination with h_in*mdot_in, flow work plus thermal
            // energy carried with the species
            mcvdTdt -= m_uk[n] / mw[n] * mdot_spec;
        }
    }

    RHS[1] = m_vdot;
    if (m_energy) {
        LHS[2] = m_mass * m_plasma->cv_mass();
    } else {
        RHS[2] = 0;
    }
}

size_t PlasmaReactor::componentIndex(const string& nm) const
{
    size_t k = speciesIndex(nm);
    if (k != npos) {
        return k + 3 + m_nspevib;
    } else if (nm == "mass") {
        return 0;
    } else if (nm == "volume") {
        return 1;
    } else if (nm == "temperature") {
        return 2;
    } else if (nm == "evib") {
        return 3;
    } else {
        return npos;
    }
}

string PlasmaReactor::componentName(size_t k) {
    if (k == 2) {
        return "temperature";
    } else if (k == 0) {
        return "mass";
    } else if (k == 1) {
        return "volume";
    } else if (k >= 3 && k < 3 + m_nspevib) {
        return "evib";
    } else {
        return Reactor::componentName(k-m_nspevib);
    }
}

void PlasmaReactor::compute_disVPower() {
    //printf("Computing discharge power");
    if (m_plasma->electricField() < 1e-21){
        // If the electric field is too low, we assume no discharge power.
        m_disVPower = 0;
    }
    else{
        m_disVPower = ElectronCharge * m_plasma->nElectron()
            * m_plasma->electronMobility()
            * pow(m_plasma->electricField(), 2);
    }
}

// TO BE IMPLEMENTED WITH REAL VALUES LATER, FOR NOW RUNS JUST TO SHOW THE CODE STRCUTURE BUL ALL EVIB IS SET TO 0
void PlasmaReactor::compute_disVibVPower() { 
    //printf("Computing vibrational power");

    size_t n_vib_species = m_nspevib;
    
    for (size_t k = 0; k<n_vib_species; k++){
        disVibVPower[k] = 0;
    } 
}

// TO BE IMPLEMENTED WITH REAL VALUES LATER, FOR NOW RUNS JUST TO SHOW THE CODE STRCUTURE BUL ALL EVIB IS SET TO 0
void PlasmaReactor::compute_RvtVPower() {
    //printf("Computing vibrational relaxation power");
    size_t n_vib_species = m_nspevib;
    
    for (size_t n=0; n<n_vib_species; n++){
        RvtVPower[n] = 0;
    }

}

// This function is put with prints so that the user sees that the vibrational species are recovered and can be read properly. 
void  PlasmaReactor::recoverVibSpecies(){
    vib_spec = m_plasma->getVibSpecies();
    printf("Vibrational species recovery:\n");
    if (vib_spec.size() == 0){
        printf("    No vibrational species found\n");
    }
    for (size_t n = 0; n < vib_spec.size(); n++){
        printf("Vibrational species %ld: %s\n", n, vib_spec[n].c_str());
    }
}

}




