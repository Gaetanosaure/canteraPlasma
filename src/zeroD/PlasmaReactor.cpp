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
        //writelog("[DEBUFG] m_plasma->nElectron(): {}\n", m_plasma->nElectron());
        //writelog("[DEBUFG] m_plasma->electronMobility(): {}\n", m_plasma->electronMobility());
        //writelog("[DEBUFG] m_plasma->electricField(): {}\n", m_plasma->electricField());
        m_disVPower = ElectronCharge * m_plasma->nElectron()
            * m_plasma->electronMobility()
            * pow(m_plasma->electricField(), 2);
        //writelog("[DEBUFG] m_disVPower: {}\n", m_disVPower);
    }
}

// TO BE IMPLEMENTED WITH REAL VALUES LATER, FOR NOW RUNS JUST TO SHOW THE CODE STRCUTURE BUL ALL EVIB IS SET TO 0
void PlasmaReactor::compute_disVibVPower() { 
    //printf("Computing vibrational power");
    size_t n_vib_species = m_nspevib;

    m_kin->getNetRatesOfProgress(&m_kr[0]); // "kr"
    
    for (size_t k = 0; k<n_vib_species; k++){

        disVibVPower[k] = 0;
        string vib_spec_here = vib_spec[k];
        
        //// DO NOT UNCOMMENT FOR NOW. THE ARCHITECTURE IS READY BUT IS WILL PRODUCE A SEGFAULT 
        //// This is because for now the yaml architecture is undecided and as such the targets and the duvbib values are not loaded in the plasma phase and cannnot
        //// be retrieved here: getTarget and getDuvib will segfault.
        //// Be careful, once the yaml architecture is loaded, there might be two objects: electron collisions and reactions. 
        //// Please ensure that the loop currently made only on reactions will also engulf the collisions, otherwise the model will seem to work but produce wrong results.


        // for (size_t n = 0; n < m_kin->nReactions(); n++) {
        //     string reac_target_spec = m_plasma->getTarget(n);
        //     if (reac_target_spec == vib_spec_here) {
        //         double DUVibValue = m_plasma->getDuvib(n)*ElectronCharge; // convert to Joules
        //         disVibVPower[k] += DUVibValue * m_kr[n] * Avogadro; //multiply by the avogadro number to actually get a power
        //     }
             
        // }
    } 
    // writelog("[DEBUG] MARKER 4\n");
}



// TO BE IMPLEMENTED WITH REAL VALUES LATER, FOR NOW RUNS JUST TO SHOW THE CODE STRCUTURE BUL ALL EVIB IS SET TO 0
void PlasmaReactor::compute_RvtVPower() {
    //printf("Computing vibrational relaxation power");
    size_t n_vib_species = m_nspevib;

    double* evib_array = new double[n_vib_species];

    m_plasma->getVibrationalEnergies(evib_array);
    
    for (size_t n=0; n<n_vib_species; n++){
        RvtVPower[n] = 0;
        double tau = compute_TauRelax(n);
        RvtVPower[n] = evib_array[n]/tau;
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

double PlasmaReactor::compute_TauRelax(size_t n){
    
    double tau = 0;
    string spec_name = vib_spec[n];
    // printf("Computing relaxation time for species %s\n", spec_name.c_str());
    if (relax_type == "Millikan&White" || relax_type == "millikan&White" || relax_type == "Millikan&white" || relax_type == "millikan&white"){
        tau = tau_millikan_white(spec_name);
    }
    else if (relax_type == "Castela" || relax_type == "castela" ){
        tau = tau_castela(spec_name);
    }
    else if (relax_type == "Constant" || relax_type == "constant"){
        tau = tau_relax_constant_model;
    }
    else if (relax_type == "Starikovskiy" || relax_type == "starikovskiy"){
        tau = tau_starikovskiy(n);
    }
    else{
        throw CanteraError("PlasmaReactor::compute_TauRelax",
                           "Error: species vibrational relaxation type not implemented. Only Castela, Constant, Millikan&White and Starikovskiy models are implemented");
    }
    return tau;
}

double PlasmaReactor::tau_millikan_white(string spec_name){
    throw CanteraError("PlasmaReactor::compute_TauRelax",
                           "Error: Millikan&White relaxation's implementation is currently incomplete");
}

double PlasmaReactor::tau_castela(string spec_name){
    double tau = 1e-11; // Almost as fast gas heating if castela is called for a species which is not N2. In that case, the species will have a very short relaxation time.
    if (spec_name == "N2") {
        double a_n2 = 221.0;
        double b_n2 = 0.029;
        double a_o2 = 229.0;
        double b_o2 = 0.0295;
        double a_o = 72.4; 
        double b_o = 0.015;
        
        double c = 101325; //Pa.s

        double T = m_plasma->temperature();
        double P = m_plasma->pressure();
        double x_n2 = m_plasma->moleFraction("N2");
        double x_o2 = m_plasma->moleFraction("O2");
        double x_o = m_plasma->moleFraction("O");

        double p_n2 = Max(P * x_n2, 1e-16); // avoid division by zero
        double p_o2 = Max(P * x_o2, 1e-16);
        double p_o = Max(P * x_o, 1e-16);

        double tau_n2 = (exp(a_n2*(pow(T, -0.3333) - b_n2) - 18.42))*c/p_n2;
        double tau_o2 = (exp(a_o2*(pow(T, -0.3333) - b_o2) - 18.42))*c/p_o2;
        double tau_o = (exp(a_o*(pow(T, -0.3333) - b_o) - 18.42))*c/p_o;

        tau = 1/(1/tau_n2 + 1/tau_o2 + 1/tau_o);
        
    }
    printf("tau_castela for species %s = %e [s] \n", spec_name.c_str(), tau);
    return tau;
}

// Fonction pour calculer k(T) en cm3/s
double PlasmaReactor::compute_k(const RelaxationEntry& entry, double T) {
    return entry.A * std::pow(T, entry.n) * std::exp(
        entry.K - entry.B / std::pow(T, 1.0 / 3.0)
                 + entry.C / std::pow(T, entry.m)
                 + entry.D / std::pow(T, entry.z)
    );
}

void PlasmaReactor::readStarikovskiyRelaxYamlFile(string filename){
    // On retrouve le chemin complet à partir du nom de fichier
    std::string full_path;
    try {
        full_path = findInputFile(filename);  // cherche dans tous les chemins Cantera
    } catch (CanteraError& err) {
        throw CanteraError("PlasmaReactor::readStarikovskiyRelaxYamlFile",
            "Could not find the YAML file for Starikovskiy relaxation: {}\n"
            "File requested: {}\n", err.what(), filename);
    }

    YAML::Node root = YAML::LoadFile(full_path);

    for (size_t n=0; n<m_nspevib; n++){
        string spec_name = vib_spec[n];
        printf("Reading STARIKOVSKIY DATA YAML file for species %s\n", spec_name.c_str()); 
        std::string key = spec_name + "_relaxations";

        std::vector<RelaxationEntry> reactions;
        for (const auto& node : root[key]) {
            RelaxationEntry r;
            r.name = node["name"].as<std::string>();
            r.target = node["target"].as<std::string>();
            r.A = node["A"].as<double>();
            r.n = node["n"].as<double>();
            r.K = node["K"].as<double>();
            r.B = node["B"].as<double>();
            r.C = node["C"].as<double>();
            r.m = node["m"].as<double>();
            r.D = node["D"].as<double>();
            r.z = node["z"].as<double>();
            reactions.push_back(r);
        }
        m_data_starikovskiy.push_back(reactions);
    }
}


double PlasmaReactor::tau_starikovskiy(size_t n){

    if (!starikovskiy_read) {
        if (starikovskiy_yaml_path == "init"){
            starikovskiy_yaml_path = "plasma_relax/starikovskiy_default.yaml"; // default path
            printf("No yaml file provided for the Starikovskiy relaxation model but this model is used\n. Using default path %s\n", starikovskiy_yaml_path.c_str());
        }
        readStarikovskiyRelaxYamlFile(starikovskiy_yaml_path);
        starikovskiy_read = true;
    }

    double one_over_tau = 0;
    double T = m_plasma->temperature();
    double avogadro_per_mol = Avogadro/1000;

    // std::cout << "Reactions rates for target " << vib_spec[n] << " at T = " << T << " K:\n";
    for (const auto& r : m_data_starikovskiy[n]) {
        double k = 1e-6*compute_k(r, T); // convert to m3/s bc the result from compute_k is in cm3/s
        // std::cout << "  " << r.name << ": k = " << k << "\n";
        double x_partner = m_plasma->moleFraction(r.name); // loop over all the species and moleFraction returns 0 if the species is not in the phase
        double mixture_molar_density = 1000*m_plasma->molarDensity(); // in cantera, the density in kmol/m^3 so we need to convert to mol/m^3 to get things right
        one_over_tau += k * x_partner * mixture_molar_density * avogadro_per_mol;
    }
    double tau = 1/one_over_tau;
    std::cout << "Starikovskiy relaxation time for " << vib_spec[n] << ": " << tau << "[s]\n";

    return tau;
} 


//// USELESS FOR THE REACTOR ITSELF BUT USEFUL FOR THE PYTHON BININGS:

std::vector<double> PlasmaReactor::get_disVibVPower() {
    compute_disVibVPower();
    return disVibVPower;
}

std::vector<double> PlasmaReactor::get_RvtVPower() {
    compute_RvtVPower();
    return RvtVPower;
}

std::vector<double> PlasmaReactor::get_eVib() {
    size_t n_vib_species = m_nspevib;
    std::vector<double> to_return(n_vib_species);

    double* evib_array = new double[n_vib_species];
    m_plasma->getVibrationalEnergies(evib_array);

    for (size_t n = 0; n < n_vib_species; ++n) {
        to_return[n] = evib_array[n];
    }

    delete[] evib_array;
    return to_return;
}

void PlasmaReactor::setVibRelaxType(string relax_type_name){
    relax_type = relax_type_name;
    printf("Relaxation type set to %s\n", relax_type.c_str());}

string PlasmaReactor::getVibRelaxType(){
    return relax_type;
}

double PlasmaReactor::getVibConstantModelTauRelax(){
    return tau_relax_constant_model;
}

void PlasmaReactor::setVibConstantModelTauRelax(double tau_to_set){
    tau_relax_constant_model = tau_to_set;
    printf("Relaxation time constant model set to %f\n", tau_relax_constant_model);}

double PlasmaReactor::Max(double a, double b){
    if (a>b) {
        return a;
    } else{
        return b;
    }
    }

}




