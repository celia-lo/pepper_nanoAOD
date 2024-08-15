import pepper
import awkward as ak
import math
import logging
from functools import partial
from copy import copy
import numpy as np
import json
import correctionlib
from tools import neutrino_reco
from tools import top_reco
from tools import cut_defs
from config_topgamma import ConfigTopGamma
from tools.utils import DeltaR
from pepper.scale_factors import (TopPtWeigter, PileupWeighter, BTagWeighter,
                                  get_evaluator, ScaleFactors)

logger = logging.getLogger(__name__)

# All processors should inherit from pepper.ProcessorBasicPhysics
class Processor(pepper.ProcessorBasicPhysics):
    # We use the ConfigTTbarLL instead of its base Config, to use some of its
    # predefined extras
    config_class = ConfigTopGamma

    def __init__(self, config, eventdir):
        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Can set and modify configuration here as well
        # Need to call parent init to make histograms and such ready

        config["histogram_format"] = "root"

        super().__init__(config, eventdir)

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps: add cuts, define objects and/or
        # compute event weights

        # Add a cut only allowing events according to the golden JSON
        # The good_lumimask method is specified in pepper.ProcessorBasicPhysics
        # It also requires a lumimask to be specified in config
        era = self.get_era(selector.data, is_mc)

        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if "top_pt_reweighting" in self.config:
                selector.add_cut(
                    "Top pt reweighting", self.do_top_pt_reweighting,
                    no_callback=True)

        if is_mc:
            selector.set_column("GenLepton", partial(self.build_genlepton_column, is_mc))
            selector.set_column("GenPhoton", partial(self.build_genphoton_column, is_mc))
            selector.set_column("GenTop", self.gentop)
            selector.set_column("GenTopPos", top_reco.gentoppos)
            selector.set_column("GenTopNeg", top_reco.gentopneg)

        if not is_mc:
            selector.add_cut("Lumi", partial(
                self.good_lumimask, is_mc, dsname))

        if not is_mc:
                selector.add_cut("JetVetoMapFilter", partial(self.FilterWithJetVetoMap, era))

        # apply MET filter
        selector.add_cut("MET filters", partial(self.met_filters, is_mc))

        # apply the number of good PV is at least 1 
        selector.add_cut("atLeastOnePV", self.add_PV_cut)

        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("Pileup reweighting", partial(
                self.do_pileup_reweighting, dsname))

        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],self.config["dataset_trigger_order"])
        #print("Trigger ",pos_triggers,neg_triggers)
        #selector.add_cut("Trigger", partial(self.passing_trigger, pos_triggers, neg_triggers))
    
        if is_mc and self.config["year"] in ("2016", "2017", "ul2016pre","ul2016post", "ul2017"):
            selector.add_cut("L1 prefiring", self.add_l1_prefiring_weights)

        # Pick electrons satisfying our criterias
        selector.set_column("Electron",self.pick_electrons)
        # Pick muons satisfying our criterias
        selector.set_column("Muon",self.pick_muons)

        # Combine electron and muon to lepton
        selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng))
        selector.add_cut("OneLep",self.one_lepton)

        # Define lepton categories, the number of lepton cut applied here
        selector.set_cat("channel",{"ele", "muon"})
        selector.set_multiple_columns(self.lepton_categories)

        selector.add_cut("pass_trig_muon",partial(self.passing_hlt,self.config['trigger_muon_path']),categories={"channel": ["muon"]})
        selector.add_cut("pass_trig_ele",partial(self.passing_hlt,self.config['trigger_ele_path']),categories={"channel": ["ele"]})

        selector.add_cut("exact_one_muon",self.exact_one_muon,categories={"channel": ["muon"]})
        selector.add_cut("exact_one_ele",self.exact_one_ele,categories={"channel": ["ele"]})
  
        selector.set_column("n_loose_muon",self.num_loose_muon)       
        selector.set_column("n_tight_muon",self.num_tight_muon)       
        selector.set_column("n_fake_muon",self.num_fake_muon)       
        selector.set_column("n_loose_ele",self.num_loose_ele)       
        selector.set_column("n_tight_ele",self.num_tight_ele)       
        selector.set_column("n_fake_ele",self.num_fake_ele)       

        selector.add_cut("muon_sf",partial(self.apply_muon_sf, is_mc))
        selector.add_cut("electron_sf",partial(self.apply_electron_sf, is_mc))

        selector.set_column("Lepton_charge", self.lepton_charge)

        # JME unc
        if (is_mc and self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
            for variarg in self.get_jetmet_variation_args():
                selector_copy = copy(selector)
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc,
                                                variarg, dsname, filler, era)
                if self.eventdir is not None:
                    logger.debug(f"Saving per event info for variation"
                                 f" {variarg.name}")
                    self.save_per_event_info(
                        dsname + "_" + variarg.name, selector_copy, False)
            filler.sys_overwrite = None

        #JME selection -> remove for photon category
        self.process_selection_jet_part(selector, is_mc,
                                        self.get_jetmet_nominal_arg(),
                                        dsname, filler, era)
        logger.debug("Selection done")

    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):

        # Pick Jets satisfying our criterias
        logger.debug(f"Running jet_part with variation {variation.name}")
        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        # comput jetfac from jer 
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, reapply_jec, variation.junc,
            variation.jer, selector.rng))

        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))
        # if "jet_puid_sf" in self.config and is_mc:
        #     selector.add_cut("JetPUIdSFs", partial(self.compute_jet_puid_sfs,is_mc))
        #selector.set_column("Jet", self.jets_with_puid)
        selector.set_column("CentralJet", self.build_centraljet_column)
        selector.set_column("ForwardJet", self.build_forwardjet_column)

        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))

        selector.set_column("mTW",self.build_mtw_column)
        selector.set_column("bJet", self.build_bjet_column)
        selector.set_column("loose_bJet", self.build_loose_bjet_column)
        selector.set_column("nbtag", self.num_btags)
        selector.set_column("n_loose_btag", self.num_loose_btags)
        selector.set_column("ncentral_jets", self.num_centralJets)
        selector.set_column("nforward_jets", self.num_forwardJets)

        selector.add_cut("preselection", self.dummycut)

        # Only accept events that have at least two jets and one bjet
        selector.add_cut("atLeast1jet",self.has_jets) # one jet
        selector.set_column("deltaR_lj",self.build_deltaR_ljet)
        #selector.add_cut("HasBtags", partial(self.btag_cut, is_mc))

        #Build different categories according to the number of jets
        #selector.set_cat("jet_btag", {"j2+_b0", "j2+_b1","j2+_b2","j2+_b3+","j2+_b1+"})
        #selector.set_multiple_columns(self.btag_categories)
        #selector.add_cut("btag_sf",partial(self.apply_btag_sf, is_mc))

        # Only accept events with MET pt more than 20 GeV
        selector.add_cut("Req MET", self.met_requirement)
        selector.add_cut("mTW_cut", self.mTW_requirement)

        selector.add_cut("dummpy", self.dummycut)

        # Only accept MC events with prompt lepton 
        if is_mc:
           selector.add_cut("isPromptLepton", self.isprompt_lepton)
           selector.set_column("Lepton_isPrompt", self.build_lepton_prompt)

        selector.add_cut("finalselection", self.dummycut)

	#ttbar semileptnic reconstruction
        #selector.set_column("Neutrino1",neutrino_reco.neutrino_reco)          
        #selector.set_column("reco_W",self.build_W_column)

        #topVars = top_reco.topreco(selector.data)
        #selector.set_column("tophad_m", topVars["mtophad"])           
        #selector.set_column("tophad_pt", topVars["pttophad"])
        #selector.set_column("tophad_eta", topVars["etatophad"])
        #selector.set_column("tophad_phi", topVars["phitophad"])

        #selector.set_column("toplep_m",   topVars["mtoplep"])
        #selector.set_column("toplep_pt",  topVars["pttoplep"])
        #selector.set_column("toplep_eta", topVars["etatoplep"])
        #selector.set_column("toplep_phi", topVars["phitoplep"])

        #selector.set_column("chi2",topVars["chisquare"])       

    def FilterWithJetVetoMap(self, era, data):
        veto_muons = data["Muon"]
        veto_muons = veto_muons[(veto_muons["isPFcand"] > 0)]

        veto_jets = data["Jet"]
        epsilon = 1e-10
        PICK = (veto_jets["pt"] > 15) & (veto_jets["neEmEF"] + veto_jets["chEmEF"] < 0.9) & (np.abs(veto_jets["phi"]) < math.pi + epsilon)
        veto_jets = veto_jets[PICK]
        veto_jets_, veto_muons_ = ak.unzip(ak.cartesian([veto_jets, veto_muons], nested=True))
        deltar = veto_jets_.delta_r(veto_muons_)
        CLEAN = ~ak.any(deltar < 0.4, axis=2)
        CLEAN = CLEAN | (ak.num(deltar, axis=2) == 0)
        veto_jets = veto_jets[CLEAN]

        dir_jetvetomap = self.config["jetvetomap"]
        dataset_to_map = {
            "2022C": "Summer22_23Sep2023.json",
            "2022D": "Summer22_23Sep2023.json",
            "2022E": "Summer22EE_23Sep2023.json",
            "2022F": "Summer22EE_23Sep2023.json",
            "2022G": "Summer22EE_23Sep2023.json"
        }
        
        map = dataset_to_map[era]
        with open(f"{dir_jetvetomap}/{map}") as rf:
            cname = json.load(rf)["corrections"][0]["name"]
        ceval = correctionlib.CorrectionSet.from_file(f"{dir_jetvetomap}/{map}")
        eta_flat = ak.flatten(veto_jets["eta"])
        phi_flat = ak.flatten(veto_jets["phi"])
        veto_map_flat = ceval[cname].evaluate("jetvetomap", eta_flat, phi_flat) > 0

        # Unflatten the veto map back to the original structure
        veto_map = ak.unflatten(veto_map_flat, ak.num(veto_jets["eta"]))
    
        # Create mask for events passing the veto
        MASK = ~ak.any(veto_map, axis=1)
        MASK = MASK | (ak.num(veto_map, axis=1) == 0)

        return MASK

    def add_PV_cut(self,data):
       PV = data["PV"]
       oneGoodPV = PV.npvsGood >0
       return oneGoodPV

    def pick_electrons(self, data):
        ele = data["Electron"]

        # We do not want electrons that are between the barrel and the end cap
        # For this, we need the eta of the electron with respect to its
        # supercluster
        sc_eta_abs = abs(ele.eta + ele.deltaEtaSC)
        is_in_transreg = (1.4442 < sc_eta_abs) & (sc_eta_abs < 1.566)
        impact = ( (sc_eta_abs<1.4442) & (abs(ele.dz) < 0.1) & (abs(ele.dxy) < 0.05) ) | ( (sc_eta_abs>1.566) & (abs(ele.dz) < 0.2) & (abs(ele.dxy) < 0.1) )

        # Electron ID, as an example we use the MVA one here
#        has_id = ele.mvaFall17V2Iso_WP90
        has_id = ele.cutBased >= 3

        # Finally combine all the requirements
        is_good = (
            has_id
            & impact
            & (~is_in_transreg)
            & (self.config["ele_eta_min"] < ele.eta)
            & (ele.eta < self.config["ele_eta_max"])
            & (self.config["good_ele_pt_min"] < ele.pt))
 
        veto_id = ele.cutBased >=1

        is_veto = (
                veto_id
              & impact 
              & (~is_in_transreg)
              & (self.config["ele_eta_min"] < ele.eta)
              & (ele.eta < self.config["ele_eta_max"])
              & (self.config["good_ele_pt_min"] < ele.pt))

        #is_fake = ( (veto_id & ~has_id)
        #          & impact 
        #          & (~is_in_transreg)   
        #          & (self.config["ele_eta_min"] < ele.eta) 
        #          & (ele.eta < self.config["ele_eta_max"])
        #          & (self.config["good_ele_pt_min"] < ele.pt)
        #          )

        is_fake = is_veto & ~is_good
        ele["is_good"] = is_good
        ele["is_fake"] = is_fake

        # Return all electrons with are deemed to be good
        return ele[is_veto]

    def pick_muons(self, data):
        muon = data["Muon"]
        etacuts = (self.config["muon_eta_min"] < muon.eta) & (muon.eta < self.config["muon_eta_max"])

        good_id = muon.tightId
        good_iso = muon.pfIsoId > 3
        is_good = (
            good_id
            & good_iso
            & etacuts
            & (self.config["good_muon_pt_min"] < muon.pt))
        
        veto_id = muon.looseId
        veto_iso = muon.pfIsoId >= 1
        is_veto = (
            veto_id
            & veto_iso
            & etacuts
            & (self.config["good_muon_pt_min"] < muon.pt))

        #fake_iso = (muon.pfIsoId >= 1) & (muon.pfIsoId <= 3)
        #is_fake = ( good_id 
        #          & fake_iso 
        #          & etacuts 
        #          & (self.config["good_muon_pt_min"] < muon.pt) )

        is_fake = is_veto & ~is_good
        muon["is_good"] = is_good
        muon["is_fake"] = is_fake

        return muon[is_veto]

    def build_lepton_column(self, is_mc, rng, data):
        """Build a lepton column containing electrons and muons."""
        electron = data["Electron"]
        muon = data["Muon"]
        # Apply Rochester corrections to muons
        #print(muon.pt)
        if "muon_rochester" in self.config:
            muon = self.apply_rochester_corr(muon, rng, is_mc)
            #print("after correction:",muon.pt)
        columns = ["pt", "eta", "phi", "mass", "pdgId","charge","is_good","is_fake"]
        if is_mc:
           columns.append("genPartIdx")
        lepton = {}
        for column in columns:
            lepton[column] = ak.concatenate([electron[column], muon[column]],
                                            axis=1)
        lepton = ak.zip(lepton, with_name="PtEtaPhiMLorentzVector",
                        behavior=data.behavior)

        # Sort leptons by pt
        # Also workaround for awkward bug using ak.values_astype
        # https://github.com/scikit-hep/awkward-1.0/issues/1288
        lepton = lepton[
            ak.values_astype(ak.argsort(lepton["pt"], ascending=False), int)]
        return lepton

    def pick_loose_photons(self, data):
        photon = data["Photon"]
        leptons = data["Lepton"]
        has_id = photon.cutBased>=2 # medium ID
        pass_psv = (photon.pixelSeed==False)

        is_in_EBorEE = (photon.isScEtaEB | photon.isScEtaEE)
        is_in_transreg = ( (1.4442 < abs(photon["eta"])) & (abs(photon["eta"]) < 1.566) )

        etacuts = (abs(photon["eta"])<2.5)
        ptcuts = (photon.pt>20)

        has_lepton_close = ak.any(
            photon.metric_table(leptons) < 0.4, axis=2)

        bitMap = photon["vidNestedWPBitmap"]

        passHoverE = (bitMap>>4&3)  >= 2
        passSIEIE = (bitMap>>6&3)  >= 2
        passChIso = (bitMap>>8&3)  >= 2
        passNeuIso = (bitMap>>10&3) >= 2
        passPhoIso = (bitMap>>12&3) >= 2

        isRelMediumPhoton = ((passHoverE) & (passNeuIso) & (passPhoIso))


        is_good = (
                isRelMediumPhoton
              & ~is_in_transreg
              & (is_in_EBorEE)
              & etacuts
              & pass_psv
              & (~has_lepton_close)
              & ptcuts)

        return photon[is_good]

    def pick_medium_photons(self, data):
        photon = data["Photon"]
        leptons = data["Lepton"]
        has_id = photon.cutBased>=2 # medium ID
        pass_psv = (photon.pixelSeed==False)

        is_in_EBorEE = (photon.isScEtaEB | photon.isScEtaEE)
    
        etacuts = (abs(photon["eta"])<2.5)
        ptcuts = (photon.pt>20)

        has_lepton_close = ak.any(
            photon.metric_table(leptons) < 0.4, axis=2)

        is_good = (
                has_id
	      & (~has_lepton_close)
              & pass_psv
              & etacuts
              & is_in_EBorEE
              & ptcuts)

        return photon[is_good]

    def good_jet(self, data):
        """Apply some basic jet quality cuts."""
        jets = data["Jet"]
        leptons = data["Lepton"]
        photons = data["Photon"]

        #j_id, j_puId, lep_dist, pho_dist, eta_min, eta_max, pt_min = self.config[[
        #    "good_jet_id", "good_jet_puId", "good_jet_lepton_distance", "good_jet_photon_distance",
        #    "good_jet_eta_min", "good_jet_eta_max", "good_jet_pt_min"]]
        
        j_id, lep_dist, pho_dist, eta_min, eta_max, pt_min = self.config[[
            "good_jet_id", "good_jet_lepton_distance", "good_jet_photon_distance",
            "good_jet_eta_min", "good_jet_eta_max", "good_jet_pt_min"]]

        if j_id == "skip":
            has_id = True
        elif j_id == "cut:loose":
            has_id = jets.isLoose
            # Always False in 2017 and 2018
        elif j_id == "cut:tight":
            has_id = jets.isTight
        elif j_id == "cut:tightlepveto":
            has_id = jets.isTightLeptonVeto
        else:
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_id))

        j_pt = jets.pt
        if "jetfac" in ak.fields(data):
            j_pt = j_pt * data["jetfac"]

        has_lepton_close = ak.any(
            jets.metric_table(leptons) < lep_dist, axis=2)
        has_photon_close = ak.any(
            jets.metric_table(photons) < pho_dist, axis=2)     

        return ( (has_id) 
                & (j_pt > pt_min)
                & (eta_min < jets.eta)
                & (jets.eta < eta_max)
                & (~has_lepton_close)
                #& (~has_photon_close)
                )

    def build_jet_column(self, is_mc, data):
        """Build a column of jets passing the jet quality cuts,
           including a 'btag' key (containing the value of the
           chosen btag discriminator) and a 'btagged' key
           (to select jets that are tagged as b-jets)."""
        is_good_jet = self.good_jet(data)
        #print("is_good_jet")
        #print(is_good_jet)
        #print("jet_pt")
        #print(data["Jet"].pt)
        jets = data["Jet"][is_good_jet]
        if "jetfac" in ak.fields(data):
            jets["pt"] = jets["pt"] * data["jetfac"][is_good_jet]
            jets["mass"] = jets["mass"] * data["jetfac"][is_good_jet]
            jets = jets[ak.argsort(jets["pt"], ascending=False)]

        # Evaluate b-tagging
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            jets["btag"] = jets["btagDeepB"]
        elif tagger == "deepjet":
            jets["btag"] = jets["btagDeepFlavB"]
        else:
            raise pepper.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = pepper.scale_factors.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        jets["btagged"] = jets["btag"] > getattr(wptuple, wp)
        #jets["pass_pu_id"] = self.has_puid(jets)
        #if is_mc:
        #    # A jet is considered to be a pileup jet if there is no gen jet
        #    # within Delta R < 0.4
        #    jets["has_gen_jet"] = ak.fill_none(
        #        jets.delta_r(jets.matched_gen) < 0.4, False)
        return jets
    
    def build_centraljet_column(self,data):
        jets = data["Jet"]
        central_region = (abs(jets.eta)<2.5)
        centraljets = jets[central_region]
 
        return centraljets

    def build_forwardjet_column(self,data):
        jets = data["Jet"]
        forward_region = (abs(jets.eta)>2.5)
        forwardjets = jets[forward_region]
 
        return forwardjets

    def build_bjet_column(self,data):
        jets = data["CentralJet"]
        bjets = jets[data["CentralJet"].btagged]

        return bjets

    def build_loose_bjet_column(self,data):
        jets = data["CentralJet"]
        tagger, wp = self.config["loose_btag"].split(":")
        if tagger == "deepcsv":
            jets["loose_btag"] = jets["btagDeepB"]
        elif tagger == "deepjet":
            jets["loose_btag"] = jets["btagDeepFlavB"]
        else:
            raise pepper.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = pepper.scale_factors.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        accept = jets["loose_btag"] > getattr(wptuple, wp)
        jets["loose_btagged"] = accept
        loose_bjets = jets[accept]

        return loose_bjets

    def build_W_column(self,data):
        nu = data["Neutrino1"]
        lep = data["Lepton"]
        wpt = (nu[:,0]+lep[:,0]).pt
        weta = (nu[:,0]+lep[:,0]).eta
        wphi = (nu[:,0]+lep[:,0]).phi
        wmass = (nu[:,0]+lep[:,0]).mass
        wp = wpt*np.cosh(weta)
        wenergy = np.sqrt(wmass*wmass+wp*wp)
      
        w = ak.zip({"pt": wpt, "eta": weta , "phi": wphi, "energy": wenergy},
                   with_name="PtEtaPhiELorentzVector",behavior=data.behavior) 
        #print(w.pt,w.eta,w.energy)
        return w

    def isprompt_lepton(self, data):
        promptmatch = self.lepton_isprompt(data)
        n_prompt_lep = ak.sum(promptmatch,axis=1)
        accept = n_prompt_lep > 0        

        return accept

    def one_good_photon(self,data):
        return ak.num(data["Photon"])>0
   
    def num_btags(self, data):
        return ak.num(data['bJet'])

    def num_loose_btags(self, data):
        return ak.num(data['loose_bJet'])

    def num_forwardJets(self, data):
        return ak.num(data['ForwardJet'])

    def num_centralJets(self, data):
        return ak.num(data['CentralJet'])

    def photon_categories_data(self,data):
        leptons = data["Lepton"]

        cats = {}
        cats["allphoton"] = (ak.num(leptons)>0)
        return cats   
       
    def photon_categories(self,data):
        cats = {}
        photons = data["Photon"]
        genpart = data["GenPart"]

        # photons matched to a gen Pho
        true_photons = photons[ak.fill_none(abs(photons.matched_gen.pdgId) == 22, False)]
        # photons matched to a gen Ele
        electron_matched = photons[ak.fill_none(abs(photons.matched_gen.pdgId) == 11, False)]
        # photons can't matched to any gen Obj
        unmatched_photons = photons[ak.is_none(photons.matched_gen,axis=1)]

        promptmatch = true_photons.matched_gen.hasFlags(['isPrompt'])
        #promptmatch = ( (promptmatch) | (true_photons.matched_gen.hasFlags(['isDirectPromptTauDecayProduct'])) |
        promptmatch = ( (promptmatch) | (true_photons.matched_gen.hasFlags(['isPromptTauDecayProduct'])) |
                        (true_photons.matched_gen.hasFlags(["fromHardProcess"])))
       
        #DR between gen and reco photon
        dr_reco_gen = ak.any(true_photons.matched_gen.metric_table(true_photons) < 0.3, axis=2)

        promptmatch = promptmatch & (dr_reco_gen)

        prompt_photons = true_photons[promptmatch] 
        nonprompt_photons = true_photons[~(promptmatch)]

        prompt_photons_event = (ak.num(prompt_photons)>0)
        ele_matched_event = ( (ak.num(prompt_photons)==0) & (ak.num(electron_matched)>0) ) 
        nonprompt_event = ( (ak.num(prompt_photons)==0) & ((ak.num(nonprompt_photons)>0) | (ak.num(unmatched_photons)>0)) )

        cats["prompt"] = ak.fill_none(prompt_photons_event,False,axis=1) 
        cats["ele_matched"] = ak.fill_none(ele_matched_event,False,axis=1)
        cats["nonprompt"] = ak.fill_none(nonprompt_event,False,axis=1)
        cats["allphoton"] = ak.num(photons)>0
        
        return cats

    def one_lepton(self,data):
        return (ak.num(data["Lepton"])>0)

    def exact_one_ele(self,data):
        nele = ak.num(data['Electron'])
        nmuon = ak.num(data['Muon'])
        accept = (nele==1) & (nmuon==0)

        return accept

    def exact_one_muon(self,data):
        nele = ak.num(data['Electron'])
        nmuon = ak.num(data['Muon'])
        accept = (nmuon==1) & (nele==0)

        return accept

    def lepton_is_tight(self,data):
        is_good = data["Lepton"]["is_good"]

        return is_good

    def num_loose_muon(self,data):
        loose_mu = data['Muon']
        n_loose_mu = ak.num(loose_mu) 
      
        return n_loose_mu

    def num_loose_ele(self,data):
        loose_ele = data['Electron']
        n_loose_ele = ak.num(loose_ele) 
      
        return n_loose_ele

    def num_tight_muon(self,data):
        tight_muon = data['Muon']['is_good']
        n_tight_muon = ak.sum(tight_muon,axis=1) 
      
        #print('is_good',n_tight_muon)

        return n_tight_muon

    def num_tight_ele(self,data):
        tight_ele = data['Electron']['is_good']
        n_tight_ele = ak.sum(tight_ele,axis=1) 
      
        return n_tight_ele

    def num_fake_muon(self,data):
        fake_muon = data['Muon']['is_fake']
        n_fake_muon = ak.sum(fake_muon,axis=1) 
      
        #print('is_fake',n_fake_muon)
        return n_fake_muon

    def num_fake_ele(self,data):
        fake_ele = data['Electron']['is_fake']
        n_fake_ele = ak.sum(fake_ele,axis=1) 
      
        return n_fake_ele

    def lepton_categories(self,data):
        cat = {}
        leps = data["Lepton"]
        cat['ele']  =  (abs(leps[:, 0].pdgId) == 11)
        cat['muon'] =  (abs(leps[:, 0].pdgId) == 13)

        return cat

    def passing_hlt(self,trigger_path,data):
        hlt = data["HLT"]
        triggered = np.full(len(data), False)

        if "16" not in self.config["year"]:
           triggered |= np.asarray(hlt[trigger_path])
        else:
           for trig_p in trigger_path:
               triggered = triggered | np.asarray(hlt[trig_p])

        return triggered

    def btag_cut(self,is_mc,data):
        num_btagged = data["nbtag"]
        num_loose_btagged = data["n_loose_btag"]
        accept = np.asarray(num_btagged >= self.config["num_atleast_btagged"])
        if is_mc and (
                "btag_sf" in self.config and len(self.config["btag_sf"]) != 0):
            weight, systematics = self.compute_weight_btag(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

        return (num_btagged >= self.config["num_atleast_btagged"])

    def btag_categories(self,data):
        cats = {}
        
        num_btagged = data["nbtag"]
        #num_btagged = data["n_loose_btag"]
        njet = ak.num(data["Jet"])

        cats["j2+_b0"] = (num_btagged == 0) & (njet >= 2)
        cats["j2+_b1"] = (num_btagged == 1) & (njet >= 2)
        cats["j2+_b2"] = (num_btagged == 2) & (njet >= 2)
        cats["j2+_b3+"] = (num_btagged >= 3) & (njet >= 2)
        cats["j2+_b1+"] = (num_btagged >= 1) & (njet >= 2)

        return cats

    def met_requirement(self, data):
        met = data["MET"].pt
        return met < 20

    def mTW_requirement(self, data):
        mTW = data["mTW"]
        return mTW < 20

    def mass_lg(self, data):
        """Return invariant mass of lepton plus photon"""
        return (data["Lepton"][:, 0] + data["Photon"][:, 0]).mass

    def build_lepton_prompt(self,data):
        lepton = data["Lepton"]
        prompt = self.lepton_isprompt(data)
        return ak.sum(prompt,axis=1)>0 

    def lepton_charge(self, data):
        charge = data["Lepton"].charge
        return charge 

    def z_cut(self,data):
        is_out_window = abs(data['mlg'] - 91.2) > 10
        return is_out_window    

    def lepton_isprompt(self,data):
        lepton = data["Lepton"]
        genpart = data["GenPart"]

        genmatchID = lepton.genPartIdx[(lepton.genPartIdx!=-1)]
        matched_genlepton = genpart[genmatchID] 
        promptmatch =  matched_genlepton.hasFlags(['isPrompt'])
        promptmatch = ( (promptmatch) | ( matched_genlepton.hasFlags(['isPromptTauDecayProduct'])) |
                        ( matched_genlepton.hasFlags(["fromHardProcess"])))
       
        return promptmatch

    def compute_electron_sf(self, data):
        # Electron reconstruction and identification
        eles = data["Electron"]
        weight = np.ones(len(data))
        systematics = {}
        # Electron identification efficiency
        for i, sffunc in enumerate(self.config["electron_sf"]):
            sceta = eles.eta + eles.deltaEtaSC
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(sceta)
                elif dimlabel == "eta":
                    params["eta"] = sceta
                else:
                    params[dimlabel] = getattr(eles, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = "electronsf{}".format(i)
            if self.config["compute_systematics"]:
                up = ak.prod(sffunc(**params, variation="up"), axis=1)
                down = ak.prod(sffunc(**params, variation="down"), axis=1)
                systematics[key] = (up / central, down / central)
            weight = weight * central

        return weight, systematics

    def compute_muon_sf(self, data):
        # Muon identification and isolation efficiency
        muons = data["Muon"]
        weight = np.ones(len(data))
        systematics = {}
        for i, sffunc in enumerate(self.config["muon_sf"]):
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(muons.eta)
                else:
                    params[dimlabel] = getattr(muons, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = f"muonsf{i}"
            if self.config["compute_systematics"]:
                if ("split_muon_uncertainty" not in self.config
                        or not self.config["split_muon_uncertainty"]):
                    unctypes = ("",)
                else:
                    unctypes = ("stat ", "syst ")
                for unctype in unctypes:
                    up = ak.prod(sffunc(
                        **params, variation=f"{unctype}up"), axis=1)
                    down = ak.prod(sffunc(
                        **params, variation=f"{unctype}down"), axis=1)
                    systematics[key + unctype.replace(" ", "")] = (
                        up / central, down / central)
            weight = weight * central
        return weight, systematics

    def compute_photon_sf(self, data):
        # Photon identification+PSV SFs
        phos = data["Photon"]
        weight = np.ones(len(data))
        systematics = {}
        for i, sffunc in enumerate(self.config["photon_sf"]):
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(phos.eta)
                else:
                    params[dimlabel] = getattr(phos, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = "photonsf{}".format(i)
            if self.config["compute_systematics"]:
                up = ak.prod(sffunc(**params, variation="up"), axis=1)
                down = ak.prod(sffunc(**params, variation="down"), axis=1)
                systematics[key] = (up / central, down / central)

            weight = weight * central
        return weight, systematics

    def compute_psv_sf(self, data):
        phos = data["Photon"]
        phos["etabin"] = ak.where(abs(phos.eta) < 1.5, 0.5, 3.5)
        weight = np.ones(len(data))
        systematics = {}
        # Photon identification+PSV SFs
        for i, sffunc in enumerate(self.config['psv_sf']):
            central = ak.prod(sffunc(etabin=phos["etabin"]),axis=1)
            key = "PSVsf{}".format(i)
            if self.config["compute_systematics"]:
                up = ak.prod(sffunc(
                    etabin=phos["etabin"], variation="up"),axis=1)
                down = ak.prod(sffunc(
                    etabin=phos["etabin"], variation="down"),axis=1)
                systematics[key] = (up / central, down / central)
            weight = weight * central

        return weight, systematics

    def compute_jet_puid_sfs(self,is_mc,data):
        jets = data["Jet"][(data["Jet"].pt < 50) & (data["Jet"].has_gen_jet)]
        weight = np.ones(len(data))
        pass_puid = jets["pass_pu_id"]
        systematics = {}
        sf_eff = []
        for i, sffunc in enumerate(self.config["jet_puid_sf"]):
            params = {}
            #print(sffunc)
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(jets.eta)
                else:
                    params[dimlabel] = getattr(jets, dimlabel)
            sf_eff.append(sffunc(pt=jets.pt,eta=jets.eta))
        prob_mc = ak.prod(ak.where(pass_puid,sf_eff[0],1-sf_eff[0]),axis=1)
        prob_data = ak.prod(ak.where(pass_puid,sf_eff[0]*sf_eff[1],1-sf_eff[0]*sf_eff[1]),axis=1)

        if is_mc:
           central = prob_data/prob_mc
        else:
           central = np.ones(len(data))
        if self.config["compute_systematics"] and  len(self.config["jet_puid_sf_unc"])>0 and is_mc:
           for i, sffunc in enumerate(self.config["jet_puid_sf_unc"]):
               #print(sffunc)
               up = sf_eff[1]+sffunc(pt=jets.pt,eta=jets.eta)
               down = sf_eff[1]-sffunc(pt=jets.pt,eta=jets.eta)
               prob_data_up = ak.prod(ak.where(pass_puid,sf_eff[0]*up,1-sf_eff[0]*up),axis=1)
               prob_data_down = ak.prod(ak.where(pass_puid,sf_eff[0]*down,1-sf_eff[0]*down),axis=1)
               up = prob_data_up/prob_mc
               down = prob_data_down/prob_mc
               key="puid_sf{}".format(i)
               systematics[key] = (up / central, down / central)

        weight = weight * central

        return  weight,systematics

    def apply_electron_sf(self,is_mc,data):
        if is_mc and ("electron_sf" in self.config
                       and len(self.config["electron_sf"]) > 0):
           weight, systematics = self.compute_electron_sf(data)
           return weight, systematics
        else:
           return np.ones(len(data))
       
    def apply_muon_sf(self,is_mc,data):
        if is_mc and ("muon_sf" in self.config
                       and len(self.config["muon_sf"]) > 0):
           weight, systematics = self.compute_muon_sf(data)
           return weight, systematics
        else:
           return np.ones(len(data))
       
    def apply_psv_sf(self,is_mc,data):
        if is_mc and ("psv_sf" in self.config
                       and len(self.config["psv_sf"]) > 0):
           weight, systematics = self.compute_psv_sf(data)
           return weight, systematics
        else:
           return np.ones(len(data))

    def apply_photon_sf(self,is_mc,data):
        if is_mc and ("photon_sf" in self.config
                       and len(self.config["photon_sf"]) > 0):
           weight, systematics = self.compute_photon_sf(data)
           return weight, systematics
        else:
           return np.ones(len(data))

    def apply_btag_sf(self, is_mc, data):
        """Apply btag scale factors."""
        if is_mc and (
                "btag_sf" in self.config and len(self.config["btag_sf"]) != 0):
            weight, systematics = self.compute_weight_btag(data)
            return weight, systematics
        else:
            return np.ones(len(data))


    def compute_weight_btag(self, data, efficiency="central", never_sys=False):
        """Compute event weights and systematics, if requested, for the b
        tagging"""
        jets = data["CentralJet"]
        wp = self.config["btag"].split(":", 1)[1]
        flav = jets["hadronFlavour"]
        eta = abs(jets.eta)
        pt = jets.pt
        discr = jets["btag"]
        weight = np.ones(len(data))
        systematics = {}
        for i, weighter in enumerate(self.config["btag_sf"]):
            central = weighter(wp, flav, eta, pt, discr, "central", efficiency)
            if not never_sys and self.config["compute_systematics"]:
                if "btag_splitting_scheme" in self.config:
                    scheme = self.config["btag_splitting_scheme"].lower()
                elif ("split_btag_year_corr" in self.config and
                        self.config["split_btag_year_corr"]):
                    scheme = "years"
                else:
                    scheme = None
                if scheme is None:
                    light_unc_splits = heavy_unc_splits = {"": ""}
                elif scheme == "years":
                    light_unc_splits = heavy_unc_splits = \
                        {"corr": "_correlated", "uncorr": "_uncorrelated"}
                elif scheme == "sources":
                    heavy_unc_splits = {name: f"_{name}"
                                        for name in weighter.sources}
                    light_unc_splits = {"corr": "_correlated",
                                        "uncorr": "_uncorrelated"}
                else:
                    raise ValueError(
                        f"Invalid btag uncertainty scheme {scheme}")

                for name, split in heavy_unc_splits.items():
                    systematics[f"btagsf{i}" + name] = self.compute_btag_sys(
                        central, "heavy up" + split, "heavy down" + split,
                        weighter, wp, flav, eta, pt, discr, efficiency)
                for name, split in light_unc_splits.items():
                    systematics[f"btagsf{i}light" + name] = \
                        self.compute_btag_sys(
                            central, "light up" + split, "light down" + split,
                            weighter, wp, flav, eta, pt, discr, efficiency)
            weight = weight * central
        if never_sys:
            return weight
        else:
            return weight, systematics

    def build_mtw_column(self,data):
        lepton = data["Lepton"]
        MET = data["MET"]
        #print(lepton.pt,lepton.phi)
        delta_phi = lepton[:, 0].phi-MET.phi
        #print(np.cos(delta_phi))
        mTW = np.sqrt(lepton[:,0].pt*MET.pt*(1-np.cos(delta_phi)))
        return mTW        

    def build_deltaR_lgamma(self,data):
        lepton = data["Lepton"]
        photon = data["Photon"]
        deltaR_lg = DeltaR(lepton[:,0],photon[:,0])
        return deltaR_lg

    def build_deltaR_ljet(self,data):
        lepton = data["Lepton"]
        jets = data["Jet"]
        deltaR_lj = DeltaR(lepton[:,0],jets[:,0])
        return deltaR_lj

    def build_deltaR_jgamma(self,data):
        jets = data["Jet"]
        photon = data["Photon"]
        deltaR_jg = DeltaR(jets[:,0],photon[:,0])
        return deltaR_jg

    def dummycut(self,data):
        return ak.num(data["Lepton"]) >= 0

    def build_genlepton_column(self, is_mc, data):

        genlepton = data["GenPart"]

        is_lepton = ( (abs(genlepton["pdgId"])==13) | (abs(genlepton["pdgId"])==11) )
        is_final_state = genlepton["status"]==1

        has_pt = genlepton["pt"]>30.
        has_eta = abs(genlepton["eta"])<2.5

        promptmatch = genlepton.hasFlags(['isPrompt'])
        promptmatch = ( (promptmatch) | (genlepton.hasFlags(['isPromptTauDecayProduct']))
                        | (genlepton.hasFlags(["fromHardProcess"]))
                      )

        genlepton = genlepton[is_lepton & is_final_state & has_pt & has_eta]

        genlepton = genlepton[ak.argsort(genlepton["pt"], ascending=False)]

        return genlepton

    def build_genphoton_column(self, is_mc, data):

        genphoton = data["GenPart"]

        genphoton = genphoton[genphoton["pdgId"]==22]
        genphoton = genphoton[(genphoton["status"]==1)]

        #pt filtering just to speed it up
        has_pt = (genphoton["pt"]>20)
        has_eta = (abs(genphoton["eta"])<2.6)
        genphoton = genphoton[has_pt & has_eta]

        genphoton = genphoton[ak.argsort(genphoton["pt"], ascending=False)]

        return genphoton

