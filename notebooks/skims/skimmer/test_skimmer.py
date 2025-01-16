import json
import dask
import dask_awkward as dak
import awkward as ak
import numpy as np
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from ndcctools.taskvine import DaskVine
import fastjet
import time
import os
import warnings
from variable_functions import *
import scipy

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [8123, 8128],
        name=f"{os.environ['USER']}-hgg2",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    m.tune("temp-replica-count", 1)
    m.tune("transfer-temps-recovery", 1)
    
    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "invalid value")
    
    with open('input_datasets.json', 'r') as f:
        samples = json.load(f)

    print('doing samples')
    sample_start = time.time()

    @dask.delayed
    def sampler(samples):
        samples_ready, samples = dataset_tools.preprocess(
            samples,
            step_size=3_000,
            skip_bad_files=True,
            recalculate_steps=True,
            save_form=False,
        )
        return samples_ready

    sampler_dict = {}
    for i in samples:
        sampler_dict[i] = sampler(samples[i])

    print('Compute')
    samples_postprocess = dask.compute(
        sampler_dict,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        prune_files=True,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]

    samples_ready = {}
    for i in samples_postprocess:
        samples_ready[i] = samples_postprocess[i]['files']

    sample_stop = time.time()
    print('samples done')
    print('full sample time is ' + str((sample_stop - sample_start)/60))

    with open("samples_ready.json", "w") as fout:
        json.dump(samples_ready, fout)

    with open("samples_ready.json", 'r') as fin:
        samples_ready = json.load(fin)

    def analysis(events):
        dataset = events.metadata["dataset"]
        print(dataset)

        events['PFCands', 'pt'] = (
            events.PFCands.pt
            * events.PFCands.puppiWeight
        )
        
        cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)
        events = events[ak.all(cut_to_fix_softdrop, axis=1)]

        # num_sub = ak.unflatten(num_subjets(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet))
        # events['FatJet', 'num_subjets'] = num_sub

        if ('hgg' in dataset) or ('hbb' in dataset):
            print("signal")
            genhiggs = events.GenPart[
                (events.GenPart.pdgId == 25)
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
            parents = events.FatJet.nearest(genhiggs, threshold=0.2)
            higgs_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 800)
                #& (events.FatJet.num_subjets >= 3)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (higgs_jets)
            )

        elif ('wqq' in dataset) or ('ww' in dataset):
            print('w background')
            genw = events.GenPart[
                (abs(events.GenPart.pdgId) == 24)
                & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
            ]
            parents = events.FatJet.nearest(genw, threshold=0.2)
            w_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 800)
                #& (events.FatJet.num_subjets >= 3)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (w_jets)
            )

        elif ('zqq' in dataset) or ('zz' in dataset):
            print('z background')
            genz = events.GenPart[
                (events.GenPart.pdgId == 23)
                & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
            ]
            parents = events.FatJet.nearest(genz, threshold=0.2)
            z_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 800)
                #& (events.FatJet.num_subjets >= 3)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (z_jets)
            )

        elif ('wz' in dataset):
            print('wz background')
            genwz = events.GenPart[
                ((abs(events.GenPart.pdgId) == 24)|(events.GenPart.pdgId == 23))
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
            parents = events.FatJet.nearest(genwz, threshold=0.2)
            wz_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 800)
                #& (events.FatJet.num_subjets >= 3)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (wz_jets)
            )
    
        else:
            print('background')
            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 800)
                #& (events.FatJet.num_subjets >= 3)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
            )
        
        events["goodjets"] = events.FatJet[fatjetSelect]
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]
        ecfs = {}
        
        events['goodjets', 'color_ring'] = ak.unflatten(
             color_ring(events.goodjets, cluster_val=0.4), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'd2b1'] = ak.unflatten(
             d2_calc(events.goodjets), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'u1'] = ak.unflatten(
             u_calc(events.goodjets, 1), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'u2'] = ak.unflatten(
             u_calc(events.goodjets, 2), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'u3'] = ak.unflatten(
             u_calc(events.goodjets, 3), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'd3'] = ak.unflatten(
             d3_calc(events.goodjets), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'm2'] = ak.unflatten(
             m2_calc(events.goodjets), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'm3'] = ak.unflatten(
             m3_calc(events.goodjets), counts=ak.num(events.goodjets)
        )
        
        events['goodjets', 'n4'] = ak.unflatten(
             n4_calc(events.goodjets), counts=ak.num(events.goodjets)
        )

        # for n in range(2,6):
        #     for v in range(1, int(scipy.special.binom(n, 2))+1):
        #         for b in range(5, 45, 5):
        #             ecf_name = f'{v}e{n}^{b/10}'
        #             ecfs[ecf_name] = make_ecf(events.goodjets, n=n, v=v, b=b/10)

        # events["ecfs"] = ak.zip(ecfs)

        # skim = ak.zip(
        #     {
        #         "FatJets": ak.flatten(events.goodjets, axis=1),
        #         #  "MET": events.MET,
        #         #  "Photon": events.Photon,
        #         # "Subjets": events.SubJet,
        #         # "FJ_PFCands": events.FatJet.constituents.pf,
        #     },
        #     depth_limit=1,
        # )

        skim_task = dak.to_parquet(
            events,
            f"/project01/ndcms/cmoore24/skims/test/{dataset}2/",
            compute=False,
        )
        return skim_task


    subset = {}
    to_skim = 'hbb'
    subset[to_skim] = samples_ready[to_skim]
    files = subset[to_skim]['files']
    form = subset[to_skim]['form']
    dict_meta = subset[to_skim]['metadata']
    keys = list(files.keys())

    batch = {}
    batch[to_skim] = {}
    batch[to_skim]['files'] = {}
    batch[to_skim]['form'] = form
    batch[to_skim]['metadata'] = dict_meta

    for i in range(0, len(files)):
        batch[to_skim]['files'][keys[i]] = files[keys[i]]
    with open('batch_check.json', 'w') as f:
        json.dump(batch, f)
    print(len(batch[to_skim]['files']))

    tasks = dataset_tools.apply_to_fileset(
        analysis,
        #dataset_tools.slice_files(samples_ready, slice(None, 20)),
        #samples_ready,
        batch,
        uproot_options={"allow_read_errors_with_report": True},
        schemaclass = PFNanoAODSchema,
    )


    print('start compute')
    computed = dask.compute(
            tasks[0],
            scheduler=m.get,
            resources={"cores": 1},
            resources_mode=None,
            lazy_transfers=True,
            prune_files=True,
            #task_mode="function_calls",
            lib_resources={'cores': 12, 'slots': 12},
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
