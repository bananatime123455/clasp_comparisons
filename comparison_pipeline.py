import sys
import json
import pathlib
from pathlib import Path

from sort_data import sort_data
from get_Checkpoint import get_checkpoint

sys.path.insert(0, './models/Clasp_2D')
from Clasp_2D_pipeline import Clasp_2D_model

sys.path.insert(0, './models/cycleGan')
from Cycle_GAN_pipeline import MRIGANPipeline

sys.path.insert(0, './models/UNIT_pix2pixHD')          
from UNIT_pipeline import UNITPipeline 

sys.path.insert(0, './models/UNIT_pix2pixHD')
from Pix2PixHD_pipeline import Pix2PixHDPipeline

sys.path.insert(0, './models/UNIT_pix2pixHD')
from MUNIT_pipeline import MUNITPipeline


from plot_comparison import plot_comparison_table


class comp_pipeline:
    def __init__(self, h5_path, modality_1, modality_2,
                 data_dir_results, models_comparing, epochs,
                 batch_size, buffer_size,
                 early_stopping_patience, early_stopping_delta,
                 model_params: dict = None):
        """
        model_params : dict, optional
            Per-model overrides of the training/early-stopping args, e.g.:
                {
                    "MUNIT": {"epochs": 5, "batch_size": 2,
                              "early_stopping_patience": 5},
                    "UNIT":  {"early_stopping_delta": 0.02},
                }
            Any key a model's entry doesn't specify (or any model not
            present in this dict at all) falls back to the shared
            epochs/batch_size/buffer_size/early_stopping_patience/
            early_stopping_delta passed above. Unknown model names are
            ignored (harmless — they just won't match any of the
            'X in self.models_comparing' branches below).
        """
        self.h5_files = sorted(Path(h5_path).glob("**/*.h5"))
        print(f"h5_files found: {len(self.h5_files)}")
        self.modality_1       = modality_1
        self.modality_2       = modality_2
        self.data_dir_results = pathlib.Path(data_dir_results)
        self.data_dir_results.mkdir(parents=True, exist_ok=True)
        self.models_comparing = models_comparing
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.buffer_size      = buffer_size
        self.es_patience      = early_stopping_patience
        self.es_delta         = early_stopping_delta
        self.model_params     = model_params or {}

    def _params_for(self, model_name: str) -> dict:
        """
        Merge this model's overrides (if any) on top of the shared
        defaults. Only the 5 core training/early-stopping keys are
        merged this way; a model-specific pipeline kwarg like MUNIT's
        munit_batch_size_scale is looked up separately with .get() at
        the call site so unrelated models' override dicts don't need
        to know about it.
        """
        overrides = self.model_params.get(model_name, {})
        return {
            'epochs': overrides.get('epochs', self.epochs),
            'batch_size': overrides.get('batch_size', self.batch_size),
            'buffer_size': overrides.get('buffer_size', self.buffer_size),
            'early_stopping_patience': overrides.get(
                'early_stopping_patience', self.es_patience),
            'early_stopping_delta': overrides.get(
                'early_stopping_delta', self.es_delta),
        }

    def compare(self):
        train, test, val = sort_data(self.h5_files)
        all_results = []
        print(f"Models comparing: {self.models_comparing}")

        if 'Clasp_2D' in self.models_comparing:
            p = self._params_for('Clasp_2D')
            print(f"Running Clasp_2D comparison... (batch_size={p['batch_size']})")
            checkpoint = get_checkpoint(self.modality_1, self.modality_2)
            results = Clasp_2D_model(
                val, self.modality_1, self.modality_2, checkpoint, p['batch_size'])
            if results:
                all_results.append(results)

        if 'cycle_GAN' in self.models_comparing:
            p = self._params_for('cycle_GAN')
            print(f"Running CycleGAN comparison... params={p}")
            cyclegan = MRIGANPipeline(
                train_files=train,
                test_files=test,
                val_files=val,
                modality_1=self.modality_1,
                modality_2=self.modality_2,
                batch_size=p['batch_size'],
                buffer_size=p['buffer_size'],
                epochs=p['epochs'],
                early_stopping_patience=p['early_stopping_patience'],
                early_stopping_delta=p['early_stopping_delta'],
                data_dir_results = "./Results/cycle_GAN_run"
            )
            results = cyclegan.train()
            if results:
                all_results.append(results)

        if 'UNIT' in self.models_comparing:
            p = self._params_for('UNIT')
            print(f"Running UNIT comparison... params={p}")
            unit = UNITPipeline(
                train_files=train,
                test_files=test,
                val_files=val,
                modality_1=self.modality_1,
                modality_2=self.modality_2,
                batch_size=p['batch_size'],
                buffer_size=p['buffer_size'],
                epochs=p['epochs'],
                early_stopping_patience=p['early_stopping_patience'],
                early_stopping_delta=p['early_stopping_delta'],
                logdir=self.data_dir_results / "unit_run",
            )
            results = unit.train()
            if results:
                all_results.append(results)

        if 'MUNIT' in self.models_comparing:
            p = self._params_for('MUNIT')
            munit_scale = self.model_params.get('MUNIT', {}).get(
                'munit_batch_size_scale', 0.5)
            print(f"Running MUNIT comparison... params={p} "
                  f"munit_batch_size_scale={munit_scale}")
            munit = MUNITPipeline(
                train_files=train,
                test_files=test,
                val_files=val,
                modality_1=self.modality_1,
                modality_2=self.modality_2,
                batch_size=p['batch_size'],
                buffer_size=p['buffer_size'],
                epochs=p['epochs'],
                early_stopping_patience=p['early_stopping_patience'],
                early_stopping_delta=p['early_stopping_delta'],
                logdir=self.data_dir_results / "munit_run",
                munit_batch_size_scale=munit_scale,
            )
            results = munit.train()
            if results:
                all_results.append(results)

        if 'pix2pixHD' in self.models_comparing:
            p = self._params_for('pix2pixHD')
            print(f"Running pix2pixHD comparison... params={p}")
            p2p = Pix2PixHDPipeline(
                train_files=train, test_files=test, val_files=val,
                modality_1=self.modality_1, modality_2=self.modality_2,
                batch_size=p['batch_size'], buffer_size=p['buffer_size'],
                epochs=p['epochs'],
                early_stopping_patience=p['early_stopping_patience'],
                early_stopping_delta=p['early_stopping_delta'],
                logdir=self.data_dir_results / "pix2pixhd_run",
            )
            results = p2p.train()
            if results:
                all_results.append(results)

        self._save_results(all_results)
        plot_comparison_table(
            all_results,
            self.modality_1, self.modality_2,
            self.data_dir_results)
        return all_results

    def _save_results(self, all_results):
        out_path = self.data_dir_results / "comparison_results.json"
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)