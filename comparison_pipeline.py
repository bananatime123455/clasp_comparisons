import sys
import json
import pathlib
from pathlib import Path
from sort_data import sort_data
from get_Checkpoint import get_checkpoint

sys.path.insert(0, './models/Clasp_2D')
from Clasp_2D_main import Clasp_2D_model

sys.path.insert(0, './models/cycleGan')
from pipeline import MRIGANPipeline

from plot_comparison import plot_comparison_table


class comp_pipeline:
    def __init__(self, h5_path, modality_1, modality_2,
                 data_dir_results, models_comparing, epochs,
                 data_dir_png_t1=None, data_dir_png_t2=None,
                 img_height=256, img_width=256,
                 batch_size=16, buffer_size=1000,
                 early_stopping_patience=5, early_stopping_delta=0.01):

        self.h5_files = sorted(Path(h5_path).glob("**/*.h5"))        
        print(f"h5_files found: {len(self.h5_files)}")
        self.modality_1       = modality_1
        self.modality_2       = modality_2
        self.data_dir_results = pathlib.Path(data_dir_results)
        self.data_dir_results.mkdir(parents=True, exist_ok=True)
        self.models_comparing = models_comparing
        self.epochs           = epochs
        self.data_dir_png_t1  = data_dir_png_t1
        self.data_dir_png_t2  = data_dir_png_t2
        self.img_height       = img_height
        self.img_width        = img_width
        self.batch_size       = batch_size
        self.buffer_size      = buffer_size
        self.es_patience      = early_stopping_patience
        self.es_delta         = early_stopping_delta

    def compare(self):
        train, test, val = sort_data(self.h5_files)
        all_results = []

        if 'Clasp_2D' in self.models_comparing:
            checkpoint = get_checkpoint(self.modality_1, self.modality_2)
            results = Clasp_2D_model(
                val, self.modality_1, self.modality_2, checkpoint)
            if results:
                all_results.append(results)

        if 'cycle_GAN' in self.models_comparing:
            if not self.data_dir_png_t1 or not self.data_dir_png_t2:
                raise ValueError(
                    "cycle_GAN requires --data_dir_png_t1 and --data_dir_png_t2")
            cyclegan = MRIGANPipeline(
                data_dir_t1=self.data_dir_png_t1,
                data_dir_t2=self.data_dir_png_t2,
                data_dir_results=str(self.data_dir_results / 'cycleGAN'),
                modality_1=self.modality_1,
                modality_2=self.modality_2,
                img_height=self.img_height,
                img_width=self.img_width,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                epochs=self.epochs,
                early_stopping_patience=self.es_patience,
                early_stopping_delta=self.es_delta,
            )
            results = cyclegan.train()  # already returns full dict
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