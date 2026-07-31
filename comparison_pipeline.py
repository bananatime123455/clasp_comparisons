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

sys.path.insert(0, './models/UNIT')          
from UNIT_pipeline import UNITPipeline 

sys.path.insert(0, './models/pix2pixHD')
from Pix2PixHD_pipeline import Pix2PixHDPipeline


from plot_comparison import plot_comparison_table


class comp_pipeline:
    def __init__(self, h5_path, modality_1, modality_2,
                 data_dir_results, models_comparing, epochs,
                 batch_size, buffer_size,
                 early_stopping_patience, early_stopping_delta):
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

    def compare(self):
        train, test, val = sort_data(self.h5_files)
        all_results = []
        print(f"Models comparing: {self.models_comparing}")

        if 'Clasp_2D' in self.models_comparing:
            print("Running Clasp_2D comparison...")
            checkpoint = get_checkpoint(self.modality_1, self.modality_2)
            results = Clasp_2D_model(
                val, self.modality_1, self.modality_2, checkpoint, self.batch_size)
            if results:
                all_results.append(results)

        if 'cycle_GAN' in self.models_comparing:
            print("Running CycleGAN comparison...")
            cyclegan = MRIGANPipeline(
                train_files=train,
                test_files=test,
                val_files=val,
                modality_1=self.modality_1,
                modality_2=self.modality_2,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                epochs=self.epochs,
                early_stopping_patience=self.es_patience,
                early_stopping_delta=self.es_delta,
                data_dir_results = "./Results/cycle_GAN_run"
            )
            results = cyclegan.train()
            if results:
                all_results.append(results)

        if 'UNIT' in self.models_comparing:                 
            print("Running UNIT comparison...")             
            unit = UNITPipeline(                            
                train_files=train,                          
                test_files=test,                            
                val_files=val,                              
                modality_1=self.modality_1,                 
                modality_2=self.modality_2,                 
                batch_size=self.batch_size,                 
                buffer_size=self.buffer_size,               
                epochs=self.epochs,                         
                early_stopping_patience=self.es_patience,   
                early_stopping_delta=self.es_delta,         
                logdir=self.data_dir_results / "unit_run",  
            )                                               
            results = unit.train()                          
            if results:                                     
                all_results.append(results)  

        if 'pix2pixHD' in self.models_comparing:
            print("Running pix2pixHD comparison...")
            p2p = Pix2PixHDPipeline(
                train_files=train, test_files=test, val_files=val,
                modality_1=self.modality_1, modality_2=self.modality_2,
                batch_size=self.batch_size, buffer_size=self.buffer_size,
                epochs=self.epochs, early_stopping_patience=self.es_patience,
                early_stopping_delta=self.es_delta,
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