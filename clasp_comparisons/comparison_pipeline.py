import pathlib
from sort_data import sort_data
import sys
sys.path.insert(0,'.models/Clasp_2D')

def __init__(self, data_dir_t1, data_dir_t2, modality_1, modality_2,data_dir_results,models_comparing, epochs):
        self.data_dir_t1 = pathlib.Path(data_dir_t1)
        self.data_dir_t2 = pathlib.Path(data_dir_t2)
        self.modality_1 = modality_1
        self.modality_2 = modality_2
        self.data_dir_results = pathlib.Path(data_dir_results)
        self.models_comparing = models_comparing
        self.epochs = epochs


def compare(self):
    #split the data into train, test, val for each modality
    mod1_data = list(self.data_dir_t1.glob('*.nii.gz'))
    mod2_data = list(self.data_dir_t2.glob('*.nii.gz'))
    tr1_train, tr1_test, tr1_val = sort_data(mod1_data)
    tr2_train, tr2_test, tr2_val = sort_data(mod2_data)


    #find the models to compare and run them
    if 'Clasp_2D' in self.models_comparing:
        print("Running Clasp_2D model...")

        cl






        # Initialize models
        lit_model = unet_generator()
        self.generator_g = lit_model.domain_models['Axial_T1_pre'].to(self.device)
        self.generator_f = lit_model.domain_models['Axial_T2'].to(self.device)
        self.discriminator_x = discriminator()
        self.discriminator_y = discriminator()

        self.generator_g = self.generator_g.to(self.device)
        self.generator_f = self.generator_f.to(self.device)
        self.discriminator_x = self.discriminator_x.to(self.device)
        self.discriminator_y = self.discriminator_y.to(self.device)

        self.generator_g_optimizer = torch.optim.Adam(self.generator_g.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.generator_f_optimizer = torch.optim.Adam(self.generator_f.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.discriminator_x_optimizer = torch.optim.Adam(self.discriminator_x.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.discriminator_y_optimizer = torch.optim.Adam(self.discriminator_y.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.tr1_train, self.tr2_train, self.tr1_test, self.tr2_test, self.tr1_val, self.tr2_val = load_datasets(
            self.data_dir_t1, self.data_dir_t2, self.img_height, self.img_width,
            self.batch_size, self.buffer_size)

        # add to __init__ after load_datasets
        print(f"T1 train: {len(list(self.tr1_train))}, T2 train: {len(list(self.tr2_train))}")
        print(f"T1 test:  {len(list(self.tr1_test))},  T2 test:  {len(list(self.tr2_test))}")

        # Early stopping monitors cycle loss across all 4 models
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=early_stopping_delta
        )