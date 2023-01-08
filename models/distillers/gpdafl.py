# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader

from models.bayes_opt.acquisition import AcquisitionFunction
from models.bayes_opt.gaussian_process import GaussianProcess, FixedNoiseGaussianProcess
from models.bayes_opt.kernels import RadialBasisFunctionKernel
from models.distillers.data_free_distiller import DataFreeGenerator, DataFreeDistiller
from models.distillers.utils import IntermediateFeatureExtractor
from utils.metrics import Mean

class GPDAFL(DataFreeDistiller):
    def __init__(
        self,
        teacher:IntermediateFeatureExtractor,
        student:torch.nn.Module,
        generator:torch.nn.Module,
        gp:GaussianProcess,
        latent_dim:Optional[int]=None,
        image_dim:Optional[Sequence[int]]=None,
    ):
        super().__init__(
            teacher=teacher,
            student=student,
            generator=generator,
            latent_dim=latent_dim,
            image_dim=image_dim,
        )
        self.gp = gp.to(self.device)

    def compile(
        self,
        optimizer_student:torch.optim.Optimizer,
        optimizer_generator:torch.optim.Optimizer,
        info_entropy_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        acquisition_loss_fn:Union[bool, AcquisitionFunction]=True,
        distill_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.KLDivLoss(reduction='batchmean'),
        student_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.CrossEntropyLoss(),
        coeff_aq:float=0.1,
        coeff_ie:float=5,
        batch_size:int=512,
        num_batches:int=120,
        gp_bound_extend:float=0,
    ):
        super().compile(
            optimizer_student=optimizer_student,
            optimizer_generator=optimizer_generator,
            onehot_loss_fn=False,
            activation_loss_fn=False,
            info_entropy_loss_fn=info_entropy_loss_fn,
            distill_loss_fn=distill_loss_fn,
            student_loss_fn=student_loss_fn,
            coeff_ie=coeff_ie,
            coeff_oh=0,
            coeff_ac=0,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        self.acquisition_loss_fn = acquisition_loss_fn
        self.gp_bound_extend = gp_bound_extend
        self.coeff_aq = coeff_aq

        # Config acquisition loss
        if self.acquisition_loss_fn is True:
            self._acquisition_loss_fn = MeanVarianceLoss()
        elif self.acquisition_loss_fn is False:
            self._acquisition_loss_fn = lambda *args, **kwargs:0
        else:
            self._acquisition_loss_fn = self.acquisition_loss_fn

        if self.acquisition_loss_fn is not False:
            self.train_metrics.update({'loss_aq': Mean()})

    def train_batch(self, data:Any):
        # Data-free, no need to unpack placeholder data
        # Teacher is frozen for inference only
        self.teacher.eval()
        ## Forward
        x_synth = self.synthesize_images()
        features_teacher, logits_teacher = self.teacher(x_synth)

        # Phase 1: Training the Generator
        self.generator.train()
        self.student.eval()
        self.optimizer_generator.zero_grad()

        self.gp(features_teacher['out']['gp_label'])
        loss_acquisition:torch.Tensor = self._acquisition_loss_fn(gp=self.gp)
        loss_info_entropy = self._info_entropy_loss_fn(logits_teacher)
        loss_generator = (
            + self.coeff_ie*loss_info_entropy
            + self.coeff_aq*loss_acquisition.mean()
        )
        
        # Phase 2: Training the Student
        self.generator.eval()
        self.student.train()
        self.optimizer_student.zero_grad()
        ## Forward
        logits_student:torch.Tensor = self.student(x_synth.clone().detach())
        log_prob_student = logits_student.log_softmax(dim=1)
        prob_teacher = logits_teacher.clone().detach().softmax(dim=1)
        loss_distill:torch.Tensor = self.distill_loss_fn(input=log_prob_student, target=prob_teacher)
        
        # Phase 3: Training the Gaussian process
        confidence:torch.Tensor = logits_teacher.clone().detach().softmax(dim=1).max(dim=1, keepdim=True)[0]
        best_variation = loss_acquisition.clone().detach().argmin()
        self.gp.update(
            x_train=features_teacher['out']['gp_label'][best_variation].clone().detach().unsqueeze(dim=0),
            y_train=confidence[best_variation].unsqueeze(dim=0),
        )

        ## Backward
        loss_generator.backward()
        self.optimizer_generator.step()
        loss_distill.backward()
        self.optimizer_student.step()

        with torch.inference_mode():
            # Metrics
            if self.info_entropy_loss_fn is not False:
                self.train_metrics['loss_ie'].update(loss_info_entropy)
            if self.acquisition_loss_fn is not False:
                self.train_metrics['loss_aq'].update(loss_acquisition.mean())
            self.train_metrics['loss_gen'].update(loss_generator)
            self.train_metrics['loss_dt'].update(loss_distill)      

    def head_start_gp(self, trainloader:DataLoader):
        with torch.inference_mode():
            for x, _ in trainloader:
                x = x.to(self.device)
                features_teacher, logits_teacher = self.teacher(x)
                confidence = logits_teacher.softmax(dim=1).max(dim=1, keepdim=True)[0]

            self.gp.update(
                x_train=features_teacher['out']['gp_label'],
                y_train=confidence,
            )

class MeanVarianceLoss(AcquisitionFunction):
    def __init__(self, beta:float=0.1):
        super().__init__()
        self.beta = beta

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(beta={self.beta})'

    def forward(self, gp:GaussianProcess) -> torch.Tensor:
        mean = gp.mean_posterior.squeeze(dim=1)
        var = gp.covar_posterior.diag()
        return -(mean + self.beta*var)

if __name__ == '__main__':
    from utils.data.small_mnist import get_SmallMNIST_loader
    from models.classifiers import LeNet5, ClassifierTrainer
    from utils.callbacks import CSVLogger, ModelCheckpoint
    from models.GANs.utils import MakeSyntheticGIFCallback

    def expt_mnist():
        # Experiment 4.1: Classification result on the MNIST dataset
        #                                       LeNet-5        HintonNets
        # Teacher:                              LeNet-5        Hinton-784-1200-1200-10
        # Student/Baseline:                     LeNet-5-HALF   Hinton-784-800-800-10
        # Teacher:                              98.91%         98.39%
        # Baseline:                             98.65%         98.11%
        # Traditional KD:                       98.91%         98.39%
        # Data-free KD:                         98.20%         97.91%    
        LATENT_DIM = 100
        IMAGE_DIM = [1, 32, 32] # LeNet-5 accepts [32, 32] images
        NUM_CLASSES = 10
        BATCH_SIZE_DISTILL = 512
        NUM_EPOCHS_DISTILL = 200
        COEFF_IE, COEFF_AQ = 1, 0.1

        GP_NOISE = 1e-3
        KERNEL, KERNEL_KWARGS = RadialBasisFunctionKernel, {'variance':10, 'length_scale':0.5}, 

        LEARNING_RATE_TEACHER, LEARNING_RATE_GENERATOR, LEARNING_RATE_STUDENT = 1e-3, 2e-1, 2e-3

        print(' Experiment 4.1: GP-DAFL on MNIST. Teacher: LeNet-5, student: LeNet-5-HALF '.center(80,'#'))
        
        dataloader = get_SmallMNIST_loader(
            resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=50,
            batch_size_val=1024
        )

        # Teacher (LeNet-5)
        teacher = LeNet5(
            half_size=False,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            ActivationLayer=torch.nn.ReLU,
            PoolLayer=torch.nn.MaxPool2d,
            return_logits=True
        )
        trainer = ClassifierTrainer(model=teacher)
        trainer.compile(
            optimizer=torch.optim.Adam(params=teacher.parameters(), lr=LEARNING_RATE_TEACHER),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
        teacher.load_state_dict(torch.load(
            f=f'./pretrained/MNIST - standardization/LeNet5_ReLU_MaxPool_9914.pt',
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['val'])

        teacher = IntermediateFeatureExtractor(
            model=teacher,
            out_layers={'flatten':teacher.flatten, 'gp_label':teacher.F6}
        )

        # Student (LeNet-5-HALF)
        student = LeNet5(
            half_size=True,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            ActivationLayer=torch.nn.ReLU,
            PoolLayer=torch.nn.MaxPool2d,
            return_logits=True
        )
        trainer = ClassifierTrainer(model=student)
        trainer.compile(
            optimizer=torch.optim.Adam(params=student.parameters(), lr=LEARNING_RATE_TEACHER),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
        trainer.evaluate(dataloader['val'])

        # Distillation
        gp = FixedNoiseGaussianProcess(
            x_train=torch.empty(0, 84),
            y_train=torch.empty(0, 1),
            kernel=KERNEL(**KERNEL_KWARGS),
            noise=GP_NOISE,
        )
        student = LeNet5(
            half_size=True,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            ActivationLayer=torch.nn.ReLU,
            PoolLayer=torch.nn.MaxPool2d,
            return_logits=True
        )
        generator = DataFreeGenerator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)

        distiller = GPDAFL(
            teacher=teacher,
            student=student,
            generator=generator,
            gp=gp,
        )
        distiller.compile(
            optimizer_student=torch.optim.Adam(params=student.parameters(), lr=LEARNING_RATE_STUDENT),
            optimizer_generator=torch.optim.Adam(params=generator.parameters(), lr=LEARNING_RATE_GENERATOR),
            info_entropy_loss_fn=True,
            acquisition_loss_fn=True,
            distill_loss_fn=torch.nn.KLDivLoss(reduction='batchmean', log_target=True),
            student_loss_fn=torch.nn.CrossEntropyLoss(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_ie=COEFF_IE,
            coeff_aq=COEFF_AQ,
        )
        ## Stage 1: Fit the Gaussian process
        distiller.head_start_gp(trainloader=dataloader['train'])
        ## Stage 2: Distillation
        csv_logger = CSVLogger(
            filename=f'./logs/{distiller.__class__.__name__}_{student.__class__.__name__}_mnist.csv',
            append=True
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{distiller.__class__.__name__}_{student.__class__.__name__}_mnist.gif',
            nrows=5, ncols=5,
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            save_freq=NUM_EPOCHS_DISTILL//50
        )

        distiller.training_loop(
            num_epochs=NUM_EPOCHS_DISTILL,
            valloader=dataloader['val'],
            callbacks=[csv_logger, gif_maker],
        )

    expt_mnist()