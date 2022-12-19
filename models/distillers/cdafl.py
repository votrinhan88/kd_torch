# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import torch
import numpy as np

from utils.metrics import Mean
from utils.modules import Concatenate, OneHotEncoding, Reshape
from models.distillers.data_free_distiller import DataFreeDistiller
from models.distillers.utils import IntermediateFeatureExtractor

class ConditionalDataFreeGenerator(torch.nn.Module):
    def __init__(
        self,
        latent_dim:int=100,
        image_dim:Sequence[int]=[1, 32, 32],
        embed_dim:Optional[int]=None,
        num_classes:int=10,
        onehot_label=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_label = onehot_label

        self.base_dim = [128, *[dim//4 for dim in self.image_dim[1:]]]

        # Traditional latent branch
        self.latent_branch = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim)),
            Reshape(out_shape=self.base_dim),
        )

        # Conditional label branch
        label_branch = []
        if self.onehot_label is False:
            label_branch.append(OneHotEncoding(num_classes=self.num_classes))
        if self.embed_dim is not None:
            # Replace Embedding with Dense for to accept interpolated label inputs
            label_branch.extend([
                torch.nn.Linear(in_features=self.num_classes, out_features=self.embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.embed_dim, out_features=np.prod(self.base_dim[1:])),
            ])
        else:
            label_branch.append(
                torch.nn.Linear(in_features=self.num_classes, out_features=np.prod(self.base_dim[1:])),
            )
        label_branch.extend([
            Reshape(out_shape=(1, *self.base_dim[1:])),
        ])
        self.label_branch = torch.nn.Sequential(*label_branch)

        # Main branch: concat both branches and upsample
        self.concat = Concatenate()
        self.conv_0 = torch.nn.BatchNorm2d(num_features=128 + 1)
        self.upsample_0 = torch.nn.Upsample(scale_factor=2)
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128 + 1, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128, eps=0.8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.upsample_1 = torch.nn.Upsample(scale_factor=2)
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64, eps=0.8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=self.image_dim[0], kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(num_features=self.image_dim[0], affine=False) 
        )

    def forward(self, z, c):
        z = self.latent_branch(z)
        c = self.label_branch(c)

        x = self.concat([z, c])
        x = self.conv_0(x)
        x = self.upsample_0(x)
        x = self.conv_1(x)
        x = self.upsample_1(x)
        x = self.conv_2(x)
        return x

class ConditionalDataFreeDistiller(DataFreeDistiller):
    def __init__(
        self,
        teacher:IntermediateFeatureExtractor,
        student:torch.nn.Module,
        generator:torch.nn.Module,
        latent_dim:Optional[int]=None,
        image_dim:Optional[Sequence[int]]=None,
        num_classes:Optional[int]=None,
        onehot_label:Optional[bool]=None,
    ):
        super().__init__(
            teacher=teacher,
            student=student,
            generator=generator,
            latent_dim=latent_dim,
            image_dim=image_dim,
        )
        self.num_classes = num_classes
        self.onehot_label = onehot_label

        if self.num_classes is None:
            self.num_classes:int = self.generator.num_classes
        
        if self.onehot_label is None:
            self.onehot_label:bool = self.generator.onehot_label
    
    def compile(
        self,
        optimizer_student:torch.optim.Optimizer,
        optimizer_generator:torch.optim.Optimizer,
        onehot_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        activation_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        info_entropy_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        conditional_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        distribution_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        distribution_layer:Optional[torch.nn.BatchNorm2d]=None,
        distill_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.KLDivLoss(reduction='batchmean'),
        student_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.CrossEntropyLoss(),
        coeff_oh:float=1,
        coeff_ac:float=0.1,
        coeff_ie:float=5,
        coeff_cn:float=1,
        coeff_ds:float=1,
        batch_size:int=500,
        num_batches:int=120,
    ):
        if batch_size % self.num_classes > 0:
            warnings.warn(
                f'`batch_size` {batch_size} is not divisible by `num_classes` '+
                f'{self.num_classes} and will give unevenly distributed batches.')

        super().compile(
            optimizer_student=optimizer_student,
            optimizer_generator=optimizer_generator,
            onehot_loss_fn=onehot_loss_fn,
            activation_loss_fn=activation_loss_fn,
            info_entropy_loss_fn=info_entropy_loss_fn,
            distill_loss_fn=distill_loss_fn,
            student_loss_fn=student_loss_fn,
            coeff_oh=coeff_oh,
            coeff_ac=coeff_ac,
            coeff_ie=coeff_ie,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        self.conditional_loss_fn = conditional_loss_fn
        self.distribution_loss_fn = distribution_loss_fn
        self.distribution_layer = distribution_layer
        self.coeff_cn = coeff_cn
        self.coeff_ds = coeff_ds

        # Config conditional loss
        if self.conditional_loss_fn is True:
            if self.onehot_label is True:
                self._conditional_loss_fn = lambda pred, target: torch.nn.functional.cross_entropy(input=pred, target=target.argmax(dim=1))
            elif self.onehot_label is False:
                self._conditional_loss_fn = torch.nn.CrossEntropyLoss()
        elif self.conditional_loss_fn is False:
            self._conditional_loss_fn = lambda *args, **kwargs:0
        else:
            self._conditional_loss_fn = self.conditional_loss_fn
        # Config distribution loss
        if self.distribution_layer is None:
            self.distribution_loss_fn = False
        
        if self.distribution_loss_fn is True:
            pass
        elif self.distribution_loss_fn is False:
            self._distribution_loss_fn = lambda *args, **kwargs:0
        else:
            self._distribution_loss_fn = self.distribution_loss_fn

        # Reconfigure metrics
        if self.onehot_label is True:
            self.val_metrics.update({'acc': CategoricalAccuracy()})
        # Additional metrics
        if self.conditional_loss_fn is not False:
            self.train_metrics.update({'loss_cn': Mean()})
        if self.distribution_loss_fn is not False:
            self.train_metrics.update({'loss_ds': Mean()})

    def train_batch(self, data):
        # Data-free, no need to unpack placeholder data
        # Teacher is frozen for inference only
        self.teacher.eval()

        # Phase 1: Training the Generator
        self.generator.train()
        self.student.eval()
        self.optimizer_generator.zero_grad()
        ## Forward
        x_synth, label = self.synthesize_images()
        features_teacher, logits_teacher = self.teacher(x_synth)
        pseudo_label = torch.argmax(input=logits_teacher.clone().detach(), dim=1)

        loss_onehot = self._onehot_loss_fn(logits_teacher, pseudo_label)
        loss_activation = self._activation_loss_fn(features_teacher['out'].get('flatten'))
        loss_info_entropy = self._info_entropy_loss_fn(logits_teacher)
        loss_conditional = self._conditional_loss_fn(logits_teacher, label)
        loss_distribution = self._distribution_loss_fn(features_teacher['in'].get('distribution')[0], self.distribution_layer)

        loss_generator = (
            self.coeff_oh*loss_onehot
            + self.coeff_ac*loss_activation
            + self.coeff_ie*loss_info_entropy
            + self.coeff_cn*loss_conditional
            + self.coeff_ds*loss_distribution
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
        
        ## Backward
        loss_generator.backward()
        self.optimizer_generator.step()
        loss_distill.backward()
        self.optimizer_student.step()

        with torch.inference_mode():
            # Metrics
            if self.onehot_loss_fn is not False:
                self.train_metrics['loss_oh'].update(new_entry=loss_onehot)
            if self.activation_loss_fn is not False:
                self.train_metrics['loss_ac'].update(new_entry=loss_activation)
            if self.info_entropy_loss_fn is not False:
                self.train_metrics['loss_ie'].update(new_entry=loss_info_entropy)
            self.train_metrics['loss_gen'].update(new_entry=loss_generator)
            self.train_metrics['loss_dt'].update(new_entry=loss_distill)

    def synthesize_images(self) -> torch.Tensor:
        latent_noise = torch.normal(mean=0, std=1, size=[self.batch_size, self.latent_dim], device=self.device)
        label = torch.randint(low=0, high=self.num_classes, size=[self.batch_size], device=self.device)
        if self.onehot_label is True:
            label = torch.nn.functional.one_hot(input=label, num_classes=self.num_classes)
        label = label.to(torch.float)
        x_synth = self.generator(latent_noise, label)
        return x_synth, label

    @staticmethod
    def _distribution_loss_fn(fmap:torch.Tensor, distribution_layer:torch.nn.BatchNorm2d) -> torch.Tensor:
        batch_mean = fmap.mean(dim=[i for i in range(fmap.dim()) if i != 1], keepdim=True)
        batch_var = ((fmap - batch_mean)**2).mean(dim=[i for i in range(fmap.dim()) if i != 1])
        batch_mean = batch_mean.squeeze()
        loss = (
            torch.linalg.norm(distribution_layer.running_mean - batch_mean, ord=2) +
            torch.linalg.norm(distribution_layer.running_var - batch_var, ord=2)
        )
        return loss

if __name__ == '__main__':
    from models.classifiers import AlexNet, ClassifierTrainer
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback
    from utils.callbacks import CSVLogger
    from utils.dataloader import get_dataloader
    from utils.metrics import CategoricalAccuracy

    def expt_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        EMBED_DIM = None
        NUM_CLASSES = 10
        BATCH_SIZE = 128
        NUM_EPOCHS_DISTILL = 200

        dataloader = get_dataloader(
            dataset='MNIST',
            # resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
            onehot_label=True,
        )

        teacher = AlexNet(
            half_size=False,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            return_logits=True,
        )
        trainer = ClassifierTrainer(model=teacher)
        trainer.compile(
            optimizer=None,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
        teacher.load_state_dict(torch.load(
            f=f'./pretrained/MNIST - standardization/AlexNet_9950.pt',
            map_location=trainer.device,
        ))
        trainer.val_metrics.update({'acc': CategoricalAccuracy()})
        trainer.evaluate(dataloader['val'])
        hooked_teacher = IntermediateFeatureExtractor(
            model=teacher,
            in_layers={'distribution': teacher.conv_1[-1]},
            out_layers={'flatten':teacher.flatten},
        )

        student = AlexNet(
            half_size=True,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            return_logits=True,
        )
        generator = ConditionalDataFreeGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_label=True,
        )

        distiller = ConditionalDataFreeDistiller(
            teacher=hooked_teacher,
            student=student,
            generator=generator,
        )
        distiller.compile(
            optimizer_student=torch.optim.Adam(student.parameters(), lr=2e-3),
            optimizer_generator=torch.optim.Adam(generator.parameters(), lr=2e-1),
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            conditional_loss_fn=True,
            distribution_loss_fn=True,
            distribution_layer=teacher.conv_1[-1],
            distill_loss_fn=torch.nn.KLDivLoss(reduction='batchmean'),
            student_loss_fn=torch.nn.CrossEntropyLoss()
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{distiller.__class__.__name__}_{student.__class__.__name__}_mnist.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{distiller.__class__.__name__}_{student.__class__.__name__}_mnist.gif',
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            save_freq=NUM_EPOCHS_DISTILL//50
        )
        distiller.training_loop(
            num_epochs=NUM_EPOCHS_DISTILL,
            valloader=dataloader['val'],
            callbacks=[csv_logger, gif_maker]
        )

    expt_mnist()