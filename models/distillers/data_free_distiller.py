# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import Mean, CategoricalAccuracy
from utils.trainers import Trainer
from utils.callbacks import Callback
from utils.modules import Reshape
from models.distillers.utils import PlaceholderDataset, IntermediateFeatureExtractor

class DataFreeGenerator(torch.nn.Module):
    def __init__(self, latent_dim:int=100, image_dim:Sequence[int]=[1, 32, 32]):
        super().__init__()
        self.latent_dim = latent_dim    
        self.image_dim = image_dim    

        self.base_dim = [128, *[dim//4 for dim in self.image_dim[1:]]]

        self.linear_1 = torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim))
        self.reshape = Reshape(out_shape=self.base_dim)
        self.conv_0 = torch.nn.BatchNorm2d(num_features=128)
        self.upsample_0 = torch.nn.Upsample(scale_factor=2)
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        x = self.linear_1(x)
        x = self.reshape(x)
        x = self.conv_0(x)
        x = self.upsample_0(x)
        x = self.conv_1(x)
        x = self.upsample_1(x)
        x = self.conv_2(x)
        return x

class DataFreeDistiller(Trainer):
    def __init__(
        self,
        teacher:IntermediateFeatureExtractor,
        student:torch.nn.Module,
        generator:torch.nn.Module,
        latent_dim:Optional[int]=None,
        image_dim:Optional[Sequence[int]]=None
    ):
        super().__init__()
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)
        self.generator = generator.to(self.device)
        self.image_dim = image_dim

        if latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        elif latent_dim is not None:
            self.latent_dim = latent_dim
        
        if self.image_dim is None:
            self.image_dim:int = self.student.input_dim
        elif self.image_dim is not None:
            self.image_dim = image_dim

    def compile(
        self,
        optimizer_student:torch.optim.Optimizer,
        optimizer_generator:torch.optim.Optimizer,
        onehot_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        activation_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        info_entropy_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        distill_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.KLDivLoss(reduction='batchmean'),
        student_loss_fn:Callable[[Any], torch.Tensor]=torch.nn.CrossEntropyLoss(),
        coeff_oh:float=1,
        coeff_ac:float=0.1,
        coeff_ie:float=5,
        batch_size:int=512,
        num_batches:int=120,
    ):
        if not isinstance(onehot_loss_fn, (Callable, bool)):
            warnings.warn('`onehot_loss_fn` should be of type `Callable[[Any], torch.Tensor]` or `bool`.')
        if not isinstance(activation_loss_fn, (Callable, bool)):
            warnings.warn('`activation_loss_fn` should be of type `Callable[[Any], torch.Tensor]` or `bool`.')
        if not isinstance(info_entropy_loss_fn, (Callable, bool)):
            warnings.warn('`info_entropy_loss_fn` should be of type `Callable[[Any], torch.Tensor]` or `bool`.')
        
        super().compile()
        self.optimizer_student = optimizer_student
        self.optimizer_generator = optimizer_generator
        self.onehot_loss_fn = onehot_loss_fn
        self.activation_loss_fn = activation_loss_fn
        self.info_entropy_loss_fn = info_entropy_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.student_loss_fn = student_loss_fn
        self.coeff_oh = coeff_oh
        self.coeff_ac = coeff_ac
        self.coeff_ie = coeff_ie
        self.batch_size = batch_size
        self.num_batches = num_batches

        # Config one-hot loss
        if self.onehot_loss_fn is True:
            self._onehot_loss_fn = torch.nn.CrossEntropyLoss()
        elif self.onehot_loss_fn is False:
            self._onehot_loss_fn = lambda *args, **kwargs:0
        else:
            self._onehot_loss_fn = self.onehot_loss_fn
        # Config activation loss
        if self.activation_loss_fn is True:
            pass
        elif self.activation_loss_fn is False:
            self._activation_loss_fn = lambda *args, **kwargs:0
        else:
            self._activation_loss_fn = self.activation_loss_fn
        # Config information entropy loss
        if self.info_entropy_loss_fn is True:
            pass
        elif self.info_entropy_loss_fn is False:
            self._info_entropy_loss_fn = lambda *args, **kwargs:0
        else:
            self._info_entropy_loss_fn = self.info_entropy_loss_fn

        # Placeholder data generator
        self.train_data = DataLoader(
            dataset=PlaceholderDataset(
                num_batches=self.num_batches,
                batch_size=self.batch_size
            ),
            batch_size=self.batch_size
        )

        # Metrics
        if self.onehot_loss_fn is not False:
            self.train_metrics.update({'loss_oh': Mean()})
        if self.activation_loss_fn is not False:
            self.train_metrics.update({'loss_ac': Mean()})
        if self.info_entropy_loss_fn is not False:
            self.train_metrics.update({'loss_ie': Mean()})
        self.train_metrics.update({
            'loss_gen': Mean(),
            'loss_dt': Mean(),
        })

        self.val_metrics = {
            'loss': Mean(),
            'acc': CategoricalAccuracy(),
        }

    def train_batch(self, data:Any):
        # Data-free, no need to unpack placeholder data
        # Teacher is frozen for inference only
        self.teacher.eval()

        # Phase 1: Training the Generator
        self.generator.train()
        self.student.eval()
        self.optimizer_generator.zero_grad()
        ## Forward
        x_synth = self.synthesize_images()
        features_teacher, logits_teacher = self.teacher(x_synth)
        pseudo_label = torch.argmax(input=logits_teacher.clone().detach(), dim=1)
        
        loss_onehot = self._onehot_loss_fn(logits_teacher, pseudo_label)
        loss_activation = self._activation_loss_fn(features_teacher.get('flatten'))
        loss_info_entropy = self._info_entropy_loss_fn(logits_teacher)
        loss_generator = (
            self.coeff_oh*loss_onehot
            + self.coeff_ac*loss_activation
            + self.coeff_ie*loss_info_entropy
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

    def test_batch(self, data:Sequence[torch.Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)

        self.student.eval()
        with torch.inference_mode():
            # Forward
            logits_student = self.student(input)
            loss_student = self.student_loss_fn(logits_student, label)
            # Metrics
            self.val_metrics['loss_st'].update(new_entry=loss_student)
            self.val_metrics['acc_st'].update(label=label, prediction=logits_student)

    def synthesize_images(self) -> torch.Tensor:
        latent_noise = torch.normal(mean=0, std=1, size=[self.batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(latent_noise)
        return x_synth

    def training_loop(self, num_epochs: int, valloader: Optional[DataLoader] = None, callbacks: Optional[Sequence[Callback]] = None):
        return super().training_loop(
            trainloader=self.train_data,
            num_epochs=num_epochs,
            valloader=valloader,
            callbacks=callbacks,
        )

    @staticmethod
    def _activation_loss_fn(inputs:torch.Tensor) -> torch.Tensor:
        loss = -inputs.abs().mean(dim=[0, 1])
        return loss

    @staticmethod
    def _info_entropy_loss_fn(inputs:torch.Tensor) -> torch.Tensor:
        prob = torch.nn.functional.softmax(inputs, dim = 1).mean(dim = 0)
        loss = (prob * torch.log10(prob)).sum()
        return loss

if __name__ == '__main__':
    from utils.dataloader import get_dataloader
    from models.classifiers import LeNet5, ClassifierTrainer
    from utils.callbacks import CSVLogger, ModelCheckpoint
    from models.GANs.utils import MakeSyntheticGIFCallback

    def expt_mnist(pretrained_teacher:bool=True, skip_baseline:bool=True):
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
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 256, 512
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 10, 200
        COEFF_OH, COEFF_AC, COEFF_IE = 1, 0.1, 5

        LEARNING_RATE_TEACHER, LEARNING_RATE_GENERATOR, LEARNING_RATE_STUDENT = 1e-3, 2e-1, 2e-3
        
        print(' Experiment 4.1: DAFL on MNIST. Teacher: LeNet-5, student: LeNet-5-HALF '.center(80,'#'))
        
        dataloader = get_dataloader(
            dataset='MNIST',
            resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
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

        if pretrained_teacher is True:
            teacher.load_state_dict(torch.load(
                f=f'./pretrained/MNIST - standardization/LeNet5_ReLU_MaxPool_9914.pt',
                map_location=trainer.device,
            ))
        elif pretrained_teacher is False:
            trainer.evaluate(dataloader['val'])
            best_callback = ModelCheckpoint(
                target=teacher,
                filepath=f'./logs/{teacher.__class__.__name__}_best.pt',
                monitor='val_acc',
                save_best_only=True,
                save_state_dict_only=True,
                initial_value_threshold=0.96,
            )
            csv_logger = CSVLogger(filename=f'./logs/{teacher.__class__.__name__}.csv', append=True)
            trainer.training_loop(
                trainloader=dataloader['train'],
                num_epochs=NUM_EPOCHS_TEACHER,
                valloader=dataloader['val'],
                callbacks=[csv_logger, best_callback],
            )
            
            teacher.load_state_dict(torch.load(
                f=f'./logs/{teacher.__class__.__name__}_best.pt',
                map_location=trainer.device,
            ))
        trainer.evaluate(dataloader['val'])

        teacher = IntermediateFeatureExtractor(
            model=teacher,
            out_layers={'flatten':teacher.flatten}
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
        if skip_baseline is False:
            trainer.evaluate(dataloader['val'])
            best_callback = ModelCheckpoint(
                target=student,
                filepath=f'./logs/{student.__class__.__name__}_st.pt',
                monitor='val_acc',
                save_best_only=True,
                save_state_dict_only=True,
                initial_value_threshold=0.96,
            )
            csv_logger = CSVLogger(filename=f'./logs/{student.__class__.__name__}_st.csv', append=True)
            trainer.training_loop(
                trainloader=dataloader['train'],
                num_epochs=NUM_EPOCHS_TEACHER,
                valloader=dataloader['val'],
                callbacks=[csv_logger, best_callback],
            )
            
            student.load_state_dict(torch.load(
                f=f'./logs/{student.__class__.__name__}_st.pt',
                map_location=trainer.device,
            ))
        trainer.evaluate(dataloader['val'])

        # Distillation
        student = LeNet5(
            half_size=True,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            ActivationLayer=torch.nn.ReLU,
            PoolLayer=torch.nn.MaxPool2d,
            return_logits=True
        )
        generator = DataFreeGenerator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)

        distiller = DataFreeDistiller(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=torch.optim.Adam(params=student.parameters(), lr=LEARNING_RATE_STUDENT),
            optimizer_generator=torch.optim.Adam(params=generator.parameters(), lr=LEARNING_RATE_GENERATOR),
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            distill_loss_fn=torch.nn.KLDivLoss(reduction='batchmean', log_target=True),
            student_loss_fn=torch.nn.CrossEntropyLoss(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
        )

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

    def expt_cifar10(pretrained_teacher:bool=False):
        # Algorithm                     Required data   FLOPS   params  CIFAR-10  CIFAR-100
        # Teacher                       Original data   16G     21M     95.58%    77.84%
        # Standard back-propagation     Original data   557M    11M     93.92%    76.53%
        # Knowledge Distillation [8]    Original data   557M    11M     94.34%    76.87%
        # Normal distribution           No data         557M    11M     14.89%     1.44%
        # Alternative data              Similar data    557M    11M     90.65%    69.88%
        # Data-Free Learning (DAFL)     No data         557M    11M     92.22%    74.47%
        LATENT_DIM = 1000
        IMAGE_DIM = [32, 32, 3]
        NUM_CLASSES = 10
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 128, 1024
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 200, 2000
        COEFF_OH, COEFF_AC, COEFF_IE = 0.05, 0.01, 5

        LEARNING_RATE_TEACHER = 0.1 # 1e-1 to 1e-2 to 1e-3
        LEARNING_RATE_GENERATOR = 2e-2
        LEARNING_RATE_STUDENT = 1e-1

        print(' Experiment 4.4: DAFL on CIFAR-10. Teacher: ResNet-34, student: ResNet-18 '.center(80,'#'))

        def augmentation_fn(x):
            x = tf.pad(tensor=x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
            x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], *IMAGE_DIM])
            x = tf.image.random_flip_left_right(image=x)
            return x
        ds = dataloader(
            dataset='cifar10',
            augmentation_fn=augmentation_fn,
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024
        )

        # Teacher (ResNet-34)
        teacher = ResNet_DAFL(ver=34, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        teacher.build()
        teacher.compile(
                metrics=['accuracy'], 
                optimizer=OPTIMIZER_TEACHER,
                loss=keras.losses.SparseCategoricalCrossentropy())

        if pretrained_teacher is True:
            teacher.load_weights(filepath=f'./pretrained/cifar10/mean0_std1/ResNet-DAFL-34_9499.h5')
        elif pretrained_teacher is False:
            def schedule(epoch:int, learing_rate:float):
                if epoch in [80, 120]:
                    learing_rate = learing_rate*0.1
                return learing_rate
            lr_scheduler = LearningRateSchedulerCustom(
                schedule=schedule,
                optimizer_name='optimizer',
                verbose=1)
            best_callback = keras.callbacks.ModelCheckpoint(
                filepath=f'./logs/{teacher.name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
            )
            csv_logger = keras.callbacks.CSVLogger(
                filename=f'./logs/{teacher.name}.csv',
                append=True
            )
            teacher.fit(
                ds['train'],
                epochs=NUM_EPOCHS_TEACHER,
                callbacks=[best_callback, lr_scheduler, csv_logger],
                validation_data=ds['test']
            )
            teacher.load_weights(filepath=f'./logs/{teacher.name}_best.h5')
        teacher.evaluate(ds['test'])
        teacher = convert_to_functional(
            model=teacher,
            inputs=keras.layers.Input(shape=teacher.input_dim))
        teacher = add_intermediate_outputs(
            model=teacher,
            layers=teacher.get_layer('glb_pool'))

        # Student (ResNet-18)
        student = ResNet_DAFL(ver=18, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        student.build()
        student.compile(metrics='accuracy')
        student.evaluate(ds['test'])

        generator = DataFreeGenerator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
        generator.build()

        # Train one student with default data-free learning settings
        distiller = DataFreeDistiller(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=OPTIMIZER_STUDENT,
            optimizer_generator=OPTIMIZER_GENERATOR,
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
            confidence=None
        )

        def schedule(epoch:int, learing_rate:float):
            if epoch in [800, 1600]:
                learing_rate = learing_rate*0.1
            return learing_rate
        lr_scheduler = LearningRateSchedulerCustom(
            schedule=schedule,
            optimizer_name='optimizer_student',
            verbose=1)
        csv_logger = CSVLogger_custom(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            nrows=5, ncols=5,
            postprocess_fn=lambda x:x*tf.constant([[[0.2470, 0.2435, 0.2616]]]) + tf.constant([[[0.4914, 0.4822, 0.4465]]]),
            normalize=False,
            # save_freq=NUM_EPOCHS_DISTILL//50
        )

        distiller.fit(
            epochs=NUM_EPOCHS_DISTILL,
            callbacks=[lr_scheduler, csv_logger, gif_maker],
            shuffle=True,
            validation_data=ds['test']
        )

    expt_mnist(pretrained_teacher=True, skip_baseline=True)