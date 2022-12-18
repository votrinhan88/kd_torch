# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch

from utils.metrics import Mean, CategoricalAccuracy
from utils.trainers import Trainer

class Distiller(Trainer):
    '''Traditional knowledge distillation scheme, training the student on both the
    transfer set and the soft targets produced by a pre-trained teacher.

    Args:
        `teacher`: Pre-trained teacher model. Must return logits.
        `student`: To-be-trained student model. Must return logits.
        `image_dim`: Dimension of input images, leave as `None` to be parsed from
            student. Defaults to `None`.

    Kwargs:
        Additional keyword arguments passed to `keras.Model.__init__`.

    Distilling the Knowledge in a Neural Network - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531
    '''
    def __init__(self,
                 teacher:torch.nn.Module,
                 student:torch.nn.Module,
                 image_dim:Optional[Sequence[int]]=None):
        """Initialize distiller.
        
        Args:
            `teacher`: Pre-trained teacher model. Must return logits.
            `student`: To-be-trained student model. Must return logits.
            `image_dim`: Dimension of input images, leave as `None` to be parsed from
                student. Defaults to `None`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.__init__`.
        """        
        super().__init__()
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)
        self.image_dim = image_dim
        
        if self.image_dim is None:
            self.image_dim:int = self.student.input_dim
        elif self.image_dim is not None:
            self.image_dim = image_dim

    def compile(
        self,
        optimizer:torch.optim.Optimizer,
        distill_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        student_loss_fn:Union[bool, Callable[[Any], torch.Tensor]]=True,
        coeff_dt:float=0.9,
        coeff_st:float=0.1,
        temperature:float=10,
    ):
        """Compile distiller.
        
        Args:
            `optimizer`: Optimizer for student model.
            `distill_loss_fn`: Loss function for distillation from teacher.
            `student_loss_fn`: Loss function for learning from training data.
            `coeff_dt`: weight assigned to student loss. Correspondingly, weight
                assigned to distillation loss is `1 - coeff_dt`.
            `temperature`: Temperature for softening probability distributions. Larger
                temperature gives softer distributions.
        """
        super().compile()
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.coeff_dt = coeff_dt
        self.coeff_st = coeff_st
        self.temperature = temperature

        # Config distillation loss
        if self.distill_loss_fn is True:
            self._distill_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        elif self.distill_loss_fn is False:
            self._distill_loss_fn = lambda *args, **kwargs:0
        else:
            self._distill_loss_fn = self.distill_loss_fn
        # Config student loss
        if self.student_loss_fn is True:
            self._student_loss_fn = torch.nn.CrossEntropyLoss()
            self._test_loss_fn = torch.nn.CrossEntropyLoss()
        elif self.student_loss_fn is False:
            self._student_loss_fn = lambda *args, **kwargs:0
            self._test_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self._student_loss_fn = self.student_loss_fn
            self._test_loss_fn = self.student_loss_fn

        # Metrics
        if self.distill_loss_fn is not False:
            self.train_metrics.update({'loss_dt': Mean()})
        if self.student_loss_fn is not False:
            self.train_metrics.update({'loss_st': Mean()})
        self.train_metrics.update({
            'loss': Mean(),
            'acc': CategoricalAccuracy(),
        })

        self.val_metrics = {
            'loss': Mean(),
            'acc': CategoricalAccuracy(),
        }

    def train_batch(self, data):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)

        self.teacher.eval()
        self.student.train()
        self.optimizer.zero_grad()
        # Forward
        logits_teacher:torch.Tensor = self.teacher(input)
        logits_student:torch.Tensor = self.student(input)
        # Standard loss with training data
        loss_student = self._student_loss_fn(logits_student, label)
        log_prob_student = (logits_student/self.temperature).log_softmax(dim=1)
        prob_teacher = (logits_teacher/self.temperature).softmax(dim=1)
        # Not multiplying with T^2 gives slightly better results
        loss_distill = self._distill_loss_fn(input=log_prob_student, target=prob_teacher)# * self.temperature**2
        loss = self.coeff_st*loss_student + self.coeff_dt*loss_distill
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        with torch.inference_mode():
            # Metrics
            if self.distill_loss_fn is not False:
                self.train_metrics['loss_dt'].update(new_entry=loss_distill)
            if self.student_loss_fn is not False:
                self.train_metrics['loss_st'].update(new_entry=loss_student)
            self.train_metrics['loss'].update(new_entry=loss)
            self.train_metrics['acc'].update(label=label, prediction=logits_student)

    def test_batch(self, data):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)

        self.student.eval()
        with torch.inference_mode():
            # Forward
            logits_student = self.student(input)
            loss_student = self._test_loss_fn(logits_student, label)
            # Metrics
            self.val_metrics['loss_st'].update(new_entry=loss_student)
            self.val_metrics['acc'].update(label=label, prediction=logits_student)

if __name__ == '__main__':
    from models.classifiers import HintonNet, ClassifierTrainer
    from utils.dataloader import get_dataloader
    from utils.callbacks import CSVLogger, ModelCheckpoint
    
    def expt_mnist(pretrained_teacher:bool=True, skip_baseline:bool=True):
        IMAGE_DIM = [28, 28, 1]
        NUM_CLASSES = 10
        HIDDEN_LAYERS_TEACHER, HIDDEN_LAYERS_STUDENT = [1200, 1200], [800, 800]
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 128
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 10, 10
        COEFF_DT = 0.9
        COEFF_ST = 0.1
        TEMPERATURE = 10

        print(' Standard knowledge distillation on MNIST '.center(80,'#'))

        dataloader = get_dataloader(
            dataset='MNIST',
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE,
        )

        teacher = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=HIDDEN_LAYERS_TEACHER,
            num_classes=NUM_CLASSES,
            return_logits=True
        )
        trainer = ClassifierTrainer(model=teacher)
        trainer.compile(
            optimizer=torch.optim.Adam(params=teacher.parameters(), lr=LEARNING_RATE),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )

        if pretrained_teacher is True:
            teacher.load_state_dict(torch.load(
                f=f'./logs/{teacher.__class__.__name__}_best.pt',
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

        student = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=HIDDEN_LAYERS_STUDENT,
            num_classes=NUM_CLASSES,
            return_logits=True
        )
        trainer = ClassifierTrainer(model=student)
        trainer.compile(
            optimizer=torch.optim.Adam(params=student.parameters(), lr=LEARNING_RATE),
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
        student = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=HIDDEN_LAYERS_STUDENT,
            num_classes=NUM_CLASSES,
            return_logits=True
        )
        distiller = Distiller(teacher=teacher, student=student)
        distiller.compile(
            optimizer=torch.optim.Adam(params=student.parameters(), lr=LEARNING_RATE),
            distill_loss_fn=True,
            student_loss_fn=True,
            coeff_dt=COEFF_DT,
            coeff_st=COEFF_ST,
            temperature=TEMPERATURE,
        )
        csv_logger = CSVLogger(filename=f'./logs/{distiller.__class__.__name__}.csv', append=True)
        distiller.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS_DISTILL,
            valloader=dataloader['val'],
            callbacks=[csv_logger],
        )

    for i in range(5):
        expt_mnist(pretrained_teacher=True, skip_baseline=True)