import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from mlcolvar.data import DictModule, DictLoader
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics
from mlcolvar.utils.trainer import MetricsCallback

from mlcolvar.cvs import MultiTaskCV
from mlcolvar.cvs import AutoEncoderCV
from skewencoder.skewloss import SkewLoss

__all__ = ["skewencoder_model_init", "skewencoder_model_trainer","skewencoder_model_normalization", "cv_eval"]

def skewencoder_model_init(AE_dataset, encoder_layers, loss_coeff, iter=0, PREV_ITER_FOLDER="", **kargs):
    nn_args = {'activation': 'shifted_softplus'}
    optimizer_settings= {'weight_decay': 1e-5}
    options= {'encoder': nn_args, 'decoder': nn_args, 'optimizer': optimizer_settings}
    main_cv = AutoEncoderCV(encoder_layers, options=options)
    aux_loss_fn = SkewLoss()
    subfix = None
    if "subfix" in kargs.keys():
        assert isinstance(kargs["subfix"], str)
        subfix = kargs["subfix"]

    model = MultiTaskCV(main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[loss_coeff])

    if iter > 0:
        if subfix is not None:
            main_cv = model.__class__.load_from_checkpoint(checkpoint_path=f"{PREV_ITER_FOLDER}/checkpoint_{subfix}.ckpt", main_cv=main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[loss_coeff])
        else:
            main_cv = model.__class__.load_from_checkpoint(checkpoint_path=f"{PREV_ITER_FOLDER}/checkpoint.ckpt", main_cv=main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[loss_coeff])
    stat = Statistics()
    stat(AE_dataset['data'])
    model.norm_in.set_from_stats(stat)
    return model


def skewencoder_model_trainer(model, datamodule, iter_folder, **kargs):
    # define callbacks
    metrics = MetricsCallback()
    early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10)
    # define trainer
    trainer = pl.Trainer(accelerator='cpu',callbacks=[metrics, early_stopping], max_epochs=1000, enable_checkpointing=False, enable_model_summary=False, log_every_n_steps=10)
    # fit
    trainer.fit(model,datamodule)
    subfix = None
    if "subfix" in kargs.keys():
        assert isinstance(kargs["subfix"], str)
        subfix = kargs["subfix"]
    if subfix is None:
        trainer.save_checkpoint(f"{iter_folder}/checkpoint.ckpt",weights_only=True)
    else:
        trainer.save_checkpoint(f"{iter_folder}/checkpoint_{subfix}.ckpt",weights_only=True)
    return metrics

def skewencoder_model_normalization(model, dataset, n_components = 1):
    X = dataset[:]['data']
    with torch.no_grad():
        model.postprocessing = None # reset
        s = model(torch.Tensor(X))
    norm =  Normalization(n_components, mode='min_max', stats = Statistics(s) )
    model.postprocessing = norm
    return model

def cv_eval(model, dataset):
    # data here is a torch tensor
    # TODO: extend the adaption of more data structure of data
    # data so far is a n * d matrix of which d indicates the dimension of data
    # TODO: add exception detection for if judgement
    data = dataset["data"]
    output = np.zeros(data.size(dim=1))
    with torch.no_grad():
        train_mode = model.training
        model.eval()
        output = model(data).numpy()
        model.training = train_mode
    return output