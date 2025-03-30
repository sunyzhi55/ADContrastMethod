from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net.TripleNetwork import *
from MDL_Net.MDL_Net import generate_model
from RLAD_Net.taad import get_model
from utils.api import *
from loss_function import joint_loss, loss_in_IMF
from utils.basic import get_scheduler
from Dataset import MriPetDataset, MriPetDatasetWithTowLabel, MriPetDatasetWithTwoInput, MriDataset, GMWMPETDataset

models = {
'ITCFN':{
        'Name': 'Triple_model_CoAttention_Fusion',
        'Model': Triple_model_CoAttention_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'ITFN':{
        'Name': 'Triple_model_Fusion',
        'Model': Triple_model_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'TFN':{
        'Name': 'Triple_model_Fusion_Incomplete',
        'Model': Triple_model_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'TCFN':{
        'Name': 'Triple_model_CoAttention_Fusion_Incomplete',
        'Model': Triple_model_CoAttention_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'HFBSurv': {
        'Name': 'HFBSurv',
        # 'Data': './data/summery_new.txt',
        # 'Batch': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        # 'Dataset_mode': 'fusion',
        'Model': HFBSurv,
        'dataset': MriPetDataset,
        'shape': (96, 128, 96),
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Scheduler': get_scheduler,
        'Run': run_main,
    },
'IMF':{
        'Name': 'Interactive_Multimodal_Fusion_Model',
        'Model': Interactive_Multimodal_Fusion_Model,
        'dataset': MriPetDatasetWithTowLabel,
        'shape': (96, 128, 96),
        'Loss': loss_in_IMF,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_for_IMF,
    },
'MDL':{
        'Name': 'MDL_Net',
        # generate_model(model_depth=18, in_planes=1, num_classes=2)
        'Model': generate_model,
        'dataset': GMWMPETDataset,
        'shape': (96, 128, 96),
        'Loss': CrossEntropyLoss,
        'Optimizer': SGD,
        'batch_size': 8,
        'Lr': 0.001,
        'weight_decay':0.01,
        'momentum':0.9,
        'label_smoothing':0.2,
        'Epoch': 150,
        'Run': run_main_for_MDL,
        'Scheduler': get_scheduler,
},
'RLAD':{
        'Name': 'RLAD_Net',
        # generate_model(model_depth=18, in_planes=1, num_classes=2)
        'Model': get_model,
        'dataset': MriDataset,
        'shape': (128, 128, 128),
        'Loss': CrossEntropyLoss,
        'Optimizer': Adam,
        'batch_size': 2,
        'Lr': 0.001,
        'weight_decay':1e-5,
        'Epoch': 150,
        'Run': run_main_for_RLAD,
        'Scheduler': get_scheduler,
}

}
