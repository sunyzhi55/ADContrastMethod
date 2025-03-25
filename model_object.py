from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net.TripleNetwork import *
from MDL_Net.MDL_Net import generate_model, MDL_Net
from Net.api import *
from loss_function import joint_loss, loss_in_IMF
from utils import get_scheduler

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
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_main
    },
'IMF':{
        'Name': 'Interactive Multimodal Fusion Model',
        'Model': Interactive_Multimodal_Fusion_Model,
        'Loss': loss_in_IMF,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_for_IMF
    },
'MDL':{
        'Name': 'MDL_Net',
        # generate_model(model_depth=18, in_planes=1, num_classes=2)
        'Model': generate_model,
        'Loss': CrossEntropyLoss,
        'Optimizer': SGD,
        'batch_size': 8,
        'Lr': 0.001,
        'weight_decay':0.01,
        'momentum':0.9,
        'label_smoothing':0.2,
        'Epoch': 200,
        'Run': run_main_for_MDL,
        'Scheduler': get_scheduler,
}

}