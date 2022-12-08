from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Union


def prepare_trf_based_model_params(
    model: Union[Module, DistributedDataParallel],
    trf: Module,
    weight_decay: float = 1e-2,
    trf_lr: Optional[float] = 2e-5
):
    if trf_lr is not None:
        # 区别对待Bert的lr和其他部分的lr
        trf_param_id_lst = list(map(id, trf.parameters()))
        other_params = filter(lambda p: id(p[1]) not in trf_param_id_lst, model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': trf.parameters(), 'lr': trf_lr},
            {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optimizer_params