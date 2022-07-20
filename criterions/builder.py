import criterions
from utils import distributed_ops


def build_criterion(name, args, distributed=False, device=None):
    if name in vars(criterions):
        criterion = criterions.__dict__[name](**args)
    else:
        raise NotImplementedError(f'Criterion {name} not found.')

    criterion = distributed_ops.send_to_device(criterion, distributed=distributed, device=device)
    return criterion
