import torch.optim
import torch.distributed as dist


def build_optimizer(params, cfg):
    if cfg.scale_lr:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        full_batch_size = cfg.batch_size * world_size
        cfg.args.lr *= full_batch_size/cfg.args.base_batch_size

    if cfg.method in vars(torch.optim):
        opt = torch.optim.__dict__[cfg.method](
            params,
            **cfg.args
        )
    elif cfg.method.lower() == 'lars':
        import apex
        opt = torch.optim.SGD(
            params,
            **cfg.args
        )
        opt = apex.parallel.LARC.LARC(
            opt,
            trust_coefficient=0.001,
            clip=False
        )
    else:
        raise NotImplementedError(f'Optimizer {cfg.method} not found.')

    return opt
