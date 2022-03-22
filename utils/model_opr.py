import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def load_model(model, model_path, strict=True, cpu=False):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model, strict=strict)


def load_model_filter_list(model, model_path, filter_list, strict=True, cpu=False):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)
    # model.load_state_dict(loaded_model, strict=strict)
    loaded_dict = {}
    filtered_names = []
    for name, param in loaded_model.items():
        flag = True
        for filter_name in filter_list:
            if filter_name in name:
                flag = False
                break
        if flag:
            loaded_dict[name] = param
        else:
            filtered_names.append(name)
    # print('Filtered names:', filtered_names)
    model.load_state_dict(loaded_dict, strict=strict)


def load_solver(optimizer, lr_scheduler, solver_path, cpu=False):
    if cpu:
        loaded_solver = torch.load(solver_path, map_location='cpu')
    else:
        loaded_solver = torch.load(solver_path)
    iteration = loaded_solver['iteration']
    optimizer.load_state_dict(loaded_solver['optimizer'])
    lr_scheduler.load_state_dict(loaded_solver['lr_scheduler'])
    del loaded_solver
    torch.cuda.empty_cache()

    return iteration


def save_model(model, model_path):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), model_path)


def save_solver(optimizer, lr_scheduler, iteration, solver_path):
    solver = dict()
    solver['optimizer'] = optimizer.state_dict()
    solver['lr_scheduler'] = lr_scheduler.state_dict()
    solver['iteration'] = iteration
    torch.save(solver, solver_path)
