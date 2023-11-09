import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np
import pandas as pd

"""
code base on https://github.com/sksq96/pytorch-summary
"""


def summary_csv(model, input_size, batch_size=-1, device=None, dtypes=None, path1=None, path2=None):
    """

    Args:
        model:
        input_size: (C, H, W) or (C, D, H, W)
        batch_size:
        device:
        dtypes:
        path1: save path of model summary dataframe (.csv)
        path2: save path of size information dataframe (.csv)

    Returns: model summary dataframe(model_summary), size information dataframe(size_info)

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # dataframe header
    header = ["Layer (type)", "Output Shape", "Param #"]

    total_params = 0
    total_output = 0
    trainable_params = 0
    rows = []
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        row_new = [layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"])]
        rows.append(row_new)
    rows = np.array(rows)
    model_summary = pd.DataFrame(rows, columns=header)
    model_summary.set_index(header[0], inplace=True)
    if path1 is not None:
        model_summary.to_csv(path1)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    data = ["{0:,}".format(total_params), "{0:,}".format(trainable_params),
            "{0:,}".format(total_params - trainable_params), "%0.2f" % total_input_size, "%0.2f" % total_output_size,
            "%0.2f" % total_params_size, "%0.2f" % total_size]
    index = ["Total params", "Trainable params", "Non-trainable params", "Input size (MB)",
             "Forward/backward pass size (MB)", "Params size (MB)", "Estimated Total Size (MB)"]
    size_info = pd.DataFrame(data=data, index=index, columns=['#'])
    if path2 is not None:
        size_info.to_csv(path2)

    # return summary
    return model_summary, size_info
