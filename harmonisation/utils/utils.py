def detach(x, detach=False):
    if detach:
        return x.detach()
    else:
        return x


def compute_modules(input_needed, inputs, nets_dict, modules, device,
                    detach_input=False):

    if input_needed["name"] in ['mask', 'site']:
        # Just need to move data to device an in the net dictionnary
        tmp = inputs["dataset"][input_needed["name"]]
        inputs[input_needed["net"]][input_needed["name"]] = tmp.to(device)
        return inputs

    elif input_needed["recompute"] or (
            input_needed["name"] not in inputs[input_needed["net"]]):

        if any(input_needed["name"] in m.outputs for m in modules.values()):
            for name, m in modules.items():
                if input_needed["name"] in m.outputs:
                    module_name = name
                    break
            net = modules[module_name]
        elif input_needed["net"] in nets_dict:
            net = nets_dict[input_needed["net"]]
        elif input_needed["net"] == "dataset":
            # Just need to move data to device
            tmp = inputs[input_needed["net"]][input_needed["name"]]
            inputs[input_needed["net"]][input_needed["name"]] = tmp.to(device)
            return inputs

        net_inputs = [{"name": name,

                       "net": "dataset" if (
                           not any(name in m.outputs for m in modules.values())
                       ) and (
                           input_needed["net"] == input_needed["from"]
                       ) else input_needed["from"],

                       "from": input_needed["from"] if (
                           input_needed["net"] != input_needed["from"]
                       ) or (
                           any(name in m.outputs for m in modules.values())
                       ) else "dataset",

                       "detach": input_needed["detach"],
                       "recompute": False}
                      for name in net.inputs]

        # print('1', input_needed)
        # print(net.inputs)

        for net_input in net_inputs:
            inputs = compute_modules(
                net_input, inputs, nets_dict, modules, device)

        # print('2', input_needed)
        # for i in inputs:
        #     for j in inputs[i]:
        #         print(i, j)

        net_preds = net(
            *(detach(inputs[params["from"]][params["name"]],
                     params["detach"]).to(device)
                for params in net_inputs))
        if type(net_preds) == dict:
            inputs[input_needed["net"]].update(net_preds)
        else:
            inputs[input_needed["net"]][input_needed["name"]] = net_preds
        return inputs
    else:
        tmp = inputs[input_needed["net"]][input_needed["name"]]
        inputs[input_needed["net"]][input_needed["name"]] = tmp.to(device)
        return inputs

    # if input_needed not in inputs.keys():
    #     if any([input_needed.split('_')[0] in x for x in modules.keys()]):
    #         module_name = input_needed.split('_')[0]
    #         net_inputs = modules[module_name].inputs

    #         if 'fake' in input_needed:
    #             net_inputs = [
    #                 name + '_fake' if name not in ['mask'] else name
    #                 for name in net_inputs]

    #         for name in net_inputs:
    #             inputs = compute_modules(
    #                 name, inputs, nets_dict, modules)

    #         inputs[input_needed] = modules[module_name](
    #             *(detach(inputs[name], detach_input) for name in net_inputs))
    #         return inputs

    #     elif input_needed in nets_dict['autoencoder'].outputs:
    #         net = nets_dict['autoencoder']
    #         for name in net.inputs:
    #             inputs = compute_modules(
    #                 name, inputs, nets_dict, modules)
    #         inputs.update(net(
    #             *(detach(inputs[name], detach_input) for name in net.inputs)))
    #         return inputs
    #     else:
    #         for net_name in nets_dict.keys():
    #             if net_name in input_needed:
    #                 net = nets_dict[net_name]
    #                 break

    #         net_inputs = net.inputs

    #         if 'fake' in input_needed:
    #             net_inputs = [
    #                 name + '_fake' if name not in ['mask'] else name
    #                 for name in net_inputs]
    #             base_name = '{}_fake_{}'
    #         else:
    #             base_name = '{}_{}'

    #         for name in net_inputs:
    #             inputs = compute_modules(
    #                 name, inputs, nets_dict, modules)
    #         net_pred = net(
    #             *(detach(inputs[name], detach_input) for name in net_inputs))
    #         inputs.update(
    #             {base_name.format(name, net_name): net_pred[name]
    #              for name in net_pred.keys()})
    #         return inputs
    # else:
    #     return inputs
