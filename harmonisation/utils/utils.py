def detach(x, detach=False):
    if detach:
        return x.detach()
    else:
        return x


def compute_modules(input_needed, inputs, nets_dict, modules,
                    detach_input=False):
    if input_needed not in inputs.keys():
        if any([input_needed.split('_')[0] in x for x in modules.keys()]):
            module_name = input_needed.split('_')[0]
            net_inputs = modules[module_name].inputs

            if 'fake' in input_needed:
                net_inputs = [
                    name + '_fake' if name not in ['mask'] else name
                    for name in net_inputs]

            for name in net_inputs:
                inputs = compute_modules(
                    name, inputs, nets_dict, modules)

            inputs[input_needed] = modules[module_name](
                *(detach(inputs[name], detach_input) for name in net_inputs))
            return inputs

        elif input_needed in nets_dict['autoencoder'].outputs:
            net = nets_dict['autoencoder']
            for name in net.inputs:
                inputs = compute_modules(
                    name, inputs, nets_dict, modules)
            inputs.update(net(
                *(detach(inputs[name], detach_input) for name in net.inputs)))
            return inputs
        else:
            for net_name in nets_dict.keys():
                if net_name in input_needed:
                    net = nets_dict[net_name]
                    break

            try:
                net_inputs = net.inputs
            except:
                print(input_needed)
                print(input_needed)

            if 'fake' in input_needed:
                net_inputs = [
                    name + '_fake' if name not in ['mask'] else name
                    for name in net_inputs]
                base_name = '{}_fake_{}'
            else:
                base_name = '{}_{}'

            for name in net_inputs:
                inputs = compute_modules(
                    name, inputs, nets_dict, modules)
            net_pred = net(
                *(detach(inputs[name], detach_input) for name in net_inputs))
            inputs.update(
                {base_name.format(name, net_name): net_pred[name]
                 for name in net_pred.keys()})
            return inputs
    else:
        return inputs
