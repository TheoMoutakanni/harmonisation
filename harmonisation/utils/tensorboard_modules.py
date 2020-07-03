

def add_to_writer(writer, data, n_iter, specs):
    for spec in specs:
        if spec['type'] == 'scalar':
            writer.add_scalar(spec['name'], data[spec['name']], n_iter)
        elif spec['type'] == 'histogram':
            writer.add_histogram(
                spec['name'],
                data[spec['name']].detach().cpu().numpy(), n_iter)
        elif spec['type'] == 'embedding':
            writer.add_embedding(data[spec['name']],
                                 metadata=data[spec['label']])
