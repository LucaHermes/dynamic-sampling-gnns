import model.dsgnn_cells
import model.gumbel
import model.state_models
import model.attention_models
import copy

sampling_modules = {
    'gumbel' : model.gumbel.GumbelSampling
}
state_modules = {
    'res'  : model.state_models.ResidualBlock,
}
attention_modules = {
    'gat' : model.attention_models.GATAttention,
    'dot' : model.attention_models.DotAttention,
    'zero' : model.attention_models.ZeroAttention,
    'simple_dot' : model.attention_models.Dot
}
state_cells = {
    'dsgnn' : model.dsgnn_cells.DSGNNCell,
}

def build_module_from_config(module_dict, config):
    if config is None:
        config = {}
    
    config = copy.copy(config)
    module = module_dict.get(config.pop('name', None))

    if module is None:
        return None

    return module(**config)