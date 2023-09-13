from src.conv_onet import models


def get_model(cfg):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.

    Returns:
        decoder (nn.module): the network model.
    """

    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']
    use_view_direction = cfg['model']['use_view_direction']
    decoder = models.decoder_dict['point'](
        cfg=cfg, c_dim=c_dim,
        pos_embedding_method=pos_embedding_method, use_view_direction=use_view_direction)
    return decoder
