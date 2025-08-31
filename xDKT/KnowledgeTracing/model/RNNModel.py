# import torch,you can execute all files in google collab , each file is one cell
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig
)
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig

class DKT_xLSTM(nn.Module):
    """Deep Knowledge Tracing with xLSTM"""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT_xLSTM, self).__init__()

        # xLSTM configuration (YAML format in a string)
        xlstm_cfg = """
        mlstm_block:
          mlstm:
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 2
        slstm_block:
          slstm:
            backend: cuda
            num_heads: 2
            conv1d_kernel_size: 4
            bias_init: powerlaw_blockdependent
          feedforward:
            proj_factor: 1.3
            act_fn: gelu
        context_length: 50  # MAX_STEP
        num_blocks: 7
        embedding_dim: 100  # You can adjust this value if needed
        slstm_at: [1]
        """

        # Load and parse the configuration
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))

        # Initialize the xLSTM block stack using the parsed configuration
        self.xlstm_stack = xLSTMBlockStack(cfg)

        # Final output layer (fully connected layer to map to the output dimension)
        self.fc = nn.Linear(cfg.embedding_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Forward pass using xLSTM"""
        # xLSTM expects the input in the form (batch_size, seq_len, features)
        # If the input is in a different shape, you may need to reshape it
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (batch_size, seq_len, features), got {x.shape}")

        # Pass input through the xLSTM stack
        x = self.xlstm_stack(x)  # Shape should be (batch_size, seq_len, embedding_dim)

        # Final fully connected layer (output layer)
        x = self.fc(x)            # Shape should be (batch_size, seq_len, output_dim)

        # Apply sigmoid activation to get the probabilities (between 0 and 1)
        x = self.sig(x)           # Shape should be (batch_size, seq_len, output_dim)

        return x
