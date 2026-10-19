"""wdiff — weight-space diff analysis of NLA modules vs. their target model.

Research tooling, deliberately outside `nla/` (uses pandas/matplotlib, which
are not part of the package's dependency contract). Streams individual tensors
over HTTP range requests or via safetensors.safe_open; never loads a full
model into a framework.
"""

__version__ = "0.1.0"
