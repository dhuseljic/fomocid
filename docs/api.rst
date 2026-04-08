API Reference
=============

The API reference documents the reusable Python package that sits underneath the
tutorial scripts and notebooks.

The public package surface is organized by subsystem:

- ``fomocid.data`` for benchmark datamodules
- ``fomocid.config_types`` for the normalized YAML config contract
- ``fomocid.ssl`` for SSL model factories and Lightning modules
- ``fomocid.eval`` for feature extraction and downstream metrics
- ``fomocid.analysis`` for projections and nearest-neighbor figures
- ``fomocid.utils`` for config, IO, and notebook helpers

.. toctree::
   :maxdepth: 1

   api/fomocid
   api/fomocid.config_types
   api/fomocid.data
   api/fomocid.ssl
   api/fomocid.eval
   api/fomocid.analysis
   api/fomocid.utils
