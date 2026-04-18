from setuptools import setup, find_packages

setup(
    name="dgl_ptm",
    version="0.1.0",
    description="A poverty trap model plugin for DGL-ABM",
    author="Victoria Garibay et al.",
    packages=find_packages(),
    install_requires=[
        "dgl-abm>=0.1.0",
        "pluggy",
        "pytorch",
        "dgl",
        "numpy",
        "scipy"],
    entry_points={
        "dgl_abm.plugins": [
            "dgl_ptm = dgl_ptm.ptm_plugin.extensions.step_extension",
            "dgl_ptm_agent_update = dgl_ptm.ptm_plugin.extensions.agent_update_extension",
        ],
    },
    python_requires=">=3.8",
)