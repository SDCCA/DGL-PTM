import pytest
import dgl_ptm
import os
import shutil
from pathlib import Path

os.environ["DGLBACKEND"] = "pytorch"

# a test for running the model
class TestModelRun:
    def test_run(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')

        model.set_model_parameters()
        model.initialize_model()
        model.run()

        # check if wroking directory is created
        work_dir =  Path("./my_model")
        assert work_dir.exists()

        # assert config
        assert model._model_identifier == 'my_model'
        assert model.device == 'cpu'

        # assert that the model has run
        assert model.step_count == 5

        # remove work_dir
        shutil.rmtree(work_dir)
