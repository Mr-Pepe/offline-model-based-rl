eval "$(conda shell.bash hook)"
conda activate thesis

pytest -m "fast or medium" -c "./benchmark/test/pytest.ini"