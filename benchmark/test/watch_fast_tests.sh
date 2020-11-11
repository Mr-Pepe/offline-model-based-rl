eval "$(conda shell.bash hook)"
conda activate thesis

ptw --clear --config benchmark/test/pytest.ini --runner 'pytest -m fast'