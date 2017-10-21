yes | sudo pip uninstall myp2p
pip wheel --wheel-dir=wheels myp2p_lib/
sudo pip install wheels/myp2p-0.0.2-py2.py3-none-any.whl
