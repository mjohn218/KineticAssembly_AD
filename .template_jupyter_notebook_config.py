#------------------------------------------------------------------------------
# SLURM script to run the JupyterLab
# The Advanced Research Computing at Hopkins (ARCH)
# Software Team < help@rockfish.jhu.edu >
# Date: Feb, 18 2022
#
# Configuration file for jupyter-notebook.

# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

c.NotebookApp.allow_password_change = True

#c.NotebookApp.keyfile = u'/home/dbass13/.jupyter/ssl/arch_rockfish.key'
#c.NotebookApp.certfile = u'/home/dbass13/.jupyter/ssl/arch_rockfish.pem'

c.NotebookApp.open_browser = False

# Forces users to use a password for the Notebook server.
c.NotebookApp.password = u'sha1:ffd78f3ad55a:4eaf4b2d3dee57b172bc98300dc0198d74fa0a36'

c.NotebookApp.password_required = True
c.NotebookApp.quit_button = True

## The port the notebook server will listen
c.NotebookApp.port_retries = 1

## (sec) Time window used to  check the message and data rate limits.
#c.NotebookApp.rate_limit_window = 3

#  Terminals may also be automatically disabled if the terminado package is not
#  available.
c.NotebookApp.terminals_enabled = True

c.NotebookApp.ip = '*'
c.NotebookApp.port = PORT

