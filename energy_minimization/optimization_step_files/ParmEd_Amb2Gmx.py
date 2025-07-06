import parmed as pmd
import os
import glob

amber = pmd.load_file('your_name.top', 'your_name.crd')
amber.save('your_name_gro.top')
amber.save('your_name.gro')

