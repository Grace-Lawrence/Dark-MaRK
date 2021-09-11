import sys
import astropy.units as u
from nibbler import run_elements
#try:
#    samp_num = int(sys.argv[1])
#except IndexError:
#    samp_num=0
samp_num = 0
save_dir = './test'
#save_dir = '/fred/oz071/glawrence/SAM_VS_ANYL/BS_peak_day/BS_PeakDay/new_results'
run_elements(method="L&S_anyl", samp_num=samp_num,BS_num=1, sigma_cs='Nucleus', min_ee=.01*u.keV, max_ee=36*u.keV, recoil = "Nuclear", save_dir = save_dir)