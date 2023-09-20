from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
import multiprocessing as mp

if __name__ == "__main__":
    print("Start job")
    mp.set_start_method("forkserver")
    bcm = get_bcm_object("FRA", 'main')
    for p in bcm.priors:
        print(p)
    
    print("success!!!")