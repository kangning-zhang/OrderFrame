
from matplotlib import gridspec
from pathlib import Path
import pulsepytools.pulsedata.data as pdata


scan_id_list = pdata.select_scans(
    config_path=Path(".").resolve(), config_suffix="myproject", data_suffix="myproject",
    register_dir=Path(".").resolve())

def saveList(myList,filename):
    np.save(filename,myList)

saveList(scan_id_list,'scan.npy')


for i in range(0,len(scan_id_list)):
    scan_id = scan_id_list[i]
    scan = pdata.Scan(scan_id)
    frame_nr = [1997,2000,2003,2006]
    scaling_factor = 4. / 7
    rm_miniframe = False
    for j in range(0,len(frame_nr)):
        local_frame_file = scan.get_scaled_frame_file(frame_nr[j], scaling_factor)
        if local_frame_file.is_file():
            local_frame_file.unlink()
        example_frame_preproc = scan.get_scaled_frame(frame_nr[j], scaling_factor)