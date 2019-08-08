import requests

def download(id, file_to_write):
    drive_url = "https://docs.google.com/uc?export=download"
    req_session = requests.Session()
    session_res = req_session.get(drive_url, params = { 'id' : id }, stream = True)

    for k, v in session_res.cookies.items():
        if k.startswith('download_warning'):
            params = { 'id' : id, 'confirm' : v }
            session_res = session_res.get(drive_url, params = params, stream = True)
            break

    size = 32768
    with open(file_to_write, "wb") as f:
        for packet in session_res.iter_content(size):
            f.write(packet) if packet else None

if __name__ == "__main__":
    xml_file_id = '1DGaKA3Vship_rZ4rRLy1x4fpU6mxBptG'
    mapping_file_id = '1kYqEBwSqRjQUE9Tngul6k50GhC6M7MsB'
    bin_file_id = '12f_fNrt-so9n9dFtsSRRQ57XSLpCq6P6'
    destination = './ssd_inception_v2_mo_fp16/frozen_inference_graph'
    download(xml_file_id, destination + '.xml')
    download(mapping_file_id, destination + '.mapping')
    download(bin_file_id, destination + '.bin')
