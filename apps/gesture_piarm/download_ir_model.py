import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    xml_file_id = '1DGaKA3Vship_rZ4rRLy1x4fpU6mxBptG'
    mapping_file_id = '1kYqEBwSqRjQUE9Tngul6k50GhC6M7MsB'
    bin_file_id = '12f_fNrt-so9n9dFtsSRRQ57XSLpCq6P6'
    destination = './ssd_inception_v2_mo_fp16/frozen_inference_graph'
    download_file_from_google_drive(xml_file_id, destination + '.xml')
    download_file_from_google_drive(mapping_file_id, destination + '.mapping')
    download_file_from_google_drive(bin_file_id, destination + '.bin')
