import requests

def main():
    pb_file_id = '1TuK2SMtkFlT5SRrhjWCz4Z5xXE-gcRDd'
    file_to_write = './tensorflow_model/frozen_inference_graph.pb'
    drive_url = "https://docs.google.com/uc?export=download"

    req_session = requests.Session()
    session_res = req_session.get(drive_url, params = { 'id' : pb_file_id }, stream = True)

    for k, v in session_res.cookies.items():
        if k.startswith('download_warning'):
            params = { 'id' : pb_file_id, 'confirm' : v }
            session_res = session_res.get(drive_url, params = params, stream = True)
            break

    size = 32768
    with open(file_to_write, "wb") as f:
        for packet in session_res.iter_content(size):
            f.write(packet) if packet else None

if __name__ == "__main__":
    main()
