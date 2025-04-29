# This is really only used if you do the identity stuff
# but we ain't gonna do that because im tired

import os
import urllib.request

def doprep():
    models_folder = "models/"
    embeddings_model = "res10_300x300_ssd_iter_140000.caffemodel"
    deploy_prototxt = "deploy.prototxt"

    embeddings_url = "https://github.com/gopinath-balu/computer_vision/raw/refs/heads/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel"
    deploy_url = "https://github.com/BVLC/caffe/raw/refs/heads/master/models/bvlc_reference_caffenet/deploy.prototxt"

    os.makedirs(models_folder, exist_ok=True)

    if os.path.isfile(models_folder + embeddings_model):
        print("Found face embeddings model... Proceeding.")
    else:
        print("You don't have the face embeddings model, so it will be downloaded for you...")
        urllib.request.urlretrieve(embeddings_url, models_folder + embeddings_model)
        print("Download complete.")

    if not os.path.isfile(models_folder + deploy_prototxt):
        print("You don't have the deploy prototxt, so it will be downloaded for you...")
        urllib.request.urlretrieve(deploy_url, models_folder + deploy_prototxt)
        print("Download complete.")
    else:
        print("Found deploy prototxt... Proceeding.")
