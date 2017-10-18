import subprocess
import uuid
import argparse
import os

def process_args():
    """Define arguments and assign default values to the ones that are not set.
    Returns:
      args: The parsed namespace with defaults assigned to the flags.
    """
    parser = argparse.ArgumentParser(description='Runs Flowers Sample E2E pipeline.')
    parser.add_argument('--project', default=None, help='The project to which the job will be submitted.')
    parser.add_argument('--cloud', action='store_true', help='Run preprocessing on the cloud.')
    parser.add_argument('--train_input_path', default=None, help='Input specified as uri to CSV file for the train set')
    parser.add_argument('--eval_input_path', default=None, help='Input specified as uri to CSV file for the eval set.')
    parser.add_argument('--eval_set_size', default=50, help='The size of the eval dataset.')
    parser.add_argument('--deploy_model_name', default='pix2pix', help='If --cloud is used, the model is deployed with this name. The default is pix2pix.')
    parser.add_argument('--deploy_model_version', default='v' + uuid.uuid4().hex[:4], help='If --cloud is used, the model is deployed with this version. The default is four random characters.')
    (parser.add_argument('--gcs_bucket', default='pix2pixdata', help='Google Cloud Storage bucket to be used for uploading intermediate data'),)
    parser.add_argument('--output_dir', default=None, help='Google Cloud Storage or Local directory in which to place outputs.')
    parser.add_argument('--runner', default='DataflowRunner', help='See Dataflow runners, may be blocking or not, on cloud or not, etc.')
    parser.add_argument('--temp_location', help='Dataflow GCS temp loc')
    parser.add_argument('--staging_location', help='Dataflow GCS staging loc')
    parser.add_argument('--setup_file', default='./setup.py', help='Dataflow setup.py')
    args, _ = parser.parse_known_args()
    if args.cloud and not args.project:
        args.project = get_cloud_project()
    if not args.staging_location:
        args.staging_location = 'gs://' + args.gcs_bucket + '/staging'
    if not args.temp_location:
        args.temp_location = 'gs://' + args.gcs_bucket + '/temp'
    return args


def get_cloud_project():
    cmd = ['gcloud',
     '-q',
     'config',
     'list',
     'project',
     '--format=value(core.project)']
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            if not res:
                raise Exception('--cloud specified but no Google Cloud Platform project found.\nPlease specify your project name with the --project flag or set a default project: gcloud config set project YOUR_PROJECT_NAME')
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('gcloud is not installed. The Google Cloud SDK is necessary to communicate with the Cloud ML service. Please install and set up gcloud.')
            raise
