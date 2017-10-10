
import uuid
import os
import lib.ml_args as ml_args
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

class Derp:
    def __init__(self, args):
        self.args = args
        
    def train(self,eval_file_path, train_file_path):
        """Train a model using the eval and train datasets.
        Args:
          train_file_path: Path to the train dataset.
          eval_file_path: Path to the eval dataset.
        """
        trainer_args = [
            '--output_path', self.args.output_dir,
            '--eval_data_paths', eval_file_path,
            '--eval_set_size', str(self.args.eval_set_size),
            '--train_data_paths', train_file_path
        ]

        if self.args.cloud:
            job_name = 'pix2pix_model' + datetime.datetime.now().strftime(
                '_%y%m%d_%H%M%S')
            command = [
                'gcloud', 'ml-engine', 'jobs', 'submit', 'training', job_name,
                '--stream-logs',
                '--module-name', "trainer.task",
                '--staging-bucket', self.args.gcs_bucket,
                '--region', 'us-central1',
                '--project', self.args.project,
                '--package-path', 'trainer',
                '--runtime-version', self.args.runtime_version,
                '--'
            ] + trainer_args
        else:
            command = [
                'gcloud', 'ml-engine', 'local', 'train',
                '--module-name', "trainer.task",
                '--package-path', 'trainer',
                '--',
            ] + trainer_args
            subprocess.check_call(command)

def main():
    args=ml_args.process_args()

    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml','v1', credentials=credentials)
    pipeline = Derp(args)
    pipeline.train()

if __name__ == '__main__':
    main()
