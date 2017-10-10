import uuid
import os
import lib.ml_args as ml_args
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

class Herp:
    def __init__(self, cloud, project=None,  **kwargs):
        self.kwargs = kwargs
        self.cloud= str(cloud)
        self.project = str(project)
    def train(self):
        print("cloud=" + self.cloud)
        print("project=" + self.project)
        for key, value in self.kwargs.items():
            print("%s = %s" % (key, value))
         
def main():
    args=ml_args.process_args()
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml','v1', credentials=credentials)
    pipeline = Herp(**vars(args))
    pipeline.train()

if __name__ == '__main__':
    main()
