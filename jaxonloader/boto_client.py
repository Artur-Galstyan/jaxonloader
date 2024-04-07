import boto3
from botocore import UNSIGNED
from botocore.client import Config


class BotoClient:
    @classmethod
    def get(cls):
        return boto3.client(
            service_name="s3",
            config=Config(
                signature_version=UNSIGNED,
                region_name="eu-central-1",
            ),
            endpoint_url="https://eu-central-1.linodeobjects.com",
        )
